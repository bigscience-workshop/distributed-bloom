from __future__ import annotations

import multiprocessing as mp
import random
import threading
import time
from typing import Dict, Optional, List, Sequence, Union

import torch
from hivemind import DHT, MAX_DHT_TIME_DISCREPANCY_SECONDS, BatchTensorDescriptor, get_dht_time
from hivemind.moe.server.layers import add_custom_models_from_file
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from src import BloomConfig, declare_active_modules
from src.bloom.from_pretrained import DTYPE_MAP, load_pretrained_block
from src.data_structures import CHAIN_DELIMITER, UID_DELIMITER, ServerState
from src.dht_utils import get_remote_module_infos
from src.server.backend import TransformerBackend
from src.server.block_selection import choose_best_blocks
from src.server.cache import MemoryCache
from src.server.handler import TransformerConnectionHandler
from src.server.throughput import get_host_throughput

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class Server(threading.Thread):
    """
    Runs Server, periodically checks that the network is balanced,
    restarts the Server with other layers if the imbalance is significant
    """

    def __init__(
        self,
        prefix: Optional[str],
        converted_model_name_or_path: str,
        throughput: Union[float, str],
        num_blocks: Optional[int] = None,
        block_indices: Optional[str] = None,
        num_handlers: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 4096,
        torch_dtype: Union[str, torch.dtype] = "auto",
        cache_size_bytes: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        initial_peers: Sequence[str] = (),
        compression: CompressionType = CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        custom_module_path: Optional[str] = None,
        update_period: float = 30,
        expiration: Optional[float] = None,
        max_block_selection_delay: float = 1,
        use_auth_token: Optional[str] = None,
        *,
        start: bool,
        **kwargs,
    ):
        """Create a server with one or more bloom blocks. See run_server.py for documentation."""

        super().__init__()

        self.converted_model_name_or_path = converted_model_name_or_path
        self.num_handlers = num_handlers
        self.min_batch_size, self.max_batch_size = min_batch_size, max_batch_size
        self.compression = compression
        self.stats_report_interval, self.update_period = stats_report_interval, update_period
        self.use_auth_token = use_auth_token

        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)

        if prefix is None:
            prefix = converted_model_name_or_path
            assert UID_DELIMITER not in prefix and CHAIN_DELIMITER not in prefix, (
                f"Cannot use model name as prefix (contains '{UID_DELIMITER}' or '{CHAIN_DELIMITER}'); "
                f"Please specify --prefix manually when starting a server"
            )
            logger.info(f"Automatic dht prefix: {prefix}")
        self.prefix = prefix

        assert (block_indices is None) != (num_blocks is None), "please specify num_blocks or block_indices, not both"

        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.expiration = expiration

        self.dht = DHT(initial_peers=initial_peers, start=True, **kwargs)
        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.memory_cache = MemoryCache(device, cache_size_bytes)

        assert isinstance(throughput, float) or throughput in ["auto", "eval"]
        if throughput in ["auto", "eval"]:
            throughput = get_host_throughput(device, force_eval=(throughput == "eval"))
        self.throughput = throughput

        if isinstance(torch_dtype, str):
            torch_dtype = DTYPE_MAP[torch_dtype]
        assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
        self.torch_dtype = torch_dtype

        self.block_config = BloomConfig.from_pretrained(converted_model_name_or_path, use_auth_token=use_auth_token)

        if block_indices is not None:
            try:
                first_block_index, last_block_index = block_indices.split(":")
                first_block_index, last_block_index = map(int, map(str.strip, (first_block_index, last_block_index)))
            except Exception as e:
                logger.error(f"Failed to parse --block_indices ({e}), must be start:end (e.g. 0:18)")
                raise
            block_indices = range(first_block_index, last_block_index)
        else:
            # If multiple servers (e.g., launched on the same machine by a script) get to this line at the same time,
            # this delay decreases the probability of a race condition while choosing the best blocks to serve.
            time.sleep(random.random() * max_block_selection_delay)

            assert num_blocks is not None
            uids = [f"{prefix}.{block_index}" for block_index in range(self.block_config.n_layer)]
            module_infos = get_remote_module_infos(self.dht, uids, expiration_time=float("inf"))
            block_indices = choose_best_blocks(num_blocks, module_infos)
        self.block_indices = block_indices

        self.stop = threading.Event()
        if start:
            self.start()

    def run(self):
        self.module_container = ModuleContainer.create(
            dht=self.dht,
            prefix=self.prefix,
            converted_model_name_or_path=self.converted_model_name_or_path,
            block_config=self.block_config,
            memory_cache=self.memory_cache,
            throughput=self.throughput,
            block_indices=self.block_indices,
            num_handlers=self.num_handlers,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            torch_dtype=self.torch_dtype,
            device=self.device,
            compression=self.compression,
            stats_report_interval=self.stats_report_interval,
            update_period=self.update_period,
            expiration=self.expiration,
            use_auth_token=self.use_auth_token,
            start=True,
        )
        try:
            self.stop.wait()
        finally:
            self.module_container.shutdown()

    def shutdown(self):
        self.stop.set()

        self.dht.shutdown()
        self.dht.join()


class ModuleContainer(threading.Thread):
    """Serves a set of specific Bloom layers for inference, forward, and backward. Announces itself over the DHT."""

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, TransformerBackend],
        *,
        device: torch.device,
        num_connection_handlers: int,
        throughput: float,
        update_period: float,
        expiration: Optional[float] = None,
        start: bool,
        **kwargs,
    ):
        super().__init__()

        self.dht, self.module_backends = dht, module_backends
        self.throughput, self.update_period, self.expiration = throughput, update_period, expiration
        self.conn_handlers = [
            TransformerConnectionHandler(dht, self.module_backends) for _ in range(num_connection_handlers)
        ]
        self.runtime = Runtime(self.module_backends, device=device, **kwargs)
        self.dht_handler_thread = ModuleAnnouncerThread(
            self.module_backends,
            dht,
            throughput=throughput,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        self.checkpoint_saver = None  # no need to save checkpoints since we do not change model state

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Runs ModuleContainer in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        logger.info(f"Serving {len(self.module_backends)} blocks:")
        for expert_name, backend in self.module_backends.items():
            num_parameters = sum(p.numel() for p in backend.module.parameters() if p.requires_grad)
            logger.info(f"{expert_name}: {backend.module.__class__.__name__}, {num_parameters} parameters")

        if not self.dht.is_alive():
            self.dht.run_in_background(await_ready=True)

        if self.module_backends:
            self.dht_handler_thread.start()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

        for process in self.conn_handlers:
            if not process.is_alive():
                process.start()
            process.ready.result()

        try:
            self.runtime.run()
        finally:
            self.shutdown()

    # noinspection PyMethodOverriding
    @classmethod
    def create(
        cls,
        *,
        dht: DHT,
        prefix: str,
        converted_model_name_or_path: str,
        block_config: BloomConfig,
        memory_cache: MemoryCache,
        throughput: float,
        block_indices: List[int],
        num_handlers: Optional[int],
        min_batch_size: int,
        max_batch_size: int,
        torch_dtype: torch.dtype,
        device: Union[str, torch.device],
        compression: CompressionType,
        stats_report_interval: Optional[int],
        update_period: float,
        expiration: Optional[float],
        use_auth_token: Optional[str],
        start: bool,
    ) -> ModuleContainer:
        module_uids = [f"{prefix}.{block_index}" for block_index in block_indices]
        declare_active_modules(
            dht,
            module_uids,
            expiration_time=get_dht_time() + expiration,
            state=ServerState.JOINING,
            throughput=throughput,
        )
        logger.info(f"Announced that blocks {block_indices} are joining")

        blocks = {}
        for module_uid, block_index in zip(module_uids, block_indices):
            block = load_pretrained_block(
                converted_model_name_or_path,
                block_index,
                block_config,
                torch_dtype=torch_dtype,
                use_auth_token=use_auth_token,
            )
            for param in block.parameters():
                param.requires_grad = False

            blocks[module_uid] = TransformerBackend(
                module_uid,
                block,
                memory_cache=memory_cache,
                args_schema=(BatchTensorDescriptor(1, 2048, block_config.hidden_size, compression=compression),),
                kwargs_schema={},
                outputs_schema=(BatchTensorDescriptor(1, 2048, block_config.hidden_size, compression=compression),),
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )

        return cls(
            dht,
            blocks,
            throughput=throughput,
            num_connection_handlers=num_handlers,
            device=device,
            stats_report_interval=stats_report_interval,
            update_period=update_period,
            expiration=expiration,
            start=start,
        )

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts ModuleContainer in a background thread. if await_ready, this method will wait until the container
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("ModuleContainer didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """
        An event (multiprocessing.Event) that is set when the container is ready to process requests.

        Example
        =======
        >>> container.start()
        >>> container.ready.wait(timeout=10)
        >>> print("Container ready" if container.ready.is_set() else "Container didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def shutdown(self):
        """
        Gracefully terminate the container, process-safe.
        Please note that terminating container otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        if self.module_backends:
            self.dht_handler_thread.stop.set()
            self.dht_handler_thread.join()

            declare_active_modules(
                self.dht,
                self.module_backends.keys(),
                expiration_time=get_dht_time() + self.expiration,
                state=ServerState.OFFLINE,
                throughput=self.throughput,
            )
            logger.info(f"Announced that blocks {list(self.module_backends.keys())} are offline")

        self.ready.clear()

        for process in self.conn_handlers:
            process.terminate()
            process.join()
        logger.debug("Connection handlers terminated")

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

        logger.debug(f"Shutting down runtime")
        self.runtime.shutdown()

        logger.info("Module container shut down succesfully")


class ModuleAnnouncerThread(threading.Thread):
    """Periodically announces that this container hosts the specified modules, visible to all DHT peers"""

    def __init__(
        self,
        module_backends: Dict[str, TransformerBackend],
        dht: DHT,
        *,
        throughput: float,
        update_period: float = 30,
        expiration: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.module_backends = module_backends
        self.dht = dht
        self.throughput = throughput
        self.update_period = update_period
        self.expiration = expiration
        self.stop = threading.Event()

    def run(self) -> None:
        while True:
            declare_active_modules(
                self.dht,
                self.module_backends.keys(),
                expiration_time=get_dht_time() + self.expiration,
                state=ServerState.ONLINE,
                throughput=self.throughput,
            )
            if self.stop.wait(self.update_period):
                break
