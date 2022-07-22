from __future__ import annotations

import enum
import random
import threading
from typing import Collection, Dict, List, Optional, Sequence, Tuple, Union

from hivemind import DHT, P2P, DHTExpiration, MSGPackSerializer
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from src.client.routing.routing_strategy import ALL_ROUTING_STRATEGIES, RoutingStrategyBase
from src.client.routing.sequence_info import RemoteSequenceInfo
from src.data_structures import ModuleUID, RemoteModuleInfo, RemoteSpanInfo, ServerState
from src.dht_utils import get_remote_module_infos
from src.server.handler import TransformerConnectionHandler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequenceManager(threading.Thread):
    """
    Sequence manager is a thread that keeps track of information on remote servers that constitute a RemoteSequential.
    TL;DR it tells you, which peers you should ask to get a specific layer. It is used in RemoteSequential.

    When created, RemoteSequenceManager looks up which servers serve necessary layers by reading from DHT.
    Using this information, sequence manager can form sequences of servers that collectively have the full sequence.

    To form such a sequence, call .make_sequence with the appropriate optimization policy (see make_sequence docstr).

    :param dht: a running hivemind.DHT instance, connected to peers that serve the corresponding blocks
    :param block_uids: a sequence of DHT keys (strings) corresponding to remote layers
    :param p2p: an optional P2P replica (if not specified, create one via dht.replicate_p2p())
    :param update_period: by default, refresh DHT information once in this many seconds

    :note: RemoteSequenceManager takes up some CPU and network I/O to operate in background. It is recommended to avoid
      running redundant sequence managers for the same set of layers.

    Example
    =======
    >>> sequence_manager = RemoteSequenceManager(dht=..., block_uids=('me/my-model.0', 'me/my-model.1', 'me/my-model.2')
    >>> seq1_full_model = sequence_manager.make_sequence()
    >>> seq2_partial = sequence_manager.make_sequence(start_index=0, end_index=2)  # the end index is exclusive
    >>> seq1_fastest = sequence_manager.make_sequence()

    """

    def __init__(
        self,
        dht: DHT,
        block_uids: Sequence[ModuleUID],
        *,
        p2p: Optional[P2P] = None,
        start: bool,
        update_period: float = 30,
        routing_strategies: Dict[str, RoutingStrategyBase] = None,
    ):  # NB: if you add any more parameters, please make sure you pass them to sub-sequences in .__getitem__ below!
        super().__init__(daemon=True)
        self.dht, self.p2p = dht, (p2p if p2p is not None else dht.replicate_p2p())
        self.sequence_info = RemoteSequenceInfo.make_empty(block_uids)  # to be updated in a background thread

        if routing_strategies is None:
            routing_strategies = {key: Strategy(self.sequence_info) for key, Strategy in ALL_ROUTING_STRATEGIES.items()}
        self.routing_strategies = routing_strategies
        self.last_update_time: DHTExpiration = -float("inf")
        self.update_period = update_period

        self._rpc_info = None  # TODO move to RemoteSequenceInfo
        self._lock_changes = threading.Lock()  # TODO move to RemoteSequenceInfo
        self.ready = threading.Event()  # whether or not you are ready to make_sequence
        self.update_()  # TODO replace with background thread and await ready

        if start:
            self.run_in_background()

    def run(self) -> None:
        self.ready.set()
        threading.Event().wait()
        # TODO

    def make_sequence(
        self,
        strategy: Union[None, str, RoutingStrategyBase] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        **kwargs,
    ) -> Sequence[RemoteSpanInfo]:
        """
        Form a sequence of remote servers that collectively serve all consecutive layers

        :param strategy: the routing algorithm to use (e.g. random, fastest, balanced), see routing_strategy.py
        :param start_index: optional index of the first module in a sequence, default = the first of block_uids
        :param end_index: optional index of the last module (non-inclusive), default = after last of block_uids
        :param kwargs: additional keyword arguments, depending on your chosen routing strategy
        """
        assert self.is_alive()
        if not self.ready.is_set():
            logger.warning(f"{self.__class__.__name__} is still initializing, waiting until it's ready...")
            self.ready.wait()
            logger.warning(f"Finished waiting for {self.__class__.__name__} to initialize")
        if strategy is None:
            strategy = next(iter(self.routing_strategies))
        if not isinstance(strategy, RoutingStrategyBase):
            strategy = self.routing_strategies[strategy]
        return strategy.make_sequence(start_index, end_index, **kwargs)

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequenceManager:
        """Get a RemoteSequenceManager for a sub-sequence of blocks"""
        assert isinstance(ix, (int, slice))
        if not isinstance(ix, slice):
            ix = slice(int(ix), int(ix) + 1, 1)

        self.ready.wait()
        with self._lock_changes:
            subseq = RemoteSequenceManager(
                self.dht,
                self.block_uids[ix],
                p2p=self.p2p,
                update_period=self.update_period,
                start=False,
            )  # NB: if you've added more parameters to __init__, please forward them in the instantiation above
            subseq.sequence_info = self.sequence_info[ix]
            subseq._rpc_info = self._rpc_info
            subseq.last_update_time = self.last_update_time
            if self.is_alive():
                subseq.run_in_background()
        return subseq

    def update_(self):
        with self._lock_changes:
            self.sequence_info.update_(self.dht)
            for name, strategy in self.routing_strategies.items():
                strategy.update_()

    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready:
            self.ready.wait(timeout)

    def __len__(self):
        return len(self.block_uids)

    @property
    def block_uids(self) -> Sequence[ModuleUID]:
        return self.sequence_info.block_uids

    @property
    def block_infos(self) -> Sequence[RemoteModuleInfo]:
        return self.sequence_info.block_infos

    @property
    def rpc_info(self):
        """Return the rpc_info queried from one of the servers that hold the first block"""
        if self._rpc_info is None:
            retries = 0
            for i in range(self.max_retries):
                try:
                    self.update_()
                    peer_id = random.choice(list(self.block_infos[0].servers.keys()))
                    stub = TransformerConnectionHandler.get_stub(self.p2p, peer_id)
                    outputs = RemoteExpertWorker.run_coroutine(
                        stub.rpc_info(runtime_pb2.ExpertUID(uid=self.block_uids[0]))
                    )
                    self._rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
                except Exception as e:
                    retries += 1
                    if retries >= self.max_retries:
                        raise e
                    else:
                        logger.warning(f"Tried to call rpc_info, but caught {repr(e)}", exc_info=True)
        return self._rpc_info

    @property
    def max_retries(self) -> int:
        logger.warning(
            "RemoteSequenceManager.max_retries is deprecated and will be removed when dbaranchuk@ implements"
            " chained forward/backward. If you have questions about the roadmap, please ping yozh@ ."
        )
        return 3
