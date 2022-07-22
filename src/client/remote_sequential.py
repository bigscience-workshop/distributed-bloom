from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from hivemind import DHT, P2P, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

import src
from src.client.inference_session import RemoteSequentialInferenceSession
from src.client.remote_block import RemoteTransformerBlock
from src.client.routing.sequence_manager import RemoteSequenceManager
from src.data_structures import UID_DELIMITER
from src.dht_utils import _create_remote_modules_from_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequential(nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: src.DistributedBloomConfig,
        dht: DHT,
        dht_prefix: Optional[str] = None,
        p2p: Optional[P2P] = None,
        sequence_manager: Optional[RemoteSequenceManager] = None,
    ):
        logger.warning(f"{self.__class__.__name__} is in active development; expect adventures")
        super().__init__()
        self.config = config
        self.dht = dht
        self.dht_prefix = dht_prefix or config.dht_prefix
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p()) if p2p is None else p2p

        num_blocks = self.config.n_layer if sequence_manager is None else len(sequence_manager)
        block_uids = tuple(f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(num_blocks))
        if sequence_manager is None:
            logger.debug(f"Creating new sequence manager for block uids: {block_uids}")
            self.sequence_manager = RemoteSequenceManager(dht, block_uids, p2p=self.p2p, start=True)
            self.is_subsequence = False
        else:
            logger.debug(f"Reusing sequence manager with {len(sequence_manager)} modules")
            self.sequence_manager = sequence_manager
            assert isinstance(sequence_manager.block_uids, tuple)
            self.is_subsequence = self.sequence_manager.block_uids != block_uids

    def forward(self, inputs: torch.Tensor):
        assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3 and inputs.shape[-1] == self.config.n_embed
        for block in iter(self):
            for retry_index in range(self.sequence_manager.max_retries):
                try:
                    (outputs,) = block(inputs)
                    assert isinstance(outputs, torch.Tensor)
                    assert outputs.shape == inputs.shape, f"Expected {block} output {inputs.shape}, got {outputs.shape}"
                    inputs = outputs
                    break
                except Exception as e:
                    if retry_index == self.sequence_manager.max_retries - 1:
                        raise e
                    else:
                        logging.debug(f"Caught {e} when running forward for block {block_index}", exc_info=True)
        return inputs

    def __getitem__(self, ix: Union[int, slice]) -> Union[RemoteTransformerBlock, RemoteSequential]:
        assert isinstance(ix, (int, slice))
        if isinstance(ix, int):
            assert 0 <= ix < len(self)
            (module,) = _create_remote_modules_from_infos([self.sequence_manager.block_infos[ix]], self.p2p)
            return module
        else:
            return RemoteSequential(
                self.config,
                self.dht,
                dht_prefix=self.dht_prefix,
                p2p=self.p2p,
                sequence_manager=self.sequence_manager[ix],
            )

    def __iter__(self):
        for block_index in range(len(self)):
            yield self[block_index]

    def __len__(self):
        return len(self.sequence_manager)

    def inference_session(self) -> RemoteSequentialInferenceSession:
        self.sequence_manager.update_()
        return RemoteSequentialInferenceSession(self.sequence_manager, self.p2p)

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"
