import dataclasses
from typing import Iterable, Tuple, Type, TypeVar

from hivemind import DHT, get_logger, use_hivemind_log_handler

from src.data_structures import ModuleUID, RemoteModuleInfo
from src.dht_utils import get_remote_module_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class RemoteSequenceInfo:
    """
    A dataclass that stores general information about which servers hold any given layer;
    - updated by RemoteSequenceManager in a background thread
    - accessed by routing strategies in .on_update

    :note: this class should *not* be modified by RoutingStrategy.on_update to avoid interference between strategies;
     Any metadata specific to one routing strategy, it should be stored inside that strategy. Any information that
     is used by most routing strategies should be moved from said strategies to this class.

    """

    block_uids: Tuple[ModuleUID, ...]
    block_infos: Tuple[RemoteModuleInfo, ...]  # note: the contents of RemoteModuleInfo can and will be updated

    @classmethod
    def make_empty(cls: Type[T], block_uids: Iterable[ModuleUID]) -> T:
        block_uids = tuple(block_uids)
        empty_block_infos = tuple(RemoteModuleInfo(uid, dict()) for uid in block_uids)
        return cls(block_uids, empty_block_infos)

    def __getitem__(self, ix: slice):
        assert isinstance(ix, slice)
        return RemoteSequenceInfo(self.block_uids[ix], self.block_infos[ix])

    def __len__(self):
        return len(self.block_uids)

    def update_(self, dht: DHT):
        new_block_infos = get_remote_module_infos(dht, self.block_uids, expiration_time=float("inf"))
        assert len(new_block_infos) == len(self.block_uids)
        for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
            if info is None:
                logger.warning(f"Found no block info for block {uid}")
                continue
            if not isinstance(info, RemoteModuleInfo):
                logger.warning(f"Unexpected dht entry type for {uid}: {info}")
                continue
            if not info.servers:
                logger.warning(f"Found no active peers for block {uid}")
                continue
            if info.uid != uid:
                logger.warning(f"The DHT entry for {uid} actually points to {info.uid}")
                continue
            self.block_infos[block_index].servers = info.servers
