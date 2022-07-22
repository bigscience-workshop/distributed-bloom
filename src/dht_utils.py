"""
Utilities for declaring and retrieving active model layers using a shared DHT.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

from hivemind.dht import DHT, DHTNode, DHTValue
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P, PeerID
from hivemind.utils import DHTExpiration, MPFuture, get_dht_time, get_logger, use_hivemind_log_handler

import src
from src.data_structures import CHAIN_DELIMITER, UID_DELIMITER, ModuleUID, RemoteModuleInfo, ServerInfo, ServerState

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


def declare_active_modules(
    dht: DHT,
    uids: Sequence[ModuleUID],
    expiration_time: DHTExpiration,
    state: ServerState,
    throughput: float,
    wait: bool = True,
) -> Union[Dict[ModuleUID, bool], MPFuture[Dict[ModuleUID, bool]]]:
    """
    Declare that your node serves the specified modules; update timestamps if declared previously

    :param uids: a list of module ids to declare
    :param wait: if True, awaits for declaration to finish, otherwise runs in background
    :param throughput: specify your performance in terms of compute throughput
    :param expiration_time: declated modules will be visible for this many seconds
    :returns: if wait, returns store status for every key (True = store succeeded, False = store rejected)
    """
    if isinstance(uids, str):
        uids = [uids]
    if not isinstance(uids, list):
        uids = list(uids)
    for uid in uids:
        assert isinstance(uid, ModuleUID) and UID_DELIMITER in uid and CHAIN_DELIMITER not in uid
    return dht.run_coroutine(
        partial(
            _declare_active_modules,
            uids=uids,
            expiration_time=expiration_time,
            state=state,
            throughput=throughput,
        ),
        return_future=not wait,
    )


async def _declare_active_modules(
    dht: DHT,
    node: DHTNode,
    uids: List[ModuleUID],
    expiration_time: DHTExpiration,
    state: ServerState,
    throughput: float,
) -> Dict[ModuleUID, bool]:
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    return await node.store_many(
        keys=uids,
        subkeys=[dht.peer_id.to_base58()] * len(uids),
        values=[(state.value, throughput)] * len(uids),
        expiration_time=expiration_time,
        num_workers=num_workers,
    )


def get_remote_module(
    dht: DHT,
    uid_or_uids: Union[ModuleUID, List[ModuleUID]],
    expiration_time: Optional[DHTExpiration] = None,
    return_future: bool = False,
) -> Union[List[Optional[src.RemoteTransformerBlock]], MPFuture[List[Optional[src.RemoteTransformerBlock]]]]:
    """
    :param uid_or_uids: find one or more modules with these ids from across the DHT
    :param expiration_time: if specified, return modules that expire no sooner than this (based on get_dht_time)
    :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
    :returns: a list of [RemoteTransformerBlock if found else None]
    """
    single_uid = isinstance(uid_or_uids, ModuleUID)
    uids = [uid_or_uids] if single_uid else uid_or_uids
    infos = dht.run_coroutine(
        partial(_get_remote_module_infos, uids=uids, expiration_time=expiration_time), return_future
    )

    if return_future:

        async def _unpack(infos_future: MPFuture, dht: DHT):
            p2p = await dht.replicate_p2p()
            modules = _create_remote_modules_from_infos(await infos_future, p2p)
            return modules[0] if single_uid else modules

        return RemoteExpertWorker.run_coroutine(_unpack(infos, dht), return_future)
    p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())
    modules = _create_remote_modules_from_infos(infos, p2p)
    return modules[0] if single_uid else modules


def get_remote_module_infos(
    dht: DHT,
    uid_or_uids: Union[ModuleUID, Sequence[ModuleUID]],
    expiration_time: Optional[DHTExpiration] = None,
) -> List[Optional[RemoteModuleInfo]]:
    single_uid = isinstance(uid_or_uids, ModuleUID)
    uids = [uid_or_uids] if single_uid else uid_or_uids
    infos = dht.run_coroutine(
        partial(_get_remote_module_infos, uids=uids, expiration_time=expiration_time), return_future=False
    )
    return infos[0] if single_uid else infos


async def _get_remote_module_infos(
    dht: DHT, node: DHTNode, uids: Sequence[ModuleUID], expiration_time: Optional[DHTExpiration]
) -> List[Optional[RemoteModuleInfo]]:
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: Dict[ModuleUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    modules: List[Optional[RemoteModuleInfo]] = [None] * len(uids)
    for i, uid in enumerate(uids):
        metadata = found[uid]
        if metadata is None or not isinstance(metadata.value, dict):
            if metadata is not None:
                logger.error(f"Incorrect metadata for {uid}: {metadata}")
            continue
        servers = {}
        for peer_id, server_info in metadata.value.items():
            try:
                peer_id = PeerID.from_base58(peer_id)
                state, throughput = server_info.value
                if not (
                    isinstance(state, int)
                    and isinstance(throughput, float)
                    and math.isfinite(throughput)
                    and throughput >= 0.0
                ):
                    raise ValueError(f"Invalid server info: {server_info}")
                servers[peer_id] = ServerInfo(ServerState(state), throughput)
            except (TypeError, ValueError) as e:
                logger.error(f"Incorrect peer entry for uid={uid}, peer_id={peer_id}: {e}")
        if servers:
            modules[i] = RemoteModuleInfo(uid, servers)
    return modules


def _create_remote_modules_from_infos(
    infos: Sequence[Optional[RemoteModuleInfo]], p2p: P2P
) -> List[Optional[src.RemoteTransformerBlock]]:
    modules: List[Optional[src.RemoteTransformerBlock]] = []
    for info in infos:
        if info is not None:
            modules.append(src.RemoteTransformerBlock(info, p2p))
        else:
            modules.append(None)
    return modules
