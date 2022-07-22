"""RoutingStrategies are helpers for RemoteSequenceManager (sequence_manager.py) that implement make_sequence"""
import random
from abc import ABC
from typing import List, Optional, Tuple

from src.client.routing.sequence_info import RemoteSequenceInfo
from src.data_structures import RemoteSpanInfo, ServerState


class RoutingStrategyBase(ABC):
    def update_(self):
        """Called when sequence manager fetches new info from the dht"""
        raise NotImplementedError()

    def make_sequence(self, start_index: int = 0, end_index: Optional[int] = None, **kwargs):
        """Form and return a sequence;"""
        raise NotImplementedError()


class RandomRoutingStrategy(RoutingStrategyBase):
    """choose a random compatible server at each branch and include all layers served by it"""

    def __init__(self, sequence_info: RemoteSequenceInfo):
        self.sequence_info = sequence_info
        self.spans_by_priority: List[RemoteSpanInfo] = []  # sorted from best to worst
        self.spans_containing_block: Tuple[List[RemoteSpanInfo], ...] = tuple([] for _ in range(len(sequence_info)))

    def update_(self):
        for uid, info in zip(self.sequence_info.block_uids, self.sequence_info.block_infos):
            assert info is not None, f"Found no remote peers for block {uid}"
            # TODO change this to waiting and warning - instead of crashing the thread :)

        closed_spans = []
        active_spans = {}
        for block_index, info in enumerate(self.sequence_info.block_infos):
            for peer_id, server in info.servers.items():
                if server.state != ServerState.ONLINE:
                    continue
                if peer_id not in active_spans:
                    active_spans[peer_id] = RemoteSpanInfo(start=block_index, end=block_index + 1, peer_id=peer_id)
                else:  # peer_id in active_spans
                    active_spans[peer_id].end = block_index + 1

            for peer_id in list(active_spans.keys()):
                if (
                    peer_id not in info.servers
                    or info.servers[peer_id].state != ServerState.ONLINE
                    or block_index == len(self.sequence_info.block_infos) - 1
                ):
                    closed_spans.append(active_spans.pop(peer_id))
        assert not active_spans

        closed_spans.sort(key=lambda span: span.end - span.start, reverse=True)
        self.spans_by_priority = closed_spans

        spans_containing_block = tuple(list() for _ in range(len(self.sequence_info.block_infos)))
        for span in closed_spans:
            for block_index in range(span.start, span.end):
                spans_containing_block[block_index].append(span)

        self.spans_containing_block = spans_containing_block
        assert self.spans_by_priority and self.spans_containing_block

    def make_sequence(self, start_index: int = 0, end_index: Optional[int] = None, **kwargs):
        assert not kwargs, f"Unexpected kwargs: {kwargs}"
        end_index = end_index if end_index is not None else len(self.sequence_info)
        span_sequence = []
        current_index = start_index
        while current_index < end_index:
            candidate_spans = self.spans_containing_block[current_index]
            chosen_span = random.choice(candidate_spans)
            assert chosen_span.start <= current_index < chosen_span.end
            span_sequence.append(chosen_span)
            current_index = chosen_span.end
        return span_sequence


ALL_ROUTING_STRATEGIES = dict(RANDOM=RandomRoutingStrategy)
