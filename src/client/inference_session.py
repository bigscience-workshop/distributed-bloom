from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, List, Optional

import torch
from hivemind import (
    P2P,
    anext,
    deserialize_torch_tensor,
    get_logger,
    nested_flatten,
    serialize_torch_tensor,
    use_hivemind_log_handler,
)
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import StubBase
from hivemind.proto import runtime_pb2

from src.client.routing import RemoteSequenceManager
from src.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from src.server.handler import TransformerConnectionHandler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteTransformerBlockInferenceSession:
    """
    An interface to a single multi-step *inference* session for a specific remote module on a specific server

    :note: this inference session is *not* fault-tolerant out of the box
    """

    def __init__(self, uid: ModuleUID, rpc_info: RPCInfo, inputs_queue: asyncio.Queue, outputs_aiter: AsyncIterator):
        self.uid, self.rpc_info = uid, rpc_info
        # warning: this code manages async objects that are only usable inside RemoteExpertWorker's background thread;
        # using them in any other EventLoop may cause side-effects including, headaches, diarrhea, and loss of sleep
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.stepped = False
        self.closed = False

    @classmethod
    async def _create(
        cls, stub: StubBase, uid: ModuleUID, rpc_info: RPCInfo, timeout: Optional[float] = None
    ) -> RemoteTransformerBlockInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        inputs_queue = asyncio.Queue()
        outputs_stream = await stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue, timeout), timeout=timeout)
        return cls(uid, rpc_info, inputs_queue, outputs_stream)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, timeout: Optional[float]) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    def step(self, new_hidden_states: torch.Tensor):
        """Inference step: send a chunk of input tensors and receive a chunk of outputs"""
        if self.closed:
            raise Exception("Session is closed, cannot perform step")
        # serialize inputs and put them into the queue
        inputs = (new_hidden_states,)
        outputs_serialized = RemoteExpertWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=[
                        serialize_torch_tensor(tensor, proto.compression)
                        for tensor, proto in zip(inputs, nested_flatten(self.rpc_info["forward_schema"]))
                    ],
                )
            )
        )
        outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
        assert outputs[0].shape == inputs[0].shape, f"expected outputs[0] to be hidden states but got {outputs[0]}"
        return outputs[0]

    async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
        """Inference step on serialized data. This code is meant to be run inside RemoteExpertWorker"""
        await self._inputs_queue.put(inputs_serialized)
        self.stepped = True
        return await anext(self._outputs_stream)

    def close(self):
        """Finish a given inference session, close the underlying connection"""
        if self._outputs_stream is None:
            return  # already closed
        RemoteExpertWorker.run_coroutine(self._aclose_stream())
        self._outputs_stream = self._inputs_queue = None
        self.closed = True

    async def _aclose_stream(self):
        """Close the inference session. This code is meant to be run inside RemoteExpertWorker"""
        if self._outputs_stream is None:
            return  # already closed
        if self.stepped:
            await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
            try:
                await anext(self._outputs_stream)
            except StopAsyncIteration:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, *exc_details):
        self.close()


class RemoteSequentialInferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, p2p: P2P, timeout: Optional[float] = None):
        self.sequence_manager = sequence_manager
        self.p2p = p2p
        self.closed = False
        self.chosen_spans: List[RemoteSpanInfo] = []
        self.stack = contextlib.ExitStack()
        self.inference_sessions: List[RemoteTransformerBlockInferenceSession] = []
        self.timeout = timeout

    def __enter__(self):
        assert not self.closed and not self.chosen_spans
        self.stack.__enter__()
        # TODO(yozh) replace this code with a fault-tolerant chain that can be reconstructed if some peers fail
        self.chosen_spans.extend(self.sequence_manager.make_sequence())

        for chosen_span in self.chosen_spans:
            stub = TransformerConnectionHandler.get_stub(self.p2p, chosen_span.peer_id)
            span_uids: str = CHAIN_DELIMITER.join(self.sequence_manager.block_uids[chosen_span.start : chosen_span.end])
            inference_session = RemoteExpertWorker.run_coroutine(
                RemoteTransformerBlockInferenceSession._create(
                    stub, span_uids, rpc_info=self.sequence_manager.rpc_info, timeout=self.timeout
                )
            )
            self.inference_sessions.append(inference_session)
            self.stack.enter_context(inference_session)

        return self

    def step(self, inputs: torch.Tensor):
        assert not self.closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")
        for session in self.inference_sessions:
            outputs = session.step(inputs)
            assert outputs.shape == inputs.shape, f"expected {inputs.shape}, got {outputs.shape}"
            inputs = outputs
        return inputs

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self.closed:
            self.stack.__exit__(*exc_details or (None, None, None))
            self.inference_sessions.clear()
            self.closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()
