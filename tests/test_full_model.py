import pytest
import torch
import transformers
from hivemind import get_logger, use_hivemind_log_handler
from test_utils import *

from src.bloom.model import BloomForCausalLM
from src.client.remote_model import DistributedBloomForCausalLM

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


@pytest.mark.forked
def test_full_model_exact_match(atol_forward=1e-3, atol_inference=1e-3):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    assert isinstance(model, DistributedBloomForCausalLM)
    assert len(model.transformer.h) == model.config.n_layer

    test_inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        parallel_outputs = model.forward(test_inputs).logits
        assert torch.all(torch.isfinite(parallel_outputs))
        logger.info("Forward outputs are finite")

        embs = model.transformer.word_embeddings(test_inputs)
        embs = model.transformer.word_embeddings_layernorm(embs)
        recurrent_outputs = []
        with model.transformer.h.inference_session() as sess:
            for t in range(embs.shape[1]):
                recurrent_outputs.append(sess.step(embs[:, t : t + 1, :]))
        recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
        recurrent_outputs = model.transformer.ln_f(recurrent_outputs)
        recurrent_outputs = model.lm_head(recurrent_outputs)
        assert torch.allclose(recurrent_outputs, parallel_outputs, rtol=0, atol=atol_inference)
        logger.info("Inference is consistent with forward")

        del model, embs, recurrent_outputs

        if REF_NAME:
            ref_model = transformers.BloomForCausalLM.from_pretrained(
                REF_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32
            )
            dummy_mask = torch.ones_like(test_inputs, dtype=torch.bool)
            # note: this creates a dummy mask to make the test compatible with older transformer versions
            # prior to https://github.com/huggingface/transformers/pull/17837
            ref_outputs = ref_model.forward(test_inputs, attention_mask=dummy_mask).logits.float()
            assert torch.allclose(ref_outputs, parallel_outputs, rtol=0, atol=atol_forward)
            logger.warning(f"Distributed forward is consistent with {type(ref_model)}.forward")
            del ref_model, ref_outputs, dummy_mask
        else:
            logger.warning("Did not test exact match with local model: REF_NAME environment variable is not set")
            assert False


@pytest.mark.forked
def test_greedy_generation(max_new_tokens=4):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]
    remote_outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
    )
    hf_outputs = BloomForCausalLM.greedy_search(model, input_ids=inputs, max_length=inputs.size(1) + max_new_tokens)
    assert torch.allclose(remote_outputs, hf_outputs), "Greedy search are not identical to HF"

    inputs_batch = tokenizer(["A cat sat on a mat", "A dog sat on a mat"], return_tensors="pt", padding=True)[
        "input_ids"
    ]
    remote_outputs_batch = model.generate(
        inputs_batch,
        max_new_tokens=max_new_tokens,
    )
    hf_outputs_batch = BloomForCausalLM.greedy_search(
        model, input_ids=inputs_batch, max_length=inputs_batch.size(1) + max_new_tokens
    )
    assert torch.allclose(
        remote_outputs_batch, hf_outputs_batch
    ), "Greedy search are not identical to HF in multibatch mode"
