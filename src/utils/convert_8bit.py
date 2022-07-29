import bitsandbytes as bnb
import torch


def replace_8bit_linear(model, threshold=6.0):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `GPT3.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes-cudaXXX` with `XXX` is your CUDA version (e.g., 11.6 = 116)
    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        threshold (`float`, *optional*):
            `int8_threshold` for outlier detection as described in the formentioned paper. This parameters is set to
            `6.0` as described by the paper.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold)

        if isinstance(module, torch.nn.Linear) and n != "lm_head":
            model._modules[n] = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            ).to(model._modules[n].weight.device)
    return model
