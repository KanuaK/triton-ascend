import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _contended_atomic_and(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + tl.program_id(0) * BLOCK + offsets)
    tl.atomic_and(out + offsets, value, sem="acq_rel")


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32])
@pytest.mark.parametrize("block", [15, 2159])
def test_atomic_and_preserves_all_block_updates(dtype, block):
    # Each of eight blocks clears a different bit in the same destination.
    # Missing any participant's update is observable, with no random inputs.
    participants = 8
    masks = ~(1 << torch.arange(participants, dtype=torch.int64))
    x = masks[:, None].expand(participants, block).contiguous().to(dtype).npu()
    output = torch.empty((block, ), dtype=dtype, device="npu")
    expected = torch.full((block, ), -256, dtype=torch.int64).to(dtype)
    for _ in range(10):
        output.fill_(-1)
        _contended_atomic_and[(participants, )](x, output, block)
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), expected)
