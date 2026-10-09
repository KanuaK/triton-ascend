from types import SimpleNamespace
from unittest.mock import patch
import subprocess

import pytest
from triton.backends.ascend import compiler


@pytest.mark.parametrize("mix,expected_subblocks", [("aiv", False), ("mix", True)])
@pytest.mark.parametrize("lock", [
    "hivm.hir.sync_block_lock_unordered",
    "hivm.hir.sync_block_lock {ordering = #hivm.ordering<unordered>}",
])
def test_unordered_lock_metadata(mix, expected_subblocks, lock):
    ir = f'func.func @kernel() attributes {{mix_mode = "{mix}", parallel_mode = "simd"}} {{{lock}}}'
    _, metadata = compiler._parse_linalg_metadata(ir, {})
    assert metadata["lock_num"] == 1 << 32
    assert metadata["lock_init_val"] == 0
    assert metadata["auto_tile_and_bind_subblock"] == expected_subblocks


@pytest.mark.parametrize("arch,expected", [("Ascend950PR_9579", True), ("Ascend910B2", False)])
def test_noinline_workaround_target(arch, expected):
    assert compiler._needs_lib_call_no_inline({"target": SimpleNamespace(arch=arch)}) == expected


@pytest.mark.parametrize("output,expected", [("--enable-lib-call-no-inline", True), ("old compiler", False)])
def test_noinline_option_capability(output, expected):
    compiler._npu_compiler_supports_option.cache_clear()
    with patch.object(compiler.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
        assert compiler._npu_compiler_supports_option("compiler", "--enable-lib-call-no-inline") == expected
    compiler._npu_compiler_supports_option.cache_clear()


def test_noinline_capability_probe_failure():
    compiler._npu_compiler_supports_option.cache_clear()
    with patch.object(compiler.subprocess, "run", side_effect=subprocess.TimeoutExpired("compiler", 10)):
        assert not compiler._npu_compiler_supports_option("compiler", "--enable-lib-call-no-inline")
    compiler._npu_compiler_supports_option.cache_clear()
