import re
import shutil
import subprocess

import pytest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_driver_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def _mock_backend_func(name, *args):
    return f"/* {name}: {args} */"


def _make_metadata():
    return SimpleNamespace(
        workspace_size=0,
        lock_init_value=0,
        lock_num=0,
        bs_task_type=0,
        mix_mode="aiv",
        shared=0,
        compile_on_910_95=False,
        parallel_mode="",
        force_simt_only=False,
        debug=False,
        coalesce_factor=1,
        coalesce_axis=-1,
    )


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_generate_npu_wrapper_src_exposes_triton_launch_kernel(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.generate_npu_wrapper_src(
        constants={},
        signature={0: "*fp32", 1: "*fp32", 2: "i32"},
        metadata=_make_metadata(),
    )

    assert 'void triton_launch_kernel(' in src
    assert 'const void* const* kernel_args, const size_t* arg_sizes, int num_args' in src
    assert 'std::vector<std::vector<char>> copied_kernel_args;' in src
    assert 'std::vector<size_t> launch_arg_sizes;' in src
    assert 'std::vector<char> launch_args(total_size, 0);' in src
    assert 'memcpy(launch_args.data() + grid_offset, &gridX, sizeof(int32_t));' in src


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_generate_npu_wrapper_src_shrinks_coalesced_grid_for_both_launch_paths(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.coalesce_factor = 16
    metadata.coalesce_axis = 1

    src = driver.generate_npu_wrapper_src(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert src.count("gridY = gridY / 16;") == 2


@patch("importlib.util.module_from_spec")
@patch("importlib.util.spec_from_file_location")
@patch.object(driver, "make_npu_launcher_stub", return_value="/tmp/fake_launcher.so")
@patch.object(driver, "generate_npu_wrapper_src", return_value="// wrapper src")
@patch.object(driver, "generate_npu_header_src", return_value="// header src")
def test_npu_launcher_exposes_launcher_so_path(
    mock_header_src,
    mock_wrapper_src,
    mock_launcher_stub,
    mock_spec_from_file_location,
    mock_module_from_spec,
):
    fake_module = SimpleNamespace(launch=object())
    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    mock_module_from_spec.return_value = fake_module
    mock_spec_from_file_location.return_value = fake_spec
    src = SimpleNamespace(
        constants={"input_ptr": 1},
        signature={"input_ptr": "*fp32", "numel": "i32"},
        fn=SimpleNamespace(arg_names=["input_ptr", "numel"]),
    )
    metadata = _make_metadata()

    launcher = driver.NPULauncher(src, metadata)

    assert launcher.so_launcher_path == "/tmp/fake_launcher.so"
    assert mock_launcher_stub.call_count == 1
    assert launcher.get_launcher_so_path() == "/tmp/fake_launcher.so"
    assert mock_header_src.call_count == 1
    assert mock_wrapper_src.call_count == 1
    assert mock_launcher_stub.call_count == 1
    mock_wrapper_src.assert_called_with(
        {0: 1},
        {0: "*fp32", 1: "i32"},
        metadata,
    )
    mock_launcher_stub.assert_called_with("// header src", "// wrapper src", False)


@pytest.mark.parametrize("layout,blocks,mix,subblocks,expected_size,headers", [
    (8, 2, "aiv", False, 64, {}),  # Legacy ordered lock, low word is i64 slots.
    (1 << 32, 2, "aiv", False, 320, {0: 2}),
    (1 << 32, 2, "mix", False, 320, {0: 2}),
    (1 << 32, 2, "mix", True, 576, {0: 4}),
    ((2 << 32) | 16, 4, "aiv", False, 1280, {16: 4, 88: 4}),
])
def test_sync_lock_layout_in_both_launch_paths(tmp_path, layout, blocks, mix, subblocks, expected_size, headers):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("C++ compiler is required for the generated launcher check")
    metadata = _make_metadata()
    metadata.lock_num = layout
    metadata.mix_mode = mix
    metadata.auto_tile_and_bind_subblock = subblocks
    with patch.object(driver, "NPUUtils") as utils, \
         patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False), \
         patch.object(driver, "force_disable_ffts", return_value=False), \
         patch.object(driver, "is_ffts_supported", return_value=True), \
         patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend950PR_9579"), \
         patch.object(driver, "get_backend_func", side_effect=_mock_backend_func):
        utils.return_value.get_aivector_core_num.return_value = 56
        utils.return_value.get_aicore_num.return_value = 28
        source = driver.generate_npu_wrapper_src({}, {0: "*i32"}, metadata)
    # Execute the allocation and header initialization emitted into BOTH entry
    # points. The expected values are the NPU-IR wire protocol, not a second
    # copy of the launcher's layout calculation.
    layouts = re.findall(r"const uint64_t orderedLockI64 =.*?syncBlockLockSize = .*?;", source, re.S)
    initializers = re.findall(r"std::vector<int64_t> lockInitData.*?\n    }", source, re.S)
    assert len(layouts) == len(initializers) == 2
    checks = []
    for layout_source, initializer in zip(layouts, initializers):
        checks.append("{\n" + layout_source + "\n" + initializer +
                      f"\nif (syncBlockLockSize != {expected_size}) return 1;\n")
        for i in range(expected_size // 8):
            checks.append(f"if (lockInitData[{i}] != {headers.get(i, 0)}) return 2;\n")
        checks.append("}\n")
    cpp = tmp_path / "lock_layout.cpp"
    cpp.write_text("#include <cstdint>\n#include <vector>\nint main() {\n" + f"uint32_t blockNum = {blocks};\n" +
                   "".join(checks) + "}\n")
    binary = tmp_path / "lock_layout"
    subprocess.run([cxx, "-std=c++17", str(cpp), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)
