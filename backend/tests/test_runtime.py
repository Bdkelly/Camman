import pytest

from backend.runtime import resolve_runtime


def test_auto_profile_detects_device_tree_not_cpu_architecture(mocker):
    mocker.patch(
        "backend.runtime.jetson_model", return_value="NVIDIA Jetson Orin Nano Developer Kit"
    )
    mocker.patch("backend.runtime.torch.cuda.is_available", return_value=True)
    runtime = resolve_runtime()
    assert runtime.profile == "jetson" and runtime.precision == "fp16"
    assert runtime.detector_size == 640 and runtime.cpu_threads == 2
    assert runtime.preview_fps == runtime.inference_fps == 15
    assert resolve_runtime(profile="standard").precision == "fp32"
    assert resolve_runtime(profile="standard").detector_size == 800


def test_cpu_fallback_and_explicit_cuda_failure(mocker):
    mocker.patch("backend.runtime.torch.cuda.is_available", return_value=False)
    runtime = resolve_runtime(profile="jetson")
    assert runtime.device.type == "cpu" and runtime.precision == "fp32"
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        resolve_runtime(profile="jetson", device="cuda")


@pytest.mark.parametrize(
    "options",
    [
        {"precision": "fp16"},
        {"detector_size": 639},
        {"detector_size": 0},
        {"preview_fps": 0},
        {"inference_fps": -1},
        {"cpu_threads": 0},
        {"preview_fps": float("nan")},
    ],
)
def test_invalid_runtime_options_fail_early(options):
    with pytest.raises(ValueError):
        resolve_runtime(device="cpu", **options)


def test_thread_limits_are_deployment_only(mocker):
    torch_threads = mocker.patch("backend.runtime.torch.set_num_threads")
    cv_threads = mocker.patch("backend.runtime.cv2.setNumThreads")
    resolve_runtime(profile="standard", device="cpu").apply()
    torch_threads.assert_not_called()
    cv_threads.assert_not_called()
    resolve_runtime(profile="jetson", device="cpu", cpu_threads=3).apply()
    torch_threads.assert_called_once_with(3)
    cv_threads.assert_called_once_with(1)
