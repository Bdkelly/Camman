import json

import pytest
import serial

from training.reinforcement.verify import main, verify_pipeline


def test_fresh_training_exports_actor_and_sends_real_loopback_commands(tmp_path, mocker):
    factory = serial.serial_for_url
    opened = []

    def loopback_only(url, *args, **kwargs):
        assert url == "loop://", "Verification must never open a physical controller"
        opened.append(url)
        return factory(url, *args, **kwargs)

    mocker.patch("serial.serial_for_url", side_effect=loopback_only)
    mocker.patch("serial.Serial", side_effect=AssertionError("Physical serial is forbidden"))
    output = tmp_path / "verification"
    report = verify_pipeline(output, episodes=6, steps=64)
    assert report["status"] == "passed" and report["optimizer_updates"] > 0
    assert report["actor_parameters"] > 0
    assert report["serial"]["commands"][-1] == "Stop"
    assert any(line.startswith("V:") for line in report["serial"]["commands"])
    assert report["serial"]["lost_target_stop"] and report["serial"]["disabled_stop"]
    assert not report["serial"]["hardware_handshake_tested"]
    assert opened == ["loop://"]
    assert json.loads((output / "verification.json").read_text()) == report


def test_verification_preserves_existing_output_and_reports_error(tmp_path, capsys):
    sentinel = tmp_path / "existing.pth"
    sentinel.write_bytes(b"keep existing model")
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path)])
    assert exc.value.code == 1
    assert "Pipeline verification failed" in capsys.readouterr().err
    assert sentinel.read_bytes() == b"keep existing model"
