"""Real API/service control lifecycle with fake camera data and a recording serial sink."""

import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.capture import CapturedFrame
from backend.runtime import resolve_runtime
from backend.web.api import create_app
from backend.web.service import CameraConfig, CameraService, ControlError

TOKEN = "test-pairing-key-at-least-24-characters"
ASSETS = Path(__file__).resolve().parents[2] / "frontend/web/static"


class Serial:
    def __init__(self):
        self.commands = []
        self.closed = False

    def write(self, payload):
        self.commands.append(payload.decode().strip())
        return len(payload)

    def close(self):
        self.closed = True


@pytest.fixture
def rig():
    now = [100.0]
    service = CameraService(CameraConfig(), resolve_runtime(device="cpu"), clock=lambda: now[0])
    service.ready, service.phase = True, "ready"
    service.ser = Serial()
    service.detector = object()
    return service, now


def test_manual_expiry_and_exclusive_lease(rig):
    service, now = rig
    owner = service.acquire()["lease"]
    with pytest.raises(ControlError):
        service.acquire()
    service.command("right", owner)
    assert service.ser.commands[-1] == "V:0.2500"
    now[0] += 0.21
    service.tick()
    assert service.ser.commands[-1] == "Stop"
    service.command("start", owner)
    now[0] += 2
    service.tick()
    assert not service.tracking and service.lease is None
    assert service.ser.commands[-1] == "Stop"
    with pytest.raises(ControlError):
        service.command("start", owner)


def test_stop_revokes_pending_commands_and_late_detections(rig):
    service, now = rig
    owner = service.acquire()["lease"]
    service.command("start", owner)
    old_generation = service.generation
    service.command("stop")  # A paired observer can always stop.
    with pytest.raises(ControlError):
        service.command("right", owner)
    owner = service.acquire()["lease"]
    service.command("start", owner)
    now[0] += 0.11  # The controller enforces its decision cadence after Stop.
    packet = CapturedFrame(np.zeros((100, 100, 3), np.uint8), now[0])
    boxes = [{"box": (75, 45, 85, 55), "score": 1}]
    service.apply_detection(boxes, packet, old_generation)
    assert service.ser.commands[-1] == "Stop"
    service.apply_detection(boxes, packet, service.generation)
    assert service.ser.commands[-1].startswith("V:")
    service.release(owner)
    assert service.ser.commands[-1] == "Stop" and not service.tracking


def test_heartbeat_cannot_reanimate_expired_lease(rig):
    service, now = rig
    owner = service.acquire()["lease"]
    now[0] += 1
    service.heartbeat(owner)
    now[0] += 1.1
    service.tick()
    assert service.lease == owner
    now[0] += 1
    with pytest.raises(ControlError):
        service.heartbeat(owner)
    assert service.lease is None


def test_control_and_stop_do_not_wait_for_detector(rig):
    service, now = rig
    owner = service.acquire()["lease"]
    service.command("start", owner)
    generation = service.generation
    began, finish = threading.Event(), threading.Event()

    def slow_result():
        began.set()
        finish.wait(2)
        packet = CapturedFrame(np.zeros((100, 100, 3), np.uint8), now[0])
        service.apply_detection([{"box": (70, 40, 80, 50)}], packet, generation)

    worker = threading.Thread(target=slow_result)
    worker.start()
    assert began.wait(1)
    try:
        service.command("stop")
        assert worker.is_alive()
        assert service.ser.commands[-1] == "Stop"
    finally:
        finish.set()
        worker.join()
    assert service.ser.commands[-1] == "Stop"


def test_api_authentication_origin_validation_and_control(rig, monkeypatch):
    service, _ = rig
    monkeypatch.setattr(service, "start", lambda: None)
    with TestClient(create_app(service, TOKEN, ASSETS)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/preview.jpg").status_code == 401
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        assert client.get("/api/status").json()["ready"]
        assert (
            client.post("/api/control/acquire", headers={"Origin": "http://evil.test"}).status_code
            == 403
        )
        lease = client.post("/api/control/acquire").json()["lease"]
        assert client.post("/api/control/acquire").status_code == 409
        assert client.post("/api/control/command", json={"action": "start"}).status_code == 403
        client.headers["X-Control-Lease"] = lease
        assert client.post("/api/control/heartbeat").status_code == 200
        assert (
            client.post("/api/control/command", json={"action": "raw", "value": "V:1"}).status_code
            == 422
        )
        assert client.post("/api/control/command", json={"action": "start"}).status_code == 200
        assert service.tracking
        assert client.post("/api/control/command", json={"action": "stop"}).status_code == 200
        assert not service.tracking
        assert client.post("/api/control/heartbeat").status_code == 403
        assert "lease" not in client.get("/api/status").json()
    assert service.halt.is_set()


def test_preview_freshness_and_missing_model(rig, monkeypatch):
    service, now = rig
    monkeypatch.setattr(service, "start", lambda: None)
    service.detector = None
    with TestClient(
        create_app(service, TOKEN, ASSETS), headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        assert client.get("/api/preview.jpg").status_code == 503
        service.jpeg = cv2.imencode(".jpg", np.zeros((20, 30, 3), np.uint8))[1].tobytes()
        service.preview_at = now[0]
        response = client.get("/api/preview.jpg")
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        now[0] += 2.1
        assert client.get("/api/preview.jpg").status_code == 503
        lease = client.post("/api/control/acquire").json()["lease"]
        response = client.post(
            "/api/control/command", headers={"X-Control-Lease": lease}, json={"action": "start"}
        )
        assert response.status_code == 409


def test_camera_pipeline_eof_and_cleanup(monkeypatch):
    import backend.web.service as module

    serial = Serial()
    captured = []
    now = [100.0]
    service = CameraService(
        CameraConfig(model="detector.pth", serial_port="test"),
        resolve_runtime(device="cpu"),
        clock=lambda: now[0],
    )
    monkeypatch.setattr(module, "load_model_from_path", lambda *a, **kw: object())
    monkeypatch.setattr(module, "open_connection", lambda *a: serial)
    monkeypatch.setattr(module, "verify_velocity_controller", lambda ser: None)
    monkeypatch.setattr(
        module, "open_capture", lambda *a, **kw: SimpleNamespace(get=lambda prop: 30)
    )

    class Reader:
        ended, error = False, None

        def __init__(self, *a, **kw):
            pass

        def start(self):
            owner = service.acquire()["lease"]
            service.command("start", owner)
            return self

        def read(self, **kw):
            if captured:
                self.ended = True
                return None
            now[0] += 0.11
            packet = CapturedFrame(np.zeros((100, 100, 3), np.uint8), service.clock())
            captured.append(packet)
            return packet

        def close(self):
            captured.append("closed")

    monkeypatch.setattr(module, "FrameReader", Reader)
    monkeypatch.setattr(
        module,
        "get_ball_detection",
        lambda model, frame, *a, **kw: ([{"box": (75, 40, 85, 50)}], frame),
    )
    service._camera_loop()
    assert any(command.startswith("V:") for command in serial.commands)
    assert serial.commands[-1] == "Stop" and serial.closed
    assert service.phase == "ended" and not service.tracking and service.lease is None
    assert captured[-1] == "closed" and service.jpeg


def test_serial_failure_disables_service(rig, monkeypatch):
    service, _ = rig
    monkeypatch.setattr(service, "start", lambda: None)

    def fail(payload):
        raise OSError("disconnected")

    monkeypatch.setattr(service.ser, "write", fail)
    with TestClient(
        create_app(service, TOKEN, ASSETS), headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        owner = client.post("/api/control/acquire").json()["lease"]
        response = client.post(
            "/api/control/command", headers={"X-Control-Lease": owner}, json={"action": "right"}
        )
        assert response.status_code == 503
        assert service.halt.is_set() and not service.ready and service.lease is None


@pytest.mark.parametrize(
    "options",
    [
        {"confidence": float("nan")},
        {"capture_fps": -1},
        {"preview_fps": 60},
        {"actor": "actor.pth"},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        CameraConfig(**options)
