# Phone and touchscreen control

Camman's web interface runs on the Jetson and opens in a phone, tablet, laptop,
or local touchscreen browser. It needs a local network, **not internet access**.
The server owns capture, inference and serial communication; the browser only
receives previews/status and sends controls. PyQt remains available via `camman`.

## Install and start

From the repository root, in your activated environment:

```sh
python -m pip install -e ".[web]"
camman-web --source 0
```

Open `http://127.0.0.1:8080` and enter the generated pairing key printed in the
terminal. This starts in **preview mode**, without automatic tracking. No
`--serial-port` means **dry-run**; no motor is contacted. A video file can replace
`0`. Files play once: restart the server to replay them.

For phone access, join the same private network as the Jetson and start:

```sh
camman-web --host 0.0.0.0 --source 0 --model /path/to/detector.pth --actor /path/to/actor.pth --serial-port /dev/ttyUSB0 --profile jetson --device cuda
```

Open `http://JETSON_LAN_IP:8080` on the phone. Substitute the actual IP and model
paths. Windows serial ports look like `COM3`. Omitting `--actor` uses proportional
control; supplying an actor requires a detector. Only load checkpoints you trust.

**Jetson installation:** first follow [JETSON.md](../../backend/JETSON.md) to retain
NVIDIA's native torch/torchvision and GStreamer OpenCV. In that prepared environment,
install just the web dependencies and update the editable project without resolving
its base dependencies again:

```sh
python -m pip install 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1'
python -m pip install --no-deps -e .
```

`python -m frontend.web.app` accepts the same arguments. `--help` lists capture,
profile, device, precision, inference-FPS and preview options. The server binds
only to localhost by default; `--host 0.0.0.0` explicitly enables network access.
This command does not configure Wi-Fi or create an access point. Use an existing
private router or separately configure/test an access-point-capable adapter.

Run **one** server process. Do not use multiple Uvicorn workers, auto-reload, or
run PyQt/headless tracking against the same camera/serial device simultaneously.
The desktop and web frontends share backend code, not a live process/session.

## Using the interface

1. Enter the pairing key. Preview/status are now visible; the camera does not move.
2. Tap **Take control**. Only one browser/tab can hold the control lease.
3. Hold **Left** or **Right** to position the camera. Release to stop. Tilt remains manual.
4. Tap **Start tracking**. Detection and automatic pan start together. This is
   disabled if no detector was configured or the camera failed/ended.
5. Tap **Stop motion** to stop automatic/manual motion and revoke the lease.
   Take control again before resuming. Any paired observer can stop motion.
6. **Release control** stops motion and makes the platform available to another browser.

The page displays dry-run versus serial operation, camera readiness, ball detection,
inference time, preview freshness, and bounded recent command logs. Model selection
and calibration are server startup settings in this first web version; there is no
remote file upload, firmware flashing, or model training UI.

The browser renews its lease every 400 ms. Without renewal, the server stops motion
at the **2-second lease deadline**. Switching apps/tabs, disconnecting or closing
the page attempts an immediate release; the timeout covers a lost request. Manual
commands are only **200 ms pulses**, repeated while holding. Browser timer throttling
may cause pauses; reconnecting never automatically resumes tracking.

A dedicated control thread checks deadlines every 20 ms, independently of camera
reads and detector inference. Stop revokes the lease and invalidates in-flight
detections. The existing frame-age/target-loss checks and MCU watchdog still apply.
HTTP Stop is a software control, not a physical emergency-stop circuit. Verify the
[firmware and mechanical calibration](../../backend/firmware/controller/README.md)
on your rig before relying on motor movement.

## Preview and performance

Preview uses authenticated JPEG snapshots, at most one waiting image, encoded once
by the server and reused across browsers. Defaults: up to **960 px wide, 8 FPS,
JPEG quality 75**. Browser requests are sequential; slow links do not build a frame
queue. No audio or recording is implemented. Detection uses the original frame,
independently of preview resize. While inference runs, preview cadence can be lower
than its cap. A preview older than two seconds is labelled stale/unavailable.

For a lighter preview:

```sh
camman-web --host 0.0.0.0 --source 0 --preview-width 640 --preview-fps 5
```

This is a first local preview transport, not a measured low-latency video system.
JPEG encoding and Wi-Fi bandwidth should be benchmarked on the actual Jetson and
camera. A future WebRTC transport can replace preview delivery without changing
control/inference ownership. A local touchscreen uses this same page in a browser;
no separate touchscreen UI is needed.

## Pairing and deployment

Every API endpoint, including preview/status, requires the pairing key as a Bearer
header. The UI keeps the key **in memory only**, never in URLs or browser storage.
Reloading requires pairing again. Use a randomly generated persistent key through
`CAMMAN_WEB_TOKEN` when needed (at least 24 non-whitespace ASCII characters); the
server does not print a configured key. The default generated key changes at restart.

The page has no external scripts, fonts or CDN dependencies. Cross-origin requests
are rejected, no CORS permission is granted, and responses are not cached. Control
lease secrets are separate from pairing keys and never appear in status/log output.

Plain HTTP is for a trusted private network: it does not encrypt the key or video.
Do not expose port 8080 to the internet. For an untrusted/shared network, deploy
HTTPS or a private VPN, preserving the public Host/Origin consistently at any proxy.
This release does not provision TLS certificates, hotspots, or QR pairing automatically.

## API

All `/api/` requests require `Authorization: Bearer KEY`. Mutations use POST.
Control requests additionally use `X-Control-Lease: LEASE`, except acquire and Stop.

| Route | Purpose |
| --- | --- |
| `GET /api/status` | Readiness, tracking, controller, model, metrics, recent logs |
| `GET /api/preview.jpg` | Latest JPEG; 503 if missing/stale |
| `POST /api/control/acquire` | Obtain exclusive lease; 409 when occupied |
| `POST /api/control/heartbeat` | Renew valid lease; expired leases cannot be revived |
| `POST /api/control/release` | Stop and release ownership |
| `POST /api/control/command` | JSON `{"action":"start"}`; also `stop`, `left`, `right`, `manual-stop` |

Unauthorized access returns 401; invalid lease/origin 403; unmet prerequisites 409;
invalid command body 422; serial transport failure 503 and disables the service.
The API deliberately provides no arbitrary serial strings or filesystem paths.

Implementation: [service.py](../../backend/web/service.py) owns resources and control;
[api.py](../../backend/web/api.py) provides HTTP; [app.py](app.py) wires startup and
static assets; [static/](static/) contains the shared responsive UI.

## Tests

```sh
python -m pip install -e ".[frontend,web,training,dev]"
QT_QPA_PLATFORM=offscreen python -m pytest backend/tests/test_web.py
```

Tests exercise the real API, lease conflicts/expiry, stale detections, motor pulses,
serial errors, preview freshness, and resource cleanup with simulated hardware.
Physical controller, Android Wi-Fi and Jetson GPU behavior need on-device validation.
