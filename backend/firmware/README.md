# ESP32 firmware sources

These variants were moved from `guiapp/Platform/` without modification:

- `esp32NoBlue.cpp`: serial-only proportional pan (`P:<value>`) and text commands.
- `esp32float.cpp`: experimental proportional pan with BLE.
- `esp32main.cpp`: older text-command/Bluetooth experiment.

The serial transport sends newline-terminated commands. Manual controls use
`Left`/`Right`; actor control uses `P:<value>,T:0.00`. Choose firmware supporting
the commands and scale in use. The BLE variants contain placeholder UUIDs and
incomplete C++ declarations; `esp32main.cpp` also contains duplicated sketch
sections. They are retained for reference, not advertised as build-ready.
Firmware compilation, flashing and motor calibration are outside the Python
test suite.

Hardware diagnostics are explicit commands:

```bash
python -m backend.tools.probe
python -m backend.tools.motormove --port COM3 --command Stop
python -m backend.tools.motormove --port COM3 --pan 0.5
```

The motor utility sends the selected command to real hardware. The probe opens
serial ports and looks for a `Stopping` response to `Stop`; specify the known
port in the application if a board reset causes auto-detection to miss it.
