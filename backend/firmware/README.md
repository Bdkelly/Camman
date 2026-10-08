# Camera controller firmware

Use [controller/](controller/README.md) for the current **CAMMAN/1 VELOCITY**
protocol. It includes a buildable PlatformIO ESP32 STEP/DIR project, normalized
velocity control, serial handshake, watchdog, acceleration and software travel
limits. Its README covers wiring, calibration, building and uploading, plus the
protocol needed to implement another microcontroller.

The following earlier sketches remain for reference and are **not compatible
with the new tracking controller**:

- `esp32NoBlue.cpp`: legacy serial commands; `P:` sets a position rather than the
  velocity used by policy training.
- `esp32float.cpp`: experimental BLE/proportional control.
- `esp32main.cpp`: older text-command/Bluetooth experiment.

The BLE sketches have placeholder UUIDs and incomplete/duplicated declarations.
The current application uses serial, not BLE. Do not flash a reference sketch
for a newly trained actor. Existing detector checkpoints remain compatible;
legacy bare actor weights need retraining with the new control contract.
