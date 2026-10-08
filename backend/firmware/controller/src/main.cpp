#include <Arduino.h>
#include <AccelStepper.h>
#include "protocol.h"

// EXAMPLE wiring/calibration: configure for your board, driver and mechanism.
constexpr int STEP_PIN = 14;
constexpr int DIR_PIN = 27;
constexpr float MOTOR_STEPS_PER_REV = 200.0f;
constexpr float MICROSTEPS = 16.0f;
constexpr float GEAR_RATIO = 1.0f;  // motor revolutions per camera revolution
constexpr float STEPS_PER_DEGREE = MOTOR_STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0f;
constexpr float MAX_PAN_DEGREES_PER_SEC = 30.0f;  // match the policy's calibration
constexpr float ACCEL_DEGREES_PER_SEC2 = 120.0f;
constexpr float TRAVEL_DEGREES = 90.0f;  // +/- travel from the manually centered boot position
constexpr unsigned long WATCHDOG_MS = 750;
constexpr float MAX_SPEED = MAX_PAN_DEGREES_PER_SEC * STEPS_PER_DEGREE;
constexpr float ACCELERATION = ACCEL_DEGREES_PER_SEC2 * STEPS_PER_DEGREE;
constexpr long LIMIT_STEPS = static_cast<long>(TRAVEL_DEGREES * STEPS_PER_DEGREE);

AccelStepper motor(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
char line[48];
size_t length = 0;
bool overflow = false;
float requested = 0, speed = 0;
unsigned long lastCommand = 0, lastMicros = 0;

void stopMotion() {
    requested = speed = 0;
    motor.setSpeed(0);
}

void reply(const char* message) {
    // Telemetry must not block the stepping loop if the host isn't reading.
    if (Serial.availableForWrite() >= static_cast<int>(strlen(message) + 2)) Serial.println(message);
}

void dispatch() {
    line[length] = '\0';
    const auto command = overflow ? camman::Command{camman::Kind::Invalid, 0} : camman::parse(line);
    switch (command.kind) {
        case camman::Kind::Hello:
            reply("CAMMAN/1 VELOCITY");
            break;
        case camman::Kind::Stop:
            stopMotion();
            reply("Stopping");
            break;
        case camman::Kind::Velocity:
            requested = command.velocity;
            lastCommand = millis();
            if (requested == 0) stopMotion();
            break;
        default:
            stopMotion();
            reply("ERR invalid command");
    }
    length = 0;
    overflow = false;
}

void setup() {
    Serial.begin(115200);
    motor.setMaxSpeed(MAX_SPEED);
    motor.setCurrentPosition(0);  // No automatic homing: center the mechanism before power-on.
    lastMicros = micros();
    stopMotion();
}

void loop() {
    // Bound parser work, and never wait for a newline inside the stepping loop.
    for (int count = 0; count < 32 && Serial.available(); ++count) {
        const char ch = static_cast<char>(Serial.read());
        if (ch == '\n') dispatch();
        else if (ch != '\r') {
            if (length < sizeof(line) - 1 && !overflow) line[length++] = ch;
            else overflow = true;
        }
    }
    if (camman::expired(millis(), lastCommand, WATCHDOG_MS)) stopMotion();
    const unsigned long now = micros();
    float dt = static_cast<unsigned long>(now - lastMicros) / 1000000.0f;
    lastMicros = now;
    dt = fminf(dt, 0.05f);
    float target = requested * MAX_SPEED;
    const long position = motor.currentPosition();
    const long remaining = target >= 0 ? LIMIT_STEPS - position : LIMIT_STEPS + position;
    const float brakingSpeed = sqrtf(2 * ACCELERATION * fmaxf(0, remaining));
    target = fmaxf(-brakingSpeed, fminf(brakingSpeed, target));
    const float delta = ACCELERATION * dt;
    speed += fmaxf(-delta, fminf(delta, target - speed));
    if ((position >= LIMIT_STEPS && speed > 0) || (position <= -LIMIT_STEPS && speed < 0)) speed = 0;
    motor.setSpeed(speed);
    motor.runSpeed();
}
