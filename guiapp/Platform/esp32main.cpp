#include <HardwareSerial.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// Must match Python script
#define SERVICE_UUID        "SERVICE-CamMan-UUID"
#define CHARACTERISTIC_UUID "CHARACTERISTIC-CamMan-UUID"

const int dirPin = 12;
const int stepPin = 14;

#define motorInterfaceType 1

AccelStepper myStepper(motorInterfaceType, stepPin, dirPin);

const int STEPS_PER_REV = 200;
const int MICROSTEPS = 16;
const float GEAR_RATIO = 1.0;
const float TOTAL_STEPS_PER_REV = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO;
const float STEPS_PER_DEGREE = TOTAL_STEPS_PER_REV / 360.0;

void processCommand(String command) {
    command.trim();
    const long stepsToMove = static_cast<long>(1.0 * STEPS_PER_DEGREE);

    if (command.equals("Right")) {
        myStepper.move(-stepsToMove);
        Serial.println("Moving Right");
    } else if (command.equals("Left")) {
        myStepper.move(stepsToMove);
        Serial.println("Moving Left");
    } else if (command.equals("Stop")) {
        myStepper.stop();
        Serial.println("Stopping");
    }
}

class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        std::string value = pCharacteristic->getValue();
        if (value.length() > 0) {
            processCommand(String(value.c_str()));
        }
    }
};

void setup() {
    Serial.begin(115200);
    myStepper.setMaxSpeed(400);
    myStepper.setAcceleration(300);

    Serial.println("Starting BLE work!");

    BLEDevice::init("ESP32_BLE");
    BLEServer *pServer = BLEDevice::createServer();
    BLEService *pService = pServer->createService(SERVICE_UUID);
    BLECharacteristic *pCharacteristic = pService->createCharacteristic(
                                         CHARACTERISTIC_UUID,
                                         BLECharacteristic::PROPERTY_WRITE
                                       );
    pCharacteristic->setCallbacks(new MyCallbacks());
    pService->start();
    
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
    
    Serial.println("!BlueTooth Connection Broadcasting!");
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
    myStepper.run();
}
= 1.0; 

// Calculate total steps per full revolution of the camera
const float TOTAL_STEPS_PER_REV = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO;
// Calculate steps per degree
const float STEPS_PER_DEGREE = TOTAL_STEPS_PER_REV / 360.0;

void setup() {
    Serial.begin(115200);
    SerialBT.begin("ESP32_BT"); // Bluetooth device name
    myStepper.setMaxSpeed(400);
    myStepper.setAcceleration(300);
    Serial.println("ESP32 ready. Waiting for commands from computer.");
    SerialBT.println("ESP32 ready. Waiting for commands from computer.");
}

void processCommand(String command) {
    command.trim(); // Remove any leading/trailing whitespace
    const long stepsToMove = static_cast<long>(1.0 * STEPS_PER_DEGREE); // 1 degree of movement

    if (command.equals("Right")) {
        myStepper.move(-stepsToMove); // Negative moves it to the right
        Serial.println("Moving Right");
        SerialBT.println("Moving Right");
    } else if (command.equals("Left")) {
        myStepper.move(stepsToMove); // Positive moves it to the left
        Serial.println("Moving Left");
        SerialBT.println("Moving Left");
    } else if (command.equals("Stop")) {
        myStepper.stop();
        Serial.println("Stopping");
        SerialBT.println("Stopping");
    }
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
    if (SerialBT.available() > 0) {
        String command = SerialBT.readStringUntil('\n');
        processCommand(command);
    }
    myStepper.run();
}