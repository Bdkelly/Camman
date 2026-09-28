#include <HardwareSerial.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <AccelStepper.h>

#define SERVICE_UUID        "SERVICE-CamMan-UUID"
#define CHARACTERISTIC_UUID "CHARACTERISTIC-CamMan-UUID"

const int dirPin = 12;
const int stepPin = 14;

#define motorInterfaceType 1
AccelStepper myStepper(motorInterfaceType, stepPin, dirPin);

const float ACTION_SCALAR = 200.0;

const int DIR_POLARITY = 1;

bool deviceConnected = false;

void moveMotor(float panAction) {
    long stepsToMove = (long)(panAction * ACTION_SCALAR * DIR_POLARITY);
    if (stepsToMove != 0) {
        myStepper.move(stepsToMove);
    }
    if (panAction > 0) {
        logMsg += " (Right)";
    } else if (panAction < 0) {
        logMsg += " (Left)";
    }
    Serial.println(logMsg);
    if (deviceConnected) {
        pCharacteristic->setValue(logMsg.c_str());
        pCharacteristic->notify();
    }
}
void processCommand(String command) {
    command.trim();
    if (command.equalsIgnoreCase("Stop")) {
        myStepper.stop();
        Serial.println("Stopping");
        return;
    }
    if (command.startsWith("P:")) {
        int commaIndex = command.indexOf(',');
        if (commaIndex > 0) {
            String panValStr = command.substring(2, commaIndex);
            float panValue = panValStr.toFloat();
            if (abs(panValue) > 0.01) {
                moveMotor(panValue);
            }
        }
    }
    else if (command.equalsIgnoreCase("Right")) {
        moveMotor(1.0);
    }
    else if (command.equalsIgnoreCase("Left")) {
        moveMotor(-1.0);
    }
}
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      pServer->getAdvertising()->start(); 
    }
};
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
    myStepper.setMaxSpeed(1000);
    myStepper.setAcceleration(800);
    BLEDevice::init("ESP32_CamMan");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);
    
    pCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID,
                        BLECharacteristic::PROPERTY_READ   |
                        BLECharacteristic::PROPERTY_WRITE  |
                        BLECharacteristic::PROPERTY_NOTIFY |
                        BLECharacteristic::PROPERTY_INDICATE
                      );

    pCharacteristic->setCallbacks(new MyCallbacks());
    pCharacteristic->addDescriptor(new BLE2902());

    pService->start();
    
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0); 
    BLEDevice::startAdvertising();
    
    Serial.println("Ready via Serial (USB) and BLE.");
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
    myStepper.run();
}