#include <HardwareSerial.h>
#include <AccelStepper.h>

const int dirPin = 12;
const int stepPin = 14;
#define motorInterfaceType 1

AccelStepper myStepper(motorInterfaceType, stepPin, dirPin);

const float ACTION_SCALAR = 50.0;
const int DIR_POLARITY = 1;

void moveMotor(float panValue) {
    float targetPosition = panValue * ACTION_SCALAR * DIR_POLARITY;
    
    String logMsg = "Pan: " + String(panValue, 2);
    myStepper.moveTo((long)targetPosition);
    
    if (panValue > 0.01) {
        logMsg += " (Right)";
    } else if (panValue < -0.01) {
        logMsg += " (Left)";
    } else {
        logMsg += " (Neutral)";
    }
    
    Serial.println(logMsg);
}

void processCommand(String command) {
    command.trim();
    
    if (command.equalsIgnoreCase("Stop")) {
        myStepper.stop();
        Serial.println("Stopping");
        return;
    }
    
    if (command.startsWith("P:")) {
        String panValStr = command.substring(2);
        float panValue = panValStr.toFloat();
        
        if (panValue < -1.0) panValue = -1.0;
        if (panValue > 1.0) panValue = 1.0;
        
        moveMotor(panValue);
    }
    else if (command.equalsIgnoreCase("Right")) {
        moveMotor(1.0);
    }
    else if (command.equalsIgnoreCase("Left")) {
        moveMotor(-1.0);
    }
    else if (command.equalsIgnoreCase("Neutral") || command.equalsIgnoreCase("Center")) {
        moveMotor(0.0);
    }
}

void setup() {
    Serial.begin(115200);
    myStepper.setMaxSpeed(1000);
    myStepper.setAcceleration(800);
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
    myStepper.run();
}