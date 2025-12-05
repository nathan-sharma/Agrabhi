#include <Servo.h> 

Servo actuator; 
String command = "";
void setup() {
Serial.begin(9600);
actuator.attach(9);
delay(100);
}

void loop() {
int sensorValue = analogRead(A0); 
float gwcReading = (174.96 - 0.36*sensorValue)/1.67;
if(Serial.available() >0) {  
  char serialReading = Serial.read(); 
  if(serialReading == '\n') { 
    
    if(command == "e") {
    actuator.writeMicroseconds(2000); 
    } 
    else if(command == "r") { 
      actuator.writeMicroseconds(1000); 
    }
   else if(command =="l") { 
    Serial.println(gwcReading);
   }
    
  command = "";
  }
  else { 
  command += serialReading;
}
  

}

}
