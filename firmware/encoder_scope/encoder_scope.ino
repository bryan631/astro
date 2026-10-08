// encoder_scope: print the four IntelliScope Hall sensor voltages for the serial plotter.
// Wiring: blue 5V, white GND, black A2 (alt B), red A3 (az B), green A6 (az A), yellow A7 (alt A).
// Expect ~0.2-4.7 V sinusoids, A/B a quarter cycle apart, 36 cycles per turn of each axis.

const uint8_t PINS[4] = {A6, A3, A7, A2};  // azA azB altA altB

void setup() {
  Serial.begin(115200);
  Serial.println("azA azB altA altB");
}

void loop() {
  for (uint8_t i = 0; i < 4; i++) {
    Serial.print(analogRead(PINS[i]) * 5.0 / 1023, 2);
    Serial.print(i < 3 ? ' ' : '\n');
  }
  delay(50);
}
