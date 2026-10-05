// blink_check: bring-up test for a fresh Nano. Blinks the L LED (D13) and prints a counter at
// 115200, so both the upload and the serial link are confirmed. Re-flash astro_mcu afterwards.

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  static unsigned long n = 0;
  digitalWrite(LED_BUILTIN, n % 2 ? LOW : HIGH);
  Serial.print("blink ");
  Serial.println(n++);
  delay(250);
}
