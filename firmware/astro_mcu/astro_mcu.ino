// astro_mcu: IntelliScope encoders, environment sensors and dew heaters for the astro host.
// Board: Arduino Nano Every (ATmega4809, 5 V). Protocol: see astro/devices/mcu_protocol.py.
//
// Wiring (screw-terminal board):
//   Az encoder A/B  -> D2 / D4      Alt encoder A/B -> D7 / D8   (5 V TTL quadrature; UNVERIFIED)
//   Heater MOSFETs  -> D9 (secondary), D10 (finder lens)   (power from the bank, not the Nano)
//   BME280 I2C      -> A4 SDA / A5 SCL                      DS18B20 (optional) -> D12, 4.7k pull-up

#include <Adafruit_BME280.h>
#include <DallasTemperature.h>
#include <OneWire.h>

const char* NAME = "astro-mcu";
const char* VERSION = "0.1";

const uint8_t AZ_A = 2, AZ_B = 4, ALT_A = 7, ALT_B = 8;
const uint8_t HEATER_PINS[2] = {9, 10};
const uint8_t ONE_WIRE_PIN = 12;

const unsigned long POS_EVERY_MS = 50;     // 20 Hz
const unsigned long ENV_EVERY_MS = 1000;   // 1 Hz
const unsigned long HEARTBEAT_MS = 10000;  // heaters off if the host goes quiet

// Quadrature decoding: one table for every (previous AB, current AB) transition.
// +1 / -1 for a valid step, 0 for no change or an invalid (missed) transition.
const int8_t QUAD[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
volatile long azCount = 0, altCount = 0;
volatile uint8_t azState = 0, altState = 0;

Adafruit_BME280 bme;
bool bmeOk = false;
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature optic(&oneWire);
bool opticOk = false;
bool opticPending = false;  // a DS18B20 conversion is running (non-blocking, ~750 ms)
unsigned long opticStarted = 0;
float opticC = NAN;
const unsigned long OPTIC_CONVERSION_MS = 800;

unsigned long lastPos = 0, lastEnv = 0, lastPing = 0;
char line[48];
uint8_t lineLen = 0;

void azChange() {
  uint8_t s = (digitalRead(AZ_A) << 1) | digitalRead(AZ_B);
  azCount += QUAD[(azState << 2) | s];
  azState = s;
}

void altChange() {
  uint8_t s = (digitalRead(ALT_A) << 1) | digitalRead(ALT_B);
  altCount += QUAD[(altState << 2) | s];
  altState = s;
}

// Send BODY*CS\n where CS is the XOR of BODY's bytes, two hex digits.
void sendLine(const char* body) {
  uint8_t cs = 0;
  for (const char* p = body; *p; p++) cs ^= (uint8_t)*p;
  Serial.print(body);
  Serial.print('*');
  if (cs < 0x10) Serial.print('0');
  Serial.println(cs, HEX);
}

void setHeater(uint8_t ch, int percent) {
  if (ch > 1) return;
  percent = constrain(percent, 0, 100);
  analogWrite(HEATER_PINS[ch], map(percent, 0, 100, 0, 255));
}

void heatersOff() {
  setHeater(0, 0);
  setHeater(1, 0);
}

// Exactly two hex digits after '*', matching the XOR of the body; anything else is corrupt.
bool checksumOk(char* text) {
  char* star = strrchr(text, '*');
  if (!star || strlen(star + 1) != 2 || !isxdigit(star[1]) || !isxdigit(star[2])) return false;
  *star = '\0';
  uint8_t cs = 0;
  for (char* p = text; *p; p++) cs ^= (uint8_t)*p;
  return cs == (uint8_t)strtol(star + 1, nullptr, 16);
}

// Parse a whole decimal integer; false on empty input or trailing junk.
bool parseInt(const char* s, char** end, long* out) {
  *out = strtol(s, end, 10);
  return *end != s;
}

void handleCommand(char* text) {
  if (!checksumOk(text)) return;
  char buf[40];
  if (strcmp(text, "PING") == 0) {
    lastPing = millis();
  } else if (strncmp(text, "HEAT ", 5) == 0) {
    char *end1, *end2;
    long ch, pct;
    if (parseInt(text + 5, &end1, &ch) && *end1 == ' ' && parseInt(end1 + 1, &end2, &pct) &&
        *end2 == '\0') {
      setHeater(ch, pct);
      lastPing = millis();
    } else {
      sendLine("ERR bad HEAT");
    }
  } else if (strcmp(text, "ZERO") == 0) {
    noInterrupts();
    azCount = altCount = 0;
    interrupts();
  } else if (strcmp(text, "VER?") == 0) {
    snprintf(buf, sizeof buf, "VER %s %s", NAME, VERSION);
    sendLine(buf);
  } else {
    sendLine("ERR unknown command");
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen) {
        line[lineLen] = '\0';
        handleCommand(line);
        lineLen = 0;
      }
    } else if (lineLen < sizeof line - 1) {
      line[lineLen++] = c;
    } else {
      lineLen = 0;  // overlong line: drop it
    }
  }
}

void sendPosition() {
  noInterrupts();
  long az = azCount, alt = altCount;
  interrupts();
  char buf[40];
  snprintf(buf, sizeof buf, "POS %ld %ld", az, alt);
  sendLine(buf);
}

void sendEnvironment() {
  if (!bmeOk) {
    sendLine("ERR bme280 missing");
    return;
  }
  float t = bme.readTemperature(), rh = bme.readHumidity();
  float o = opticOk ? opticC : NAN;  // latest finished conversion (see pollOptic)
  char buf[40], ts[8], rhs[8], os[8];
  dtostrf(t, 1, 1, ts);  // AVR printf has no %f
  dtostrf(rh, 1, 1, rhs);
  if (isnan(o)) strcpy(os, "nan"); else dtostrf(o, 1, 1, os);
  snprintf(buf, sizeof buf, "ENV %s %s %s", ts, rhs, os);
  sendLine(buf);
}

// Start a DS18B20 conversion, and collect it on a later loop once it's done, so position
// reports, commands and the watchdog never wait ~750 ms for the sensor.
void pollOptic(unsigned long now) {
  if (!opticOk) return;
  if (!opticPending) {
    optic.requestTemperatures();
    opticPending = true;
    opticStarted = now;
  } else if (now - opticStarted >= OPTIC_CONVERSION_MS) {
    float t = optic.getTempCByIndex(0);
    opticC = (t == DEVICE_DISCONNECTED_C) ? NAN : t;
    opticPending = false;
  }
}

void watchdogStart() {  // ATmega4809: reset if loop() stalls for ~2 s
  _PROTECTED_WRITE(WDT.CTRLA, WDT_PERIOD_2KCLK_gc);
}

void watchdogFeed() {
  __asm__ __volatile__("wdr");
}

void setup() {
  Serial.begin(115200);
  const uint8_t encoderPins[] = {AZ_A, AZ_B, ALT_A, ALT_B};
  for (uint8_t pin : encoderPins) pinMode(pin, INPUT_PULLUP);
  for (uint8_t pin : HEATER_PINS) pinMode(pin, OUTPUT);
  heatersOff();
  azState = (digitalRead(AZ_A) << 1) | digitalRead(AZ_B);
  altState = (digitalRead(ALT_A) << 1) | digitalRead(ALT_B);
  attachInterrupt(digitalPinToInterrupt(AZ_A), azChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(AZ_B), azChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ALT_A), altChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ALT_B), altChange, CHANGE);
  bmeOk = bme.begin(0x76) || bme.begin(0x77);
  optic.begin();
  opticOk = optic.getDeviceCount() > 0;
  optic.setWaitForConversion(false);
  watchdogStart();
}

void loop() {
  watchdogFeed();
  readSerial();
  unsigned long now = millis();
  if (now - lastPos >= POS_EVERY_MS) {
    lastPos = now;
    sendPosition();
  }
  pollOptic(now);
  if (now - lastEnv >= ENV_EVERY_MS) {
    lastEnv = now;
    sendEnvironment();
  }
  if (now - lastPing > HEARTBEAT_MS) heatersOff();  // failsafe: host gone
}
