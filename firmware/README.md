# astro_mcu firmware (Arduino Nano Every)

Reads the IntelliScope encoders, the BME280 (and optional DS18B20 on the secondary), and drives
the two dew heaters. Protocol and dew logic live on the host: `astro/devices/mcu_protocol.py`,
`astro/devices/dew.py`, `astro/devices/mcu.py`.

    arduino-cli core install arduino:megaavr
    arduino-cli lib install "Adafruit BME280 Library" OneWire DallasTemperature
    arduino-cli compile --fqbn arduino:megaavr:nona4809:mode=off firmware/astro_mcu
    arduino-cli upload  --fqbn arduino:megaavr:nona4809:mode=off -p /dev/ttyACM0 firmware/astro_mcu

`mode=off` disables the ATmega328 register emulation (not needed, and slower).
Encoder pinout and voltage are unverified until Phase 3; see the wiring comment in the sketch.
