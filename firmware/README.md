# astro_mcu firmware (classic Nano or Nano Every)

Reads the IntelliScope encoders, the BME280 (and optional DS18B20 on the secondary), and drives
the two dew heaters. Protocol and dew logic live on the host: `astro/devices/mcu_protocol.py`,
`astro/devices/dew.py`, `astro/devices/mcu.py`.

Classic Nano (ATmega328P, CH340 USB, shows up as `/dev/ttyUSB*`):

    arduino-cli core install arduino:avr
    arduino-cli lib install "Adafruit BME280 Library" OneWire DallasTemperature
    arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328old firmware/astro_mcu
    arduino-cli upload  --fqbn arduino:avr:nano:cpu=atmega328old -p /dev/ttyUSB0 firmware/astro_mcu

Most CH340 clones ship the old bootloader (`cpu=atmega328old`); if upload times out, try
`arduino:avr:nano` (Optiboot). The encoders use pin-change interrupts on this board, because only
D2/D3 have external interrupts. Caveat: the old bootloader does not clear the watchdog, so a
watchdog reset (loop stalled > 2 s) reboot-loops until you power-cycle. The heater pins float as inputs while it
loops, so each MOSFET gate needs a pull-down (e.g. 10k to GND) to keep the heaters off. Burn
Optiboot to fix it properly.

Nano Every (ATmega4809, shows up as `/dev/ttyACM*`):

    arduino-cli core install arduino:megaavr
    arduino-cli lib install "Adafruit BME280 Library" OneWire DallasTemperature
    arduino-cli compile --fqbn arduino:megaavr:nona4809:mode=off firmware/astro_mcu
    arduino-cli upload  --fqbn arduino:megaavr:nona4809:mode=off -p /dev/ttyACM0 firmware/astro_mcu

`mode=off` disables the ATmega328 register emulation (not needed, and slower).
Encoder pinout and voltage are unverified until Phase 3; see the wiring comment in the sketch.
