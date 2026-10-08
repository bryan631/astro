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
`arduino:avr:nano` (Optiboot). Caveat: the old bootloader does not clear the watchdog, so a
watchdog reset (loop stalled > 2 s) reboot-loops until you power-cycle. The heater pins float as inputs while it
loops, so each MOSFET gate needs a pull-down (e.g. 10k to GND) to keep the heaters off. Burn
Optiboot to fix it properly.

Nano Every (ATmega4809, shows up as `/dev/ttyACM*`):

    arduino-cli core install arduino:megaavr
    arduino-cli lib install "Adafruit BME280 Library" OneWire DallasTemperature
    arduino-cli compile --fqbn arduino:megaavr:nona4809:mode=off firmware/astro_mcu
    arduino-cli upload  --fqbn arduino:megaavr:nona4809:mode=off -p /dev/ttyACM0 firmware/astro_mcu

`mode=off` disables the ATmega328 register emulation (not needed, and slower).
The encoders are analog Hall sensors, read on A2/A3/A6/A7; the wiring is in the sketch's header.
`encoder_scope` streams the raw sensor readings (~700/s) for checking the signals.
