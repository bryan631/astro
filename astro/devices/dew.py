"""Dew heater control: keep the optics a few degrees above the dew point.

Stepped power rather than PID: the heaters and optics react slowly, the steps are easy to
reason about, and a wrong sensor reading can't wind up an integrator.
"""

import math

from astro.devices.mcu_protocol import Environment, dew_point_c

# (margin above dew point in C, heater percent): smaller margin -> more power.
STEPS = ((5.0, 0), (3.0, 30), (2.0, 60), (1.0, 85), (-math.inf, 100))
MIN_POWER_HUMID = 20  # Florida nights: keep a little heat on above 85% RH regardless
HUMID_RH_PCT = 85


def heater_percent(env: Environment) -> int:
    """Heater power for the current conditions. Uses the optic temperature when measured
    (the secondary/lens cool below air by radiating to the sky), else the air temperature."""
    optic = env.optic_c if not math.isnan(env.optic_c) else env.temp_c
    margin = optic - dew_point_c(env.temp_c, env.rh_pct)
    power = next(p for m, p in STEPS if margin >= m)
    if env.rh_pct >= HUMID_RH_PCT:
        power = max(power, MIN_POWER_HUMID)
    return power
