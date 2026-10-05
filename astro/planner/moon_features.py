"""Moon features worth a look tonight (PL2): the ones near the shadow line, where low sunlight
throws long shadows. The terminator runs pole to pole, so only longitude matters (libration,
a few degrees, is ignored)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    lon_deg: float  # selenographic, east (toward Mare Crisium) positive
    note: str


FEATURES = (
    Feature("Mare Crisium", 59.1, "A dark oval sea near the eastern edge."),
    Feature("Petavius", 60.4, "A big crater with a crack across its floor."),
    Feature("Theophilus", 26.4, "A deep crater with a mountain in the middle."),
    Feature("Posidonius", 29.9, "A crater with winding valleys, on the shore of a sea."),
    Feature("Montes Apenninus", -3.7, "A long mountain range that catches the sunrise."),
    Feature("Plato", -9.3, "A dark, smooth crater floor, like a lake in the north."),
    Feature("Tycho", -11.4, "Bright rays splash out across the southern highlands."),
    Feature("Clavius", -14.1, "A giant crater with a curving chain of small ones inside."),
    Feature("Copernicus", -20.1, "Terraced walls and central peaks, 93 kilometers wide."),
    Feature("Gassendi", -40.1, "A cracked crater floor on the edge of a sea."),
    Feature("Aristarchus", -47.4, "The brightest spot on the Moon."),
)
BEST_WITHIN_DEG = 40  # of the terminator, on the lit side: shadows still long enough to see


def lit_side(moon_minus_sun_deg: float) -> tuple[float, bool]:
    """(terminator longitude, lit side is east of it) for the Moon's ecliptic longitude minus
    the Sun's. Waxing (0-180): sunrise line sweeps from +90 to -90, lit to its east.
    Waning (180-360): sunset line sweeps from +90 to -90, lit to its west."""
    d = moon_minus_sun_deg % 360
    return (90 - d, True) if d < 180 else (270 - d, False)


def best_features(moon_minus_sun_deg: float) -> list[tuple[Feature, float]]:
    """Lit features near the shadow line, best first, with a 0-1 quality."""
    terminator, east = lit_side(moon_minus_sun_deg)
    out = []
    for f in FEATURES:
        d = f.lon_deg - terminator if east else terminator - f.lon_deg
        if 0 <= d <= BEST_WITHIN_DEG:
            out.append((f, 1 - d / BEST_WITHIN_DEG / 2))  # 1.0 on the line, 0.5 at the limit
    return sorted(out, key=lambda fq: -fq[1])
