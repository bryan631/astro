"""The "Tonight" report: what each spot can see tonight, when, and how cloudy, ranked.

Runs anywhere Python does: on the Mele (/tonight) or in a GitHub Action that publishes it
(`python -m astro.planner.report out/index.html`, spots from SPOTS_TOML or data/spots.toml).
"""

import html
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy.coordinates import get_body
from astropy.time import Time

from astro import spots as spot_store
from astro.planner.almanac import _moon_lit, phase_name
from astro.planner.catalog import load_targets
from astro.planner.tonight import STEP_MIN, candidates, moonlight_factor, night_times
from astro.planner.weather import hour_key, hourly_cloud_cover
from astro.spots import Spot

TOP = 6  # targets listed per spot
MIN_ALT = 10.0  # even with no trees, the haze near the horizon isn't worth it


@dataclass(frozen=True)
class Pick:
    name: str
    category: str
    start: datetime  # longest stretch above the trees tonight
    end: datetime
    best_alt: float
    clear_hours: float  # hours above the trees, weighted by how clear the sky is
    cloud_pct: float | None  # mean forecast cloud while it's up
    score: float


@dataclass(frozen=True)
class SpotNight:
    spot: Spot
    picks: list[Pick]
    score: float


@dataclass(frozen=True)
class Night:
    made: datetime
    dark_start: datetime
    dark_end: datetime
    clouds: list[tuple[datetime, float | None]]  # hourly through the dark hours
    moon: str
    spots: list[SpotNight]  # best first


def _runs(mask: np.ndarray) -> tuple[int, int]:
    """(start, end) indexes of the longest run of True, end exclusive."""
    best, start = (0, 0), None
    for i, v in enumerate([*mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            best, start = max(best, (start, i), key=lambda r: r[1] - r[0]), None
    return best


def build(spots: list[Spot], now: datetime,
          clouds: Callable[[float, float], dict[str, float] | None] = hourly_cloud_cover,
          targets=None) -> Night | None:
    """Tonight's plan for every spot, or None if it doesn't get dark in the next 24 hours."""
    targets = load_targets() if targets is None else targets
    first = spots[0].site
    times = night_times(first, now, hours=24)  # from a midday run, through dawn
    if len(times) == 0:
        return None
    gaps = np.flatnonzero(np.diff(times.jd) * 24 * 60 > STEP_MIN * 1.5)
    times = times[: gaps[0] + 1] if len(gaps) else times  # tonight only, not tomorrow's dusk
    step = STEP_MIN / 60
    when = times.to_datetime(timezone=UTC)
    forecasts: dict[tuple[float, float], dict[str, float] | None] = {}
    nights = []
    for spot in spots:
        key = (round(spot.site.lat_deg, 2), round(spot.site.lon_deg, 2))
        if key not in forecasts:
            forecasts[key] = clouds(*key)
        hourly = forecasts[key]
        cloud = np.array([np.nan if hourly is None else hourly.get(hour_key(t), np.nan) for t in when])
        clear = np.where(np.isnan(cloud), 1.0, 1 - cloud / 100)
        frame = spot.site.frame(now).replicate_without_data(obstime=times)
        moon = get_body("moon", times, frame.location).transform_to(frame)
        lit, _ = _moon_lit(times)
        picks = []
        for name, cat, _, coord, weight in candidates(times, frame.location, targets):
            aa = coord.transform_to(frame)
            alt, az = aa.alt.deg, aa.az.deg
            up = (alt > spot.mask.min_alt(az)) & (alt > MIN_ALT)
            if not up.any():
                continue
            s, e = _runs(up)
            i = int(np.argmax(np.where(up, alt, -90)))
            clear_hours = float((clear * up).sum() * step)
            score = weight * clear_hours * (0.6 + 0.4 * alt[i] / 90)
            if cat not in ("planet", "moon") and moon.alt.deg[i] > 0:
                score *= moonlight_factor(cat, aa[i].separation(moon[i]).deg, float(lit[i]))
            seen = cloud[up]
            picks.append(Pick(name, cat, when[s], when[e - 1], round(float(alt[i])),
                              round(clear_hours, 1),
                              None if np.isnan(seen).all() else round(float(np.nanmean(seen))),
                              round(score, 3)))
        picks.sort(key=lambda p: -p.score)
        nights.append(SpotNight(spot, picks[:TOP], round(sum(p.score for p in picks[:5]), 2)))
    hours = sorted({hour_key(t) for t in when})
    hourly = forecasts[next(iter(forecasts))]
    lit, waxing = _moon_lit(Time(now))
    return Night(now, when[0], when[-1],
                 [(datetime.fromisoformat(h).replace(tzinfo=UTC),
                   None if hourly is None else hourly.get(h)) for h in hours],
                 f"Moon: {phase_name(lit[0], waxing[0])}, {lit[0]:.0%} lit",
                 sorted(nights, key=lambda n: -n.score))


# --- HTML ------------------------------------------------------------------------------------
def _t(t: datetime, tz) -> str:
    return t.astimezone(tz).strftime("%I:%M %p").lstrip("0")


def _cloud_class(pct: float | None) -> str:
    return "unk" if pct is None else "clear" if pct < 30 else "part" if pct < 70 else "cloudy"


def render_html(night: Night | None, spots: list[Spot], note: str = "") -> str:
    tz = spots[0].site.timezone if spots else UTC
    if night is None:
        body = f"<p>{html.escape(note or 'It does not get dark tonight.')}</p>"
        title = "Tonight"
    else:
        title = f"Tonight, {night.dark_start.astimezone(tz):%a %b %d}"
        strip = "".join(
            f'<div class="h {_cloud_class(c)}"><b>{t.astimezone(tz):%-I%p}</b>'
            f'<span>{"?" if c is None else f"{c:.0f}%"}</span></div>' for t, c in night.clouds)
        best = night.spots[0] if night.spots and night.spots[0].picks else None
        verdict = (f"Best spot: <b>{html.escape(best.spot.name)}</b>, with "
                   f"{html.escape(best.picks[0].name)} from {_t(best.picks[0].start, tz)}."
                   if best else "Nothing clears the trees tonight.")
        cards = []
        for n in night.spots:
            rows = "".join(
                f"<tr><td>{html.escape(p.name)}<small>{p.category}</small></td>"
                f"<td>{_t(p.start, tz)}–{_t(p.end, tz)}</td><td>{p.best_alt}°</td>"
                f'<td class="{_cloud_class(p.cloud_pct)}">'
                f'{"?" if p.cloud_pct is None else f"{p.cloud_pct}%"}</td></tr>' for p in n.picks)
            cards.append(f'<section><h2>{html.escape(n.spot.name)}</h2>' + (
                f"<table><tr><th>Target</th><th>Above the trees</th><th>High</th><th>Cloud</th></tr>"
                f"{rows}</table>" if rows else "<p>Nothing clears the trees tonight.</p>")
                + "</section>")
        body = (f'<p class="verdict">{verdict}</p>'
                f'<p class="sub">Dark {_t(night.dark_start, tz)} to {_t(night.dark_end, tz)} · '
                f"{night.moon}</p>"
                f'<div class="strip">{strip}</div>' + "".join(cards)
                + f'<p class="sub">Updated {_t(night.made, tz)} {night.made.astimezone(tz):%a}. '
                "Cloud is the hourly forecast (Open-Meteo); times are local.</p>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
:root {{ --bg:#120707; --fg:#f2c9c4; --dim:#9a6b66; --line:#3a1c1a; --ok:#7fbf7f; --mid:#d6b25a; --bad:#c76b6b; }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.45 system-ui, sans-serif; margin:0; padding:16px; }}
main {{ max-width:760px; margin:0 auto; }}
h1 {{ font-size:22px; margin:0 0 6px; }} h2 {{ font-size:18px; margin:18px 0 6px; }}
.verdict {{ font-size:18px; margin:6px 0; }} .sub {{ color:var(--dim); font-size:14px; }}
.strip {{ display:flex; gap:4px; overflow-x:auto; padding:6px 0; }}
.h {{ min-width:48px; text-align:center; border:1px solid var(--line); border-radius:6px; padding:4px; }}
.h b {{ display:block; font-size:12px; color:var(--dim); font-weight:normal; }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:6px 4px; border-bottom:1px solid var(--line); }}
th {{ color:var(--dim); font-weight:normal; font-size:13px; }}
td small {{ display:block; color:var(--dim); font-size:12px; }}
.clear {{ color:var(--ok); }} .part {{ color:var(--mid); }} .cloudy {{ color:var(--bad); }} .unk {{ color:var(--dim); }}
</style></head><body><main><h1>{title}</h1>{body}</main></body></html>
"""


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "out/index.html")
    text = os.environ.get("SPOTS_TOML") or (Path("data/spots.toml").read_text()
                                            if Path("data/spots.toml").exists() else "")
    spots, _ = spot_store.parse(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not spots:  # a page that says so, not a failing scheduled job
        print("No spots: set SPOTS_TOML or record treelines first", file=sys.stderr)
        out.write_text(render_html(None, [], "No spots yet: record a treeline with the tablet, "
                                               "then run scripts/publish-spots.sh."))
        return 0
    out.write_text(render_html(build(spots, datetime.now(UTC)), spots))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
