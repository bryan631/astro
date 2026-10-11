"""Cut the HYG star database down to the star names drawn on the finder view.

Keeps every star to magnitude 6.5 with a proper name (Vega): about 350 stars. Bayer letters
(δ Cyg) and numbered names (Flamsteed "61 Cyg", Gould "82 G. Eri") are left out: they clutter
the view and say nothing out loud (2026-10-10, 2026-10-11). HYG is CC BY-SA 4.0 (David Nash, astronexus):
    curl -LO https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv
    .venv/bin/python scripts/dev/pack_star_names.py hygdata_v41.csv astro/pointing/star_names.csv
"""

import csv
import sys

MAX_MAG = 6.5


def label(row: dict) -> str | None:
    if row["proper"] and not row["proper"][0].isdigit():
        return row["proper"]
    return None


def main(src: str, dest: str) -> None:
    out = []
    with open(src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["id"] == "0" or not row["mag"] or float(row["mag"]) > MAX_MAG:
            continue  # the Sun; too faint
        if name := label(row):
            out.append((name, round(float(row["ra"]) * 15, 4), round(float(row["dec"]), 4), float(row["mag"])))
    out.sort(key=lambda r: r[3])  # brightest first
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write("# Star names for the finder view, from the HYG database v4.1 (CC BY-SA 4.0, "
                "https://github.com/astronexus/HYG-Database). scripts/dev/pack_star_names.py\n")
        w = csv.writer(f)
        w.writerow(["name", "ra_deg", "dec_deg", "mag"])
        w.writerows(out)
    print(len(out), "stars")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
