"""Cut the HYG star database down to the star names drawn on the finder view.

Keeps every star to magnitude 6.5 with a proper name (Vega), a Bayer letter (β Cyg) or a
Flamsteed number (61 Cyg): about 3,100 stars. HYG is CC BY-SA 4.0 (David Nash, astronexus):
    curl -LO https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv
    .venv/bin/python scripts/dev/pack_star_names.py hygdata_v41.csv astro/pointing/star_names.csv
"""

import csv
import sys

MAX_MAG = 6.5
GREEK = {"Alp": "α", "Bet": "β", "Gam": "γ", "Del": "δ", "Eps": "ε", "Zet": "ζ", "Eta": "η",
         "The": "θ", "Iot": "ι", "Kap": "κ", "Lam": "λ", "Mu": "μ", "Nu": "ν", "Xi": "ξ",
         "Omi": "ο", "Pi": "π", "Rho": "ρ", "Sig": "σ", "Tau": "τ", "Ups": "υ", "Phi": "φ",
         "Chi": "χ", "Psi": "ψ", "Ome": "ω"}
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def label(row: dict) -> str | None:
    if row["proper"]:
        return row["proper"]
    if row["bayer"]:
        letter, _, n = row["bayer"].partition("-")
        return f"{GREEK[letter]}{n.translate(SUPERSCRIPT)} {row['con']}"
    if row["flam"]:
        return f"{row['flam']} {row['con']}"
    return None


def main(src: str, dest: str) -> None:
    out = []
    for row in csv.DictReader(open(src, encoding="utf-8")):
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
