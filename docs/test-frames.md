# Recording real finder test frames

Each case becomes a regression test (`tests/test_real_frames.py`). Use the defaults (0.8 s,
gain 200) unless noted. Close any viewer first. A MISMATCH still saves the frame: tell Claude
about it, since those are the most useful cases.

    scripts/hwcheck/capture_cases.py LABEL --expect OUTCOME --notes "what you did"

| # | Setup | Label | Expect |
|---|---|---|---|
| 1 | Clear sky, high up (60-90 deg) | `zenith` | solves |
| 2 | Clear sky, low (20-30 deg), away from trees | `low_clear` | solves |
| 3 | Clear sky, finder rotated ~90 deg on its mount | `rotated_90` | solves |
| 4 | Clear sky, the north (Polaris) area | `north` | solves |
| 5 | Clear sky, the south | `south` | solves |
| 6 | Lens cap on (plastic caps pass IR; try one of each cap you have) | `lens_cap_2` | no_stars |
| 7 | Hand or cardboard over the lens | `covered` | no_stars |
| 8 | Half the view blocked by a tree or house | `trees_half` | few_stars or solves |
| 9 | Almost all blocked by trees | `trees_full` | few_stars |
| 10 | Thin clouds or haze | `haze` | few_stars or solves |
| 11 | Focus turned well past best, toward infinity end | `defocus_far` | out_of_focus |
| 12 | Focus turned well short of best, toward close end | `defocus_near` | out_of_focus |
| 13 | Slightly soft focus (one small turn off) | `defocus_slight` | solves |
| 14 | Pointed at a streetlight or porch light | `light` | not_sky |
| 15 | Twilight, sky still bright | `twilight` | not_sky |
| 16 | Short exposure 0.2 s (`--exp 0.2`) on clear sky | `short_exp` | few_stars |
| 17 | Long exposure 2 s (`--exp 2`) on clear sky | `long_exp` | solves |
| 18 | Moon in view (`--exp 0.2`) | `moon` | auto (records what it gets) |

Remember to put the focus back where it was after 11-13 (tape a mark on the ring first).
