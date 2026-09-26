# Plan B input preflight — 2026-09-25

All four target preflights passed using the verified
`../plan_b_inputs_20260925/slots.csv` snapshot. `summary.json` records the
target populations, fold counts, and hashes. Each target directory saves the
exact joined input panel, coverage, excluded keys, manifest, and preflight
report. Input hashes and saved row counts were checked after writing.
After the separate A/B jobs changed the database file, the saved full
rushing-yards panel was recomputed and matched its earlier SHA-256 hash; see
`../plan_b_inputs_20260925/reverification.json`.

| Target | Represented rows, all seasons | Test 2023 | Test 2024 | Test 2025 |
| --- | ---: | ---: | ---: | ---: |
| Targets | 87,510 | 7,067 | 7,072 | 7,072 |
| Rushing attempts | 101,138 | 8,155 | 8,160 | 8,160 |
| Receiving yards | 87,510 | 7,067 | 7,072 | 7,072 |
| Rushing yards | 101,138 | 8,155 | 8,160 | 8,160 |

Targets and receiving yards exclude QB rows by design. Roster caps exclude
53,201 eligible target/receiving rows and 62,220 rushing rows across the
2013–2025 panel. The subsequent first-cut model run is under
`../plan_b_run_20260925/`; a passed preflight alone does not establish
convergence or accuracy.
