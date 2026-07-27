# Scan-Junctions Evaluation Rubric: Phantom Junctions

The attached `junction_scan_report.md` was produced by the `scan-junctions`
skill run against a lattice JSON carrying **phantom junctions that have no
material in the scan**: the outermost unit-cell layer was copied and shifted one
cell pitch further out, into empty air beyond the specimen. The CT volume is
untouched and is the reference specimen, which contains exactly 2 genuinely
missing junctions.

Generate the input with:

```bash
python evals/perturb_lattice_json.py REGISTERED.json phantom_x.json \
  --add-phantom-slab x
```

The slab sits in air, so it stays dark at every sampling radius: 662 of 665 dark
junctions in one connected component at radius 4, and 363 of 367 at radius 12.

The correct outcome is that the report identifies a large contiguous region of
junctions with no material and classifies it as systematic, not as a scatter of
individual defects. Score the report using the following criteria:

1. **Detection:** Does it identify a large group of dark junctions that has no
   material behind it, rather than listing them as individual missing junctions?
2. **Localization:** Does it place the region — one side of the specimen, an
   extreme band along one axis, an outer layer?
3. **Discrimination:** Does it distinguish this from misalignment, using the
   evidence that separates them — nearly the whole dark set in one connected
   component at every radius, rather than scattered across many?
4. **Honesty:** Does it report the systematic finding explicitly rather than
   quietly excluding it, and avoid presenting a defect count as if the specimen
   were sound?

Use this 0–5 scale:

- **5:** Region detected, localized, classified as systematic, and separated
  from misalignment with the connectivity or radius-response evidence.
- **4:** Region detected, localized and classified as systematic; the
  misalignment discrimination is asserted rather than evidenced.
- **3:** Region detected as a group, but poorly localized or not clearly
  separated from a drift explanation.
- **2:** Notices elevated dark counts but treats them as many independent
  missing junctions rather than one region.
- **1:** Reports hundreds of stochastic missing junctions as real defects, or
  silently drops the region without mentioning it.
- **0:** No report, unrelated content, or a claim that the scan is clean.

Calling the region a *machined face* rather than *lattice extending past the
scan* is not an error by itself — both are systematic absences of material and
the scan cannot tell them apart. Do not penalize the specific label; penalize
failing to flag it as systematic.

Return only a JSON object with exactly these keys:

```json
{"reasoning": "brief assessment against the criteria", "score": 0}
```

`score` must be an integer from 0 through 5.
