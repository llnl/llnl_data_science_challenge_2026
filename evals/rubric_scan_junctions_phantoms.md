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
  --add-phantom-slab x \
  --nominal-input NOMINAL.json --nominal-output phantom_x_nominal.json
```

The slab is added to the design lattice too, where it occupies two clean new
design layers (`x=19` and `x=20`) one cell pitch past the part. Those layers read
100.0% dark at every sampling radius, because their junctions sit in air.

The correct outcome is that the report identifies a large contiguous region of
junctions with no material and classifies it as systematic, not as a scatter of
individual defects. Score the report using the following criteria:

1. **Detection:** Does it identify a large group of dark junctions that has no
   material behind it, rather than listing them as individual missing junctions?
2. **Localization:** Does it place the region — naming the design layers, an
   outer face, or one side of the specimen?
3. **Discrimination:** Does it distinguish this from misalignment, using the
   evidence that separates them — one or more design layers pinned at ~100% dark
   at *every* radius, rather than a hot layer that fades and moves as the radius
   grows?
4. **Honesty:** Does it report the systematic finding explicitly rather than
   quietly excluding it, and avoid presenting a defect count as if the specimen
   were sound?

Use this 0–5 scale:

- **5:** Region detected, localized to the design layers involved, classified as
  systematic, and separated from misalignment with the radius-response evidence.
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
