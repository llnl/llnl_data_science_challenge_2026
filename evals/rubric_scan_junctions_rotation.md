# Scan-Junctions Evaluation Rubric: Rotated Lattice

The attached `junction_scan_report.md` was produced by the `scan-junctions`
skill run against a **deliberately misregistered** lattice: the registered JSON
was rotated about the lattice centroid by a small angle, displacing outer
junctions by roughly 15–20 voxels while leaving the centre nearly fixed. The CT
volume itself is untouched and is the reference specimen, which contains exactly
2 genuinely missing junctions.

Generate the input with:

```bash
python evals/perturb_lattice_json.py REGISTERED.json rotated_2deg.json \
  --rotate-degrees 2 --rotate-axis z
```

The correct outcome is that the report identifies the misalignment and declines
to report a defect list. Score the report using the following criteria:

1. **Detection:** Does it recognize the lattice as misaligned or misregistered
   with the scan, rather than treating the dark junctions as defects?
2. **Evidence:** Does it cite the signal that actually shows this — the dark
   set staying fragmented rather than forming one region (largest component
   never above 55% of it), the dark count collapsing as the sampling radius
   grows, and/or overlay images showing markers sitting off the lattice?
3. **Restraint:** Does it withhold a stochastic missing-junction list, or clearly
   mark any such list as untrustworthy given the alignment failure?
4. **Honesty:** Does it state the alignment verdict plainly (FAIL or UNCERTAIN)
   and avoid implying that checks it did not complete had passed?

Use this 0–5 scale:

- **5:** Alignment failure detected and named, supported by the radius response
  or overlays, with no defect list presented as trustworthy.
- **4:** Failure detected and well supported; minor gaps in evidence or wording.
- **3:** Failure detected, but the evidence is thin or the report still leans on
  a candidate list without adequate caveat.
- **2:** Notices something is wrong (high dark counts) but misattributes it, for
  example calling it a large missing region or a print defect.
- **1:** Reports the dark junctions as missing-junction defects. This is the
  failure the skill exists to prevent, regardless of how polished the report is.
- **0:** No report, unrelated content, or a claim that the scan is clean.

A report that reaches the right verdict for a stated wrong reason scores at most
3. A report claiming hundreds of missing junctions scores 1 even if every number
in it is accurate.

Return only a JSON object with exactly these keys:

```json
{"reasoning": "brief assessment against the criteria", "score": 0}
```

`score` must be an integer from 0 through 5.
