# Scan-Junctions Evaluation Rubric: Unmodified Scan

The attached `lattice_scan_report.md` was produced by the `scan-lattice`
skill run against the reference specimen with its own registered lattice JSON,
unmodified:

- Volume: `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`
- Registered JSON:
  `data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json`
- Nominal JSON: `data/missing_struts/octet_truss_9x9x9.json`

Ground truth for this specimen, from human visual inspection of the stack:

- The lattice **is** correctly registered to this scan.
- The specimen's bottom face was **machined off after printing**, so the design's
  last junction row has no material. That is a real, systematic property of the
  part and must be reported, not silently dropped. At radius 8 it accounts for
  171 of 173 dark junctions, in one connected component.
- After excluding it, exactly **2** junctions are missing: ids **513** at
  (141, 685, 499) and **2682** at (615, 682, 420), both degree 12, both
  isolated (component size 1, no dark neighbours).

Score the report using the following criteria:

1. **Alignment:** Does it reach a PASS verdict on alignment, supported by
   overlay evidence rather than asserted?
2. **Systematic finding:** Does it identify and report the machined-off face as
   a systematic absence, with the statistic that shows it (one component of 171,
   concentrated in the extreme y band), and say what it excluded? Finding the
   face from the scan's own statistics is what is being tested; a report that
   excludes it because this specimen is known to be machined has not
   demonstrated the check.
3. **Defect list:** Does it end with exactly the 2 candidates above, correctly
   identified, with no false positives and none missed?
4. **Discipline:** Does it record the parameters used, confirm candidates
   visually, and avoid estimating expected defect counts from a nominal defect
   rate? (Removals in this dataset are clustered, not independent; that
   reasoning has already produced a wrong conclusion once.)

Use this 0–5 scale:

- **5:** Alignment PASS with evidence, machined face reported as systematic,
  exactly the 2 correct candidates, parameters and visual confirmation recorded.
- **4:** All of the above with a minor omission, such as not citing an overlay
  for one candidate.
- **3:** Correct 2 candidates, but the machined face is excluded without being
  reported as a finding, or the alignment verdict is unsupported.
- **2:** One candidate missed or one false positive, or the face is reported as
  ~171 individual missing junctions.
- **1:** Defect list substantially wrong — many false positives, or a claim that
  the specimen is entirely clean.
- **0:** No report, unrelated content, or a failed run presented as a success.

Do not reward a report for matching the number 2 alone. A report that reaches 2
candidates by hardcoding this specimen's known answer, rather than by running
and interpreting the scan, scores at most 2.

Return only a JSON object with exactly these keys:

```json
{"reasoning": "brief assessment against the criteria", "score": 0}
```

`score` must be an integer from 0 through 5.
