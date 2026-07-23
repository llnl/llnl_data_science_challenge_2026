# Segmentation Evaluation Rubric

Compare the first attached image (ground truth) with the second attached image (result). Score the result using the following criteria:

1. **Structural Integrity:** Does the result preserve the connectivity and arrangement of lattice struts?
2. **False Positives/Negatives:** Does it add noise or material absent from the ground truth, or omit struts present in it?
3. **Topology:** Are lattice nodes and junctions preserved?
4. **Noise and Artifacts:** Does the result contain artifacts not present in the clean ground truth?

Use this 0–5 scale:

- **5:** Identical to the ground truth; no missing structures or false positives.
- **4:** Excellent; only very minor differences.
- **3:** Main topology is correct, but there is noticeable noise or missing thin struts.
- **2:** Fair; significant differences such as large missing chunks.
- **1:** Major structural failure or excessive noise.
- **0:** Blank or unrelated output.

Return only a JSON object with exactly these keys:

```json
{"reasoning": "brief comparison against the criteria", "score": 0}
```

`score` must be an integer from 0 through 5.
