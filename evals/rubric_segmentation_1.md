 # Segmentation Slice Evaluation Rubric

Compare the first image (ground truth) with the second image (segmentation result).

Evaluate the result using these criteria:

1. **Structural Integrity:** Does the result preserve the lattice strut connectivity shown in the ground truth?
2. **False Positives and Negatives:** Identify over-segmentation, such as extra material or noise, and under-segmentation, such as missing struts.
3. **Topology:** Are lattice nodes and junctions preserved?
4. **Noise and Artifacts:** Does the result contain noise, holes, disconnected regions, or artifacts not present in the clean ground truth?

Assign one integer score from 0 to 5:

- **5:** Identical or nearly identical; no meaningful missing structures or false positives.
- **4:** Excellent; only very minor differences.
- **3:** Main topology is correct, but there is noticeable noise or some thin/marginal struts are missing.
- **2:** Fair; significant differences, such as large missing regions, incorrect connections, or substantial noise.
- **1:** Major structural failure or excessive noise.
- **0:** Blank, unrelated, or unusable result.

Return exactly one JSON object and no Markdown fences:

{"reasoning":"Brief comparison of connectivity, errors, topology, and artifacts.","score":0}