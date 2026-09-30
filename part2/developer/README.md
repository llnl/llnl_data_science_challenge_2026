# Stage 2a baseline

`anomaly_pipeline.py` performs a conservative first pass over a TIFF stack.
It estimates an Otsu threshold from a bounded, page-wise strided sample, writes
a max-pooled material mask to a `numpy.memmap`, and samples a registered graph's
expected struts from that compact mask. Only sustained low-support members are
written as candidates for Stage 2b visual review.

Run from the repository root:

```text
python part2/developer/anomaly_pipeline.py
```

Outputs include `output/anomaly_summary.json`, `output/diagnostic_histogram.png`,
and `../interagent_feedback/anomaly_candidates.json`. The candidate file is a
JSON list. Each item has `ID`, `type`, `location`, `confidence`, `crop_size`,
`reason`, and an explicit Stage 2b review status. Candidates are not confirmed
defects.
