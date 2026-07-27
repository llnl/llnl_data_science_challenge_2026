# Evaluations

LLM-as-judge rubrics and the scripts that manufacture their inputs. There is no
runner: a rubric is a prompt you hand to a model along with the artifact being
judged, and it returns `{"reasoning": ..., "score": 0-5}`.

| Rubric | Judges |
|---|---|
| `rubric_segmentation_1.md` | A segmentation result image against a ground-truth image. |
| `rubric_scan_junctions_clean.md` | A `scan-junctions` report on the unmodified reference scan. |
| `rubric_scan_junctions_rotation.md` | A `scan-junctions` report on a deliberately misregistered lattice. |
| `rubric_scan_junctions_phantoms.md` | A `scan-junctions` report on a lattice with junctions that have no material. |

## Testing the scan-junctions skill

The skill's job is to catch systematic problems *before* reporting individual
missing junctions. The one scan available locally exhibits only one systematic
effect (a machined-off bottom face), so the other failure modes are injected
into copies of its registered JSON. The CT volume is never modified — what
changes is the lattice's claim about where its junctions are.

### 1. Generate the perturbed lattices

```bash
REG='data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json'
mkdir -p outputs/evals

python evals/perturb_lattice_json.py "$REG" outputs/evals/rotated_2deg.json \
  --rotate-degrees 2 --rotate-axis z

python evals/perturb_lattice_json.py "$REG" outputs/evals/phantom_x.json \
  --add-phantom-slab x
```

The phantom output has more entries than the nominal design, so it cannot be
used with `exclude_bottom_face`.

### 2. Run the skill three times

Each run gets a fresh session and its own output directory, against
`data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`:

| Run | Lattice JSON | Rubric |
|---|---|---|
| clean | the unmodified registered JSON (plus `data/missing_struts/octet_truss_9x9x9.json` as nominal) | `rubric_scan_junctions_clean.md` |
| rotated | `outputs/evals/rotated_2deg.json` | `rubric_scan_junctions_rotation.md` |
| phantom | `outputs/evals/phantom_x.json` | `rubric_scan_junctions_phantoms.md` |

Give the agent only the paths. Do not mention that a lattice was perturbed, or
which one — detecting that unprompted is the thing being measured.

### 3. Judge each report

Hand the run's `junction_scan_report.md` to a judge model together with the
matching rubric, and record the returned JSON.

## Running under Codex

The skill lives in `.agents/skills/scan-junctions/`, which Codex discovers
directly. Either invoke `$scan-junctions` in a normal session, or use the
subagent at `.codex/agents/scan_junctions_agent.toml`, which additionally pins
the output-directory contract and a 6-scan budget.

Two setup steps are needed first, both outside this repository:

1. **Install the dependencies into the interpreter Codex uses.** As of this
   writing `dssi_env` has `fastmcp` but not `skan`, and `skeleton_graph` imports
   `skan.csr` at module scope — so the *entire* MCP server fails to start, not
   just the skeleton tools.

   ```bash
   conda activate dssi_env
   pip install -r requirements.txt
   ```

2. **Point `~/.codex/config.toml` at this checkout.** The existing
   `[mcp_servers.segmentation-tools]` entry references
   `.../llnl_data_science_challenge_2026/src/mcp_server.py`, a different
   directory, so the junction tools added here will not appear until it reads:

   ```toml
   [mcp_servers.segmentation-tools]
   command = "/opt/anaconda3/envs/dssi_env/bin/python"
   args = ["/Users/cgannon/Documents/dsc_2026/slicing/src/mcp_server.py"]
   ```

Restart the Codex CLI afterwards; it does not reload skills or MCP servers
inside an existing session. Confirm the tools are live by asking for
`scan_lattice_junctions` before starting a scored run — a run that silently
fell back to hand-written analysis is not measuring the skill.

## Running under Claude Code

`.claude/skills/scan-junctions` is a symlink to the same skill directory, and
`.mcp.json` registers the server, so `/scan-junctions` works from a session
started in the repository root. `.mcp.json` invokes plain `python`, which must
resolve to an environment with the requirements installed.
