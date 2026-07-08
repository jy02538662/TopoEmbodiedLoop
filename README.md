# TopoEmbodiedLoop

A compact open-source demo of a **topological perception-memory-control loop** for contact-rich embodied intelligence.

The goal is simple: when a robot is doing insertion, assembly, or contact-rich manipulation, it should not just keep pushing. It should detect contact risk, slow down or release early, and reuse past recovery experience.

## What this demo integrates

This simplified version combines the core ideas of five prototypes:

| Prototype idea | Public compact module | Role |
|---|---|---|
| DualWave | `modules/reliability_gate.py` | Decide which sensor stream is more trustworthy. |
| TopoWave | `modules/contact_reasoner.py` | Estimate contact states such as `contact`, `slip`, `jam`, `release`. |
| TopoClosedLoop | `TopoReasoner` + `TopoGuard` | Stabilize contact belief and trigger safe emergency modes. |
| TideMemory | `modules/topo_memory.py` | Recall similar contact episodes and bias recovery strategy. |
| VTEC | `modules/contact_controller.py` | Perform low-force insert, release, recapture, and retreat actions. |

The loop is:

```text
sensor reliability -> contact belief -> memory prior -> emergency guard -> contact action
```

## Why this matters

Modern embodied AI systems can often understand the target, but contact remains hard:

- the robot inserts slightly off-axis,
- contact turns into slip or jam,
- force rises before the policy reacts,
- recovery direction is chosen randomly,
- similar failures are not remembered.

This demo is a small reflex layer for those cases. It is meant to complement high-level VLM/VLA policies, not replace them.

## Run

```bash
python run_demo.py
```

No third-party dependency is required.

## Compared strategies

| Strategy | Meaning |
|---|---|
| `reactive` | Uses current noisy state probabilities directly. |
| `reasoner_guard` | Adds temporal contact belief and emergency guard. |
| `full_loop` | Adds episodic memory to bias release strategy. |

## Example result

A representative run over 160 synthetic contact-rich episodes:

| Strategy | Success | Avg peak force | Avg impulse | Avg jam steps |
|---|---:|---:|---:|---:|
| `reactive` | 0.0% | 105.73 | 70.27 | 6.93 |
| `reasoner_guard` | 98.8% | 55.98 | 30.12 | 0.18 |
| `full_loop` | 98.8% | 55.94 | 29.98 | 0.17 |

Compared with `reactive`, the full loop reduces peak force by about 47% and jam steps by about 98% in this synthetic setting.

## Outputs

The demo writes:

- `results/summary.csv`
- `results/summary.json`

To regenerate the comparison figure:

```bash
python make_figures.py
```

This writes:

- `results/topo_embodied_loop_comparison.svg`

## Plain-language explanation

`reactive` is like a rushed robot that keeps pushing until it sees a jam.

`reasoner_guard` is like a cautious technician that notices contact risk trends and backs off before force spikes.

`full_loop` is like an experienced technician that also remembers which release direction worked in similar scenes.

## Scope and limitations

This is the public compact version. It is intentionally small and easy to inspect.

It is not:

- a production robot controller,
- a real robot benchmark,
- a full replacement for VLA/VLM policies,
- the complete research code of the original prototypes.

It is a minimal, reproducible demonstration of the loop structure.

## Suggested citation

If you use or discuss this prototype, cite it as:

```text
TopoEmbodiedLoop: A Topological Perception-Memory-Control Loop for Contact-Rich Embodied Intelligence, v0.1.0, 2026.
```

## License

This repository is released under a **source-available non-commercial research license**.

Allowed:

- academic research,
- personal study,
- non-commercial education,
- reproducibility evaluation,
- non-commercial demos.

Not allowed without written permission:

- commercial products,
- paid services,
- internal commercial deployment,
- sublicensing or selling the code,
- claiming the architecture, benchmark, or results as your own.

See `LICENSE` for details.
