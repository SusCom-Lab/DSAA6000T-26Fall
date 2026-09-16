# DSAA 6000T: Cloud and AI Infrastructure Systems

This repository provides implementations and supplementary materials for the system cases introduced in the lectures. Interested students can reproduce these cases and explore their technical details independently.

## Course information

- **Schedule:** Thursday, 10:00–12:50
- **Instructor:** Guoming Tang, Assistant Professor, Data Science and Analytics Thrust
  - Areas: cloud computing, sustainable computing, and AI infrastructure
  - Email: [guomingtang@hkust-gz.edu.cn](mailto:guomingtang@hkust-gz.edu.cn)
- **Teaching Assistant:** Jiyang Liu, PhD Student, Data Science and Analytics Thrust
  - Areas: LLM training and agent infrastructure
  - Email: [jliu171@connect.hkust-gz.edu.cn](mailto:jliu171@connect.hkust-gz.edu.cn)

## Course schedule

| Phase | Weeks | Focus |
| --- | --- | --- |
| I — Foundations | 1–2 | Cloud and AI infrastructure foundations |
| II — Runtime | 3–4 | Runtime abstractions and portability |
| III — Execution and State | 5–6 | Distributed execution and state management |
| IV — Measurement | 7–8 | Measurement, profiling, and benchmarking |
| V — Correctness and Reliability | 9–10 | Correctness, reliability, and reproducibility |
| VI — Resources and Optimization | 11–12 | Resource management and adaptive optimization |
| Project | 13 | Project presentation and research discussion |

## Course contents

| Week | Topic | Materials |
| --- | --- | --- |
| 1 | Profiling the components of a Llama 8B model | [Week 1 guide](week1/README.md) · [Jupyter notebook](week1/llama8b_component_profiling.ipynb) |
| 2 | Persistent Llama CPU/GPU services, routing, and fallback | [Week 2 guide](week2/README.md) · [Demo runner](week2/run_demo.sh) |
| 3 | Runtime primitives, scheduling policies, and cross-stream synchronization | [Week 3 guide](week3/README.md) · [Demo runner](week3/run_demo.sh) |

## How to use this repository

- Open the guide in the corresponding `weekN/` directory for each week's configuration and run instructions.
- Run the notebooks and examples in your own environment, using private configuration files such as `.env` when instructed.
- Keep model weights, generated traces, profiling results, credentials, and personal filesystem paths on your local system.
- Student contributions or sharing through this repository are not expected.
