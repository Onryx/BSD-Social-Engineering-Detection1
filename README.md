
This repository provides a prototype implementation of Behavioral Synchronization Detection (BSD) for coordinated social-engineering sequence analysis.

The repository includes:

- `bsd_detection.py`: BSD scoring implementation.
- `synthetic_generator.py`: deterministic synthetic sequence generator.
- `baselines.py`: content-only and graph-only baseline implementations.
- `run_experiments.py`: five-fold experiment runner.
- `config.json`: random seed, weights, threshold, generator settings, and templates.
- `folds.csv`: generated fold assignments.
- `results/sequence_predictions.csv`: generated sequence-level predictions.
- `results/metrics_summary.csv`: generated metric summary.

## Important Limitation

The generated corpus is synthetic and should be interpreted as simulation evidence only. It does not establish real-world deployment performance. The code is intended to document the scoring pipeline, generator assumptions, fold assignment, and sequence-level outputs so that the simulation can be inspected and extended.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt# BSD-Social-Engineering-Detection1
Multi-dimensional synchronization scoring model for detecting coordinated social engineering attacks.
