# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install .

# Run all tests
pytest

# Run a single test
pytest tests/test_benchmarks.py::test_mirror_rb

# Lint (used in CI)
ruff check .

# Format
ruff format .
```

## Architecture

This repo implements IBM device-level benchmarks as a Qiskit extension. The primary user interface is the `notebooks/` directory; the `qiskit_device_benchmarking/` package provides the underlying experiment classes and utilities.

### Package layout

- **`bench_code/`** — Custom experiment classes extending `qiskit-experiments` (`BaseExperiment`/`BaseAnalysis`). Each subdirectory is a self-contained benchmark:
  - `bell/` — CHSH Bell inequality test and parallel Bell state tomography
  - `mrb/` — Mirror RB (arXiv:2303.02108) and Mirror Quantum Volume; circuits are mirrored so the target output bitstring is known, enabling pass/fail without tomography
  - `prb/` — Purity RB; adds post-rotation circuits to extract Tr(ρ²) alongside standard EPC
  - `mcm_rb/` — Mid-circuit measurement RB (arXiv:2207.04836)
  - `dynamic_circuits_rb/` — Dynamic circuits RB (arXiv:2408.07677); shares `SubDecayFit` with `mcm_rb/`

- **`clops/`** — CLOPS benchmark (Circuit Layer Operations Per Second at utility scale). Supports `twirled`, `parameterized`, and `instantiated` circuit modes; parses execution spans from job metadata.

- **`mirror_test/`** — Estimator-based mirror test using a Trotterized 1D Ising model circuit. The circuit is its own inverse, so the expected output is the identity. Supports `eplg_chain`, `vf2_optimal`, or custom qubit path strategies.

- **`utilities/`** — Shared helpers:
  - `sampling_utils.py` — `EdgeGrabSampler` and `SingleQubitSampler` for generating random Clifford layers in RB experiments
  - `layer_fidelity_utils.py` — Wraps qiskit-experiments `LayerFidelity`; auto-selects the 2Q gate from basis gates (`ecr`/`cz`/`cx`)
  - `clifford_utils.py` — Stabilizer phase computation for deriving target bitstrings
  - `graph_utils.py` — rustworkx path helpers for finding best qubit chains by error
  - `gate_map.py` — Coupling map visualizer for Eagle/HeronR1/HeronR2 topologies
  - `file_utils.py` — YAML import/export with timestamped filenames

- **`verification/`** — CLI scripts for fast device validation:
  - `fast_bench.py` — Reads `config.yaml`, runs MirrorQV, writes YAML output
  - `bench_analyze.py` — Ingests fast_bench YAML, produces PDF plots (`-f <yaml> -v max --plot`)
  - `fast_layer_fidelity.py` — Runs LayerFidelity on the 100Q reported chain
  - `fast_count.py` / `count_analyze.py` — Parallel CHSH across all qubit pairs

### Key design patterns

- All experiment classes follow the qiskit-experiments pattern: a class extending `BaseExperiment` (defining `circuits()`) paired with a class extending `BaseAnalysis` (defining `_run_analysis()`).
- Qubit chain selection is a recurring concern: `get_optimal_path.py` uses `VF2Layout` with an `ErrorMap` built from backend properties (T1, T2, 2Q gate errors, faulty qubits) to find optimal chains.
- Circuits are transpiled via `StagedPassManager` or `generate_preset_pass_manager`; dynamical decoupling is added via ALAP scheduling.
- Tests use `FakeFez` from `qiskit_ibm_runtime.fake_provider` for offline execution without real hardware access.
