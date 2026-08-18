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

- **`clops/`** — CLOPS benchmark (Circuit Layer Operations Per Second at utility scale). Supports `twirled`, `parameterized`, and `instantiated` circuit modes; parses execution spans from job metadata. Implements the CLOPS_h spec (`CLOPS_h_v3.tex`, untracked at repo root):
  - Qubit set is a connected 1D chain via `select_qubit_set`, in precedence order: explicit `qubits=` → backend-reported `lf_<width>` chain (`layer_fidelity_utils.get_lf_chain`) → topology-only `graph_utils.longest_path_of_length`. The source is recorded in `job_attributes['qubit_set_source']` because the spec requires reporting whether the set was externally specified.
  - `chain_to_layers` produces the canonical two-sublayer decomposition (L1 even bonds, L2 odd bonds), cycled `L1, L2, L1, L2, …` over depth. This deliberately matches `layer_fidelity_utils.run_lf_chain` so a CLOPS number and an EPLG number describe the same layers. A 100Q chain has 99 bonds, so L1 takes the 50 even bonds (covering all 100 qubits, none idle) and L2 the 49 odd bonds (leaving both chain endpoints idle).
  - In `twirled` mode the submitted circuit is unparameterized (one fixed `SX` per qubit per layer); the Sampler's gate twirling inserts the `rz-sx-rz-sx-rz` frame ahead of the 2Q gates. That `SX` layer is *extra* relative to the spec and is kept on purpose — gate twirling only covers qubits active in a box, so removing it would leave the chain endpoints idling in L2 with no 1Q gate at all, which the spec requires. Don't "optimize" it away.
  - `layer_order` (`2q_first` default, `1q_first`) toggles barrier placement within a layer for A/B testing; both produce identical depth and gate counts.
  - `clops_label()` implements the reporting rule: bare `CLOPS` only at the canonical N=D=S=100 point, else `CLOPS_h(N, D, S)`. `report()` returns the spec's required reporting fields and warns if `lfoc_declared` was never set.

- **`mirror_test/`** — Estimator-based mirror test using a Trotterized 1D Ising model circuit. The circuit is its own inverse, so the expected output is the identity. Supports `eplg_chain`, `vf2_optimal`, or custom qubit path strategies.

- **`utilities/`** — Shared helpers:
  - `sampling_utils.py` — `EdgeGrabSampler` and `SingleQubitSampler` for generating random Clifford layers in RB experiments
  - `layer_fidelity_utils.py` — Wraps qiskit-experiments `LayerFidelity`; auto-selects the 2Q gate from basis gates (`ecr`/`cz`/`cx`). `get_lf_chain(backend, n)` reads the backend-reported `lf_<n>` chain from `properties().general_qlists`, returning `None` when absent (fake backends without chains expose an empty list, so this is the normal path). `clops.get_reported_chain` wraps it defensively for backends whose properties object lacks the attribute entirely. Importing this module pulls in pandas, matplotlib, and qiskit-experiments, so import it lazily from lightweight modules — `clops_benchmark.py` does.
  - `clifford_utils.py` — Stabilizer phase computation for deriving target bitstrings
  - `graph_utils.py` — rustworkx path helpers for finding best qubit chains by error. `path_to_edges` is the canonical "validate a chain against a coupling map" helper — it raises `ValueError` rather than returning a flag, takes a *list* of paths (callers wrap a single chain and index `[0]`), and nests its return only for paths longer than 2, so a 2-qubit "chain" breaks that indexing convention. `longest_path_of_length` is the deterministic topology-only chain search (no error data, honors a requested width, bounded by `max_steps` since longest-simple-path is exponential in general).
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
- Tests use `FakeFez` from `qiskit_ibm_runtime.fake_provider` for offline execution without real hardware access. `FakeFez` publishes `lf_4` … `lf_100` in `properties().general_qlists`, so the reported-EPLG-chain path is exercisable offline and is the default there; patch `clops.get_reported_chain` to test the topology fallback.
- `pytest` has pre-existing failures unrelated to any given change: `test_chsh_experiment` and `test_mirror_qv` fail on a matplotlib font-cache `PermissionError` in sandboxed environments (set `MPLCONFIGDIR` to a writable dir to avoid it), and `test_mirror_rb` / `test_purity_rb` fail on qiskit-experiments API drift. Check the baseline before attributing a failure to your work.

### Spec-driven work

`CLOPS_h_v3.tex` at the repo root is an untracked paper draft that specifies the CLOPS benchmark protocol. When changing `clops/`, check it for the intended behavior — several requirements (qubit-set provenance reporting, the canonical layer decomposition, the `CLOPS_h(N,D,S)` labelling rule, layer-fidelity operating conditions) exist as prose there and are implemented in code, so the two need to stay in sync.
