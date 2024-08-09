# Qiskit Device Benchmarking Notebooks

This folder contains example notebooks for running benchmarks.

- [Layer Fidelity](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/layer_fidelity.ipynb): Example notebook for running the layer fidelity using the generation code in qiskit-experiments. In this particular example the code uses the qiskit reported errors to guess the best chain of qubits for running the layer fidelity. However, it can be easily adjusted to run on an arbitrary chain.

- [Layer Fidelity Single Chain](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/layer_fidelity_single_chain.ipynb): Example notebook for running a layer fidelity experiment on a single chain. The default chain is the one reported on qiskit (100Q long chain) but this notebook can be easily modified to run on any arbitrary chain. A nice feature of this notebook is that it allows the user to find the best subchain within a larger chain, that is, if layer fidelity is run on a 100Q long chain, the user can easily find the best x = [4,5,6,...,98,99,100] long subchain within that chain. This notebook uses helper functions from module `layer_fidelity_utils.py` and should be less code heavy.

- [Bell State Tomography](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/bell_state_tomography.ipynb): Example notebook for running parallel state tomography using qiskit-experiments.

- [Device RB](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/device_rb.ipynb): Example notebook for running full device 2Q RB and Purity RB.

- [Coherence and Proxy Bell](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/bell_tphi.ipynb): Runs T1/T2 across the device and looks at hellinger fidelities of Bell states produced with repeated two-qubit gates.

- [Extract Benchmarks](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/extract_benchmarks.ipynb): Example notebook for extracting and plotting benchmarks and properties from a list of devices. This information includes LF, EPLG, 2Q errors, 1Q errors, T1s, T2s, and readout errors, but can be easily modified to include any other properties.
