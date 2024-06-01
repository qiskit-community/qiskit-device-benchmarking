# Qiskit Device Benchmarking Notebooks

This folder contains example notebooks for running benchmarks.

- [Layer Fidelity](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/layer_fidelity.ipynb): Example notebook for running the layer fidelity using the generation code in qiskit-experiments. In this particular example the code uses the qiskit reported errors to guess the best chain of qubits for running the layer fidelity. However, it can be easily adjusted to run on an arbitrary chain.

- [Bell State Tomography](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/bell_state_tomography.ipynb): Example notebook for running parallel state tomography using qiskit-experiments.

- [Device RB](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/device_rb.ipynb): Example notebook for running full device 2Q RB and Purity RB.

- [Coherence and Proxy Bell](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/bell_tphi.ipynb): Runs T1/T2 across the device and looks at hellinger fidelities of Bell states produced with repeated two-qubit gates. 
