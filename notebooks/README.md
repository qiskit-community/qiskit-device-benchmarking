# Qiskit Device Benchmarking Notebooks

This folder contains example notebooks for running benchmarks.

- [Layer Fidelity](layer_fidelity.ipynb): Example notebook for running the layer fidelity using the generation code in qiskit-experiments. In this particular example the code uses the qiskit reported errors to guess the best chain of qubits for running the layer fidelity. However, it can be easily adjusted to run on an arbitrary chain.

- [Layer Fidelity Single Chain](layer_fidelity_single_chain.ipynb): Example notebook for running a layer fidelity experiment on a single chain. The default chain is the one reported on qiskit (100Q long chain) but this notebook can be easily modified to run on any arbitrary chain. A nice feature of this notebook is that it allows the user to find the best subchain within a larger chain, that is, if layer fidelity is run on a 100Q long chain, the user can easily find the best x = [4,5,6,...,98,99,100] long subchain within that chain. This notebook uses helper functions from module `layer_fidelity_utils.py` and should be less code heavy.

- [Bell State Tomography](bell_state_tomography.ipynb): Example notebook for running parallel state tomography using qiskit-experiments.

- [Device RB](device_rb.ipynb): Example notebook for running full device 2Q RB and Purity RB.

- [Device (Direct) RB](device_rb_w_lf.ipynb): Example notebook for running full device 2Q RB using layer fidelity (direct RB). This reduces the number of single qubit gates per 2Q gate in RB. Includes an example of running unitary layer RB which allows for RZZ gate benchmarking.

- [System Characterization](system_char.ipynb): Notebook to do general (non-RB) characterization of a system. Runs Coherence (T1/T2), measurement fidelity, hellinger fidelities of Bell states produced with repeated two-qubit gates and ZZ.

- [Extract Benchmarks](extract_benchmarks.ipynb): Example notebook for extracting and plotting benchmarks and properties from a list of devices. This information includes LF, EPLG, 2Q errors, 1Q errors, T1s, T2s, and readout errors, but can be easily modified to include any other properties.

- [MCM RB](mcm_rb.ipynb): Example notebook for running Mid-circuit measurement RB experiment.

- [Dynamic circuits RB](dynamic_circuits_rb.ipynb): Example notebook for running dynamic circuits RB experiment.
  
- [Quantum Volume](quantum_volume.ipynb): Example notebook for running Quantum Volume circuits.

- [Layer Fidelity Placement](layer_fidelity_placement.ipynb): Example notebook of using layer fidelity to build an updated error map of the device that is more reflective of layered circuits. Also gives an example of a heuristic algorithm for finding the best N-qubit chain based on the error map.

- [Clifford Benchmarking](clifford_xeb_lf.ipynb): Notebook that runs Clifford benchmarking of a fully entangled brickwork circuit. First, the code uses layer fidelity to build an updated error map (similar to [Layer Fidelity Placement](layer_fidelity_placement.ipynb)), finds the best chain and runs the Clifford benchmark on that chain for the full circuit. Optionally, the user can run a non-Clifford version of the same circuit and analyze the output with XEB. 

