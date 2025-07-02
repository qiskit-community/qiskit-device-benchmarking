# Qiskit Device Benchmarking

*Qiskit Device Benchmarking* is a respository for code to run various device level benchmarks through Qiskit. The repository endevours to accomplish several goals, including, but not limited to:
- Code examples for users to replicate reported benchmarking metrics through the Qiskit backend. These will likely be notebooks to run code in the [Qiskit Experiments](https://github.com/Qiskit-Extensions/qiskit-experiments) repo, but some of the code may reside here.
- More in-depth benchmarking code that was discussed in papers and has not been integrated into Qiskit Experiments.
- Fast circuit validation tests.

The repository is not intended to define a benchmark standard. This code base is not guaranteed to be stable and may have breaking changes. 

# Structure

At the top level we have notebooks that gives users examples on how to run various benchmarks.
- [Notebooks](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/notebooks): Jupyter notebooks for running benchmarks

Under a top level folder `qiskit_device_benchmarking` we have repository code files that can be imported from python:
- [Utilities](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/qiskit_device_benchmarking/utilities): Benchmarking utility/helper code not found elsewhere in qiskit. If these prove useful they will be pushed into standard qiskit or qiskit-experiments.
- [Benchmarking Code](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/qiskit_device_benchmarking/bench_code): General folder for benchmarking code, which may include standalone code and extensions to qiskit-experiments for custom benchmarks.
- [Verification](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/qiskit_device_benchmarking/verification): Fast verification via mirror circuits using a command line program.

# Paper Code

For clarity here we provide links from various papers to the code in this repo. Not necessarily the exact code used in these manuscripts, but representative of what was run.

- [Layer Fidelity](https://arxiv.org/abs/2311.05933): David C. McKay, Ian Hincks, Emily J. Pritchett, Malcolm Carroll, Luke C. G. Govia, Seth T. Merkel. Benchmarking Quantum Processor Performance at Scale (2023). [Code](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/notebooks/layer_fidelity.ipynb)
- [Mirror QV](https://arxiv.org/abs/2303.02108): Mirko Amico, Helena Zhang, Petar Jurcevic, Lev S. Bishop, Paul Nation, Andrew Wack, David C. McKay. Defining Standard Strategies for Quantum Benchmarks (2023). [Code](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/qiskit_device_benchmarking/bench_code/mrb)
- [Mid-circuit measurement RB](https://arxiv.org/abs/2207.04836): Luke C. G. Govia, Petar Jurcevic, Christopher J. Wood, Naoki Kanazawa, Seth T. Merkel, David C. McKay. A randomized benchmarking suite for mid-circuit measurements (2022). [Code](notebooks/mcm_rb.ipynb)
- [Dynamic circuits RB](https://arxiv.org/abs/2408.07677): Liran Shirizly, Luke C. G. Govia, David C. McKay. Randomized Benchmarking Protocol for Dynamic Circuits (2024). [Code](notebooks/dynamic_circuits_rb.ipynb)
- [Clifford Benchmarking](https://arxiv.org/abs/2503.05943): Seth Merkel, Timothy Proctor, Samuele Ferracin, Jordan Hines, Samantha Barron, Luke C. G. Govia, David McKay. When Clifford benchmarks are sufficient; estimating application performance with scalable proxy circuits (2025). [Code](notebooks/clifford_xeb_lf.ipynb)

# Installation

```
git clone git@github.com:qiskit-community/qiskit-device-benchmarking.git
cd qiskit-device-benchmarking
pip install .
```

# Lint

```
pip install ruff
ruff check      # Lint files
ruff format     # Format files
```

# Contribution Guidelines

Please open a github issue or pull request if you would like to contribute.

# License

[Apache License 2.0](LICENSE.txt)

# Acknowledgements

Portions of the code in this repository was developed via sponsorship by the Army Research Office ``QCISS Program'' under Grant Number W911NF-21-1-0002. 
