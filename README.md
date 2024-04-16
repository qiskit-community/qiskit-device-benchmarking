# Qiskit Device Benchmarking

*Qiskit Device Benchmarking* is a respository for code to run various device level benchmarks through Qiskit. The repository endevours to accomplish several goals, including, but not limited to:
- Code examples for users to replicate reported benchmarking metrics through the Qiskit backend. These will likely be notebooks to run code in the [Qiskit Experiments](https://github.com/Qiskit-Extensions/qiskit-experiments) repo, but some of the code may reside here.
- More in-depth benchmarking code that was discussed in papers and has not been integrated into Qiskit Experiments.
- Fast circuit validation tests.

The repository is not intended to define a benchmark standard. This code base is not guaranteed to be stable and may have breaking changes. 

# Structure

- [Notebooks](https://github.com/qiskit-community/qiskit-device-benchmarking/notebooks): Jupyter notebooks for running benchmarks
- [Utilities](https://github.com/qiskit-community/qiskit-device-benchmarking/utilities): Benchmarking utilities not found elsewhere in qiskit. If these prove useful they will be pushed into standard qiskit or qiskit-experiments.
- [Extensions](https://github.com/qiskit-community/qiskit-device-benchmarking/extensions): Code extensions to qiskit-experiments for custom benchmarks.
- [Verification](https://github.com/qiskit-community/qiskit-device-benchmarking/verification): Fast verification code (tbd)

# Paper Code

For clarity here we provide links from various papers to the code in this repo. Not necessarily the exact code used in these manuscripts, but representative of what was run.

- [Layer Fidelity](https://arxiv.org/abs/2311.05933): David C. McKay, Ian Hincks, Emily J. Pritchett, Malcolm Carroll, Luke C. G. Govia, Seth T. Merkel. Benchmarking Quantum Processor Performance at Scale (2023). [Code](https://github.com/qiskit-community/qiskit-device-benchmarking/notebooks/lf.ipynb)

# Contribution Guidelines

Please open a github issue or pull request if you would like to contribute.

# License

[Apache License 2.0](LICENSE.txt)

# Acknowledgements

Portions of the code in this repository was developed via sponsorship by the Army Research Office ``QCISS Program'' under Grant Number W911NF-21-1-0002. 
