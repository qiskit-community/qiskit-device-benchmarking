# Mirror Circuit Benchmark

This is meant to offer a straightforward test of Estimator primitives and the ability to
deliver accurate expectation values for utility-scale circuits. It uses Trotterized time-
evolution of a 1D Ising chain as the test circuit, but with the circuit mirrored so that
its effective action is equivalent to the identity. This makes it trivial to detect
whether the returned expectation values are accurate.

Note that with its default settings -- at the time of this writing -- executing this
benchmark requires 8-9 hours of wall clock time, and about 2.5 hours of QPU time.

## Usage example

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_device_benchmarking.mirror_test.mirror_test import submit_mirror_test, analyze_mirror_result

service = QiskitRuntimeService(channel="ibm_quantum",
                               instance="your-hub/group/project")

backend = service.backend("your-favorite-ibm-quantum-computer")

job = submit_mirror_test(backend, num_qubits=100, num_gates=2500)

# wait for job to complete execution, then...
result = job.result()
analyze_mirror_result(result, accuracy_threshold=0.1, make_plots=True)
```
