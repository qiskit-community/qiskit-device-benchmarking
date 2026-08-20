# MCPS Benchmark

This benchmark measures the Maximum Circuits Per Second (MCPS) of a quantum
computing system — the device-level speed ceiling representing the fastest
possible circuit execution rate. MCPS is defined as the sustained rate at which
the system can execute the most minimal circuit: a single layer of Hadamard
gates on all qubits followed by measurement, with the standard rep delay
between circuits.

MCPS sets a hard upper bound on throughput that no higher-level benchmark can
exceed. It captures state preparation, gate execution, measurement, and the
inter-circuit delay together, reflecting the true minimum cycle time of the
device.

To measure sustained throughput rather than transient performance, the benchmark
replicates the circuit 1000 times with 1000 shots per circuit in a single job,
minimizing control electronics overhead and amortizing any per-circuit latency.
Timing of the benchmark is at the QPU level, actual circuit execution time, 
and does not consider system overheads above the device level.

## Example

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_device_benchmarking.mcps.mcps_benchmark import MCPSBenchmark

service = QiskitRuntimeService()

# Run with default settings via the Executor primitive (1000 circuits, 1000 shots)
bench = MCPSBenchmark(service, "your-favorite-ibm-quantum-computer")

# There is a standard Qiskit job and we can check its status, job_id, etc
print(bench.job.status())
QUEUED

# The mcps method will calculate the result once the job completes
# Note this call will block until the result is ready
print(bench.mcps())
{'mcps': 3500000, 'num_circuits': 1000, 'shots': 1000, 'total_time_seconds': 0.286}
```

## Variations

**Executor primitive (default):** Uses the `Executor`/`QuantumProgram` API to
sweep the circuit 1000 times in a single job via a zero-parameter
`circuit_arguments` array. This is the recommended path as it minimizes
classical overhead between circuit executions.

**Sampler primitive:** A fallback for systems that do not yet support the
Executor primitive. Submits the circuits as individual PUBs within a session.
Specify `primitive="sampler"` to use this path.

```python
# Use the Sampler fallback
bench = MCPSBenchmark(service, "your-favorite-ibm-quantum-computer", primitive="sampler")
```

**Custom qubit set:** By default the benchmark runs on all qubits of the device.
A specific subset of physical qubit indices can be specified with the `qubits`
argument.

```python
# Run on a specific subset of qubits
bench = MCPSBenchmark(service, "your-favorite-ibm-quantum-computer", qubits=list(range(100)))
```
