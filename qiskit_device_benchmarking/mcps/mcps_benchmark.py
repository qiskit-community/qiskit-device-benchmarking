# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals

from typing import Optional

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.ibm_backend import IBMBackend

"""
MCPS (Maximum Circuits Per Second) benchmark

MCPS measures the device-level speed ceiling: the maximum rate at which a
quantum processor can cycle through a minimal circuit — a Hadamard gate on
every qubit followed by measurement — plus the standard rep delay.
This is the hard upper bound on circuit throughput; no higher-level benchmark
can exceed it.

The benchmark uses a single-circuit job replicated 1000 times with 1000 shots
per circuit to minimize control electronics overhead and amortize any
per-job latency, yielding a measurement of sustained QPU throughput.

Two execution paths are provided:
  "executor": Uses the Executor primitive with QuantumProgram (default).
  "sampler":  Uses SamplerV2 as a fallback for systems without Executor support.

MCPS = total_circuits_executed / total_QPU_wall_time_seconds
"""


def create_mcps_circuit(backend, qubits: Optional[list] = None) -> QuantumCircuit:
    """Create the minimal MCPS circuit: H on every qubit, then measure all qubits.

    Args:
        backend: The IBM backend to target.
        qubits: Optional list of physical qubit indices to include. Defaults to
                all qubits on the device.

    Returns:
        A transpiled QuantumCircuit with an H gate and measurement on every included qubit.
    """
    if qubits is None:
        qubits = list(range(backend.num_qubits))
    n_qubits = len(qubits)
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return transpile(qc, backend, initial_layout=qubits, layout_method="trivial")


def run_mcps_executor(
    backend: IBMBackend,
    num_circuits: int = 1000,
    shots: int = 1000,
    qubits: Optional[list] = None,
) -> object:
    """Run the MCPS benchmark using the Executor primitive.

    A single minimal circuit is swept num_circuits times in one job via
    circuit_arguments of shape (num_circuits, 0), maximizing QPU utilization
    by minimizing per-circuit classical overhead.

    Args:
        backend: The IBM backend to benchmark.
        num_circuits: Number of circuit executions (replications).
        shots: Number of shots per circuit execution.
        qubits: Optional list of physical qubit indices. Defaults to all device qubits.

    Returns:
        A RuntimeJobV2 job handle.
    """
    from qiskit_ibm_runtime import Executor, QuantumProgram

    circuit = create_mcps_circuit(backend, qubits=qubits)
    program = QuantumProgram(shots=shots)
    program.append_circuit_item(circuit, circuit_arguments=np.empty((num_circuits, 0)))

    executor = Executor(mode=backend)
    return executor.run(program)


def run_mcps_sampler(
    backend: IBMBackend,
    num_circuits: int = 1000,
    shots: int = 1000,
    qubits: Optional[list] = None,
) -> object:
    """Run the MCPS benchmark using the SamplerV2 primitive.

    Submits num_circuits copies of the minimal circuit as separate PUBs in a
    single session job. Use this path when Executor is not available.

    Args:
        backend: The IBM backend to benchmark.
        num_circuits: Number of circuit executions.
        shots: Number of shots per circuit execution.
        qubits: Optional list of physical qubit indices. Defaults to all device qubits.

    Returns:
        A RuntimeJobV2 job handle.
    """
    circuit = create_mcps_circuit(backend, qubits=qubits)
    pubs = [(circuit,)] * num_circuits

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        job = sampler.run(pubs, shots=shots)

    return job


def calculate_mcps(job, num_circuits: int, shots: int, primitive: str = "executor") -> dict:
    """Calculate MCPS from a completed benchmark job.

    Args:
        job: The completed job returned by run_mcps_executor or run_mcps_sampler.
        num_circuits: The number of circuits that were executed.
        shots: The number of shots per circuit.
        primitive: "executor" (default) or "sampler", controls which result
                   timing API is used.

    Returns:
        A dict with keys: mcps, num_circuits, shots, total_time_seconds.
    """
    result = job.result()  # block until complete before reading metrics

    metrics = job.metrics()
    if "circuits_execution_time_ns" in metrics:
        total_time = metrics["circuits_execution_time_ns"] / 1e9
    else:
        if primitive == "executor":
            total_time = result.timing.duration
        else:
            from qiskit_ibm_runtime.execution_span import ExecutionSpans

            execution_spans: ExecutionSpans = result[0].metadata["execution"][
                "execution_spans"
            ]
            spans = execution_spans.sort()
            total_time = (spans.stop - spans[0].start).total_seconds()

    mcps = (num_circuits * shots) / total_time

    return {
        "mcps": round(mcps),
        "num_circuits": num_circuits,
        "shots": shots,
        "total_time_seconds": total_time,
    }


class MCPSBenchmark:
    """Run the MCPS (Maximum Circuits Per Second) benchmark.

    Measures the device-level ceiling throughput by running a large batch of
    minimal circuits (init + measure, no gates) on the target backend.

    Example usage::

        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_device_benchmarking.mcps import MCPSBenchmark

        service = QiskitRuntimeService()
        bench = MCPSBenchmark(service, "ibm_boston")
        print(bench.mcps())  # {"mcps": 3500, ...}

    Args:
        service: An authenticated QiskitRuntimeService instance.
        backend_name: Name of the backend to benchmark.
        num_circuits: Number of circuit replications per job. Default 1000.
        shots: Shots per circuit execution. Default 1000.
        primitive: "executor" (default) to use the Executor/QuantumProgram API,
                   or "sampler" to use SamplerV2.
        qubits: Optional list of physical qubit indices. Defaults to all device qubits.
    """

    def __init__(
        self,
        service: QiskitRuntimeService,
        backend_name: str,
        num_circuits: int = 1000,
        shots: int = 1000,
        primitive: str = "executor",
        qubits: Optional[list] = None,
    ):
        backend = service.backend(backend_name)
        self._num_circuits = num_circuits
        self._shots = shots
        self._primitive = primitive
        self._backend_name = backend_name

        if primitive == "executor":
            self.job = run_mcps_executor(backend, num_circuits=num_circuits, shots=shots, qubits=qubits)
        elif primitive == "sampler":
            self.job = run_mcps_sampler(backend, num_circuits=num_circuits, shots=shots, qubits=qubits)
        else:
            raise ValueError(f"'primitive' must be 'executor' or 'sampler', got {primitive!r}")

    def mcps(self) -> dict:
        """Return the MCPS result dict after the job completes.

        Returns:
            Dict with keys: mcps, num_circuits, shots, total_time_seconds.
        """
        return calculate_mcps(self.job, self._num_circuits, self._shots, primitive=self._primitive)
