# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Clifford utilities supplementing the ones in qiskit-experiments."""

from qiskit.quantum_info import Clifford
from qiskit.circuit import QuantumCircuit


def compute_target_bitstring(circuit: QuantumCircuit) -> str:
    """For a Pauli circuit C, which consists only of Clifford gates, compute C|0>.
    Args:
        circuit: A Pauli QuantumCircuit.
    Returns:
        Target bitstring.
    """
    # target string has a 1 for each True in the stabilizer half of the phase vector
    target = "".join(
        ["1" if phase else "0" for phase in Clifford(circuit).stab_phase[::-1]]
    )
    return target
