# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Variants of the Bell experiments

.. currentmodule:: qiskit_experiments_internal.library.quantum_volume

Classes
=======
.. autosummary::
    ::undoc-members:

    Bell

"""

from .bell_experiment import BellExperiment
from .bell_experiment import BellAnalysis
from .bell_experiment import CHSHAnalysis
from .bell_experiment import CHSHExperiment

__all__ = [
    BellExperiment,
    BellAnalysis,
    CHSHAnalysis,
    CHSHExperiment,
]
