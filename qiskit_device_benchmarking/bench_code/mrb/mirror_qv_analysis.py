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
Quantum Volume analysis class.
"""

import numpy as np
from uncertainties import unumpy as unp
from uncertainties import ufloat

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.framework import (
    BaseAnalysis,
    AnalysisResultData,
    Options,
    ExperimentData,
)
from qiskit_experiments.framework.containers import ArtifactData

# import this data processor from rb_analysis
from qiskit_device_benchmarking.bench_code.mrb.mirror_rb_analysis import (
    _ComputeQuantities,
)


class MirrorQuantumVolumeAnalysis(BaseAnalysis):
    r"""A class to analyze mirror quantum volume experiments.

    # section: overview
        Calculate the success (fraction of target measured) and polarization
        Optionally calcuate an effective HOP
    """

    def _initialize(self, experiment_data: ExperimentData):
        """Initialize curve analysis by setting up the data processor for Mirror
        RB data.

        Args:
            experiment_data: Experiment data to analyze.
        """

        target_bs = []
        self.depth = None
        self.ntrials = 0
        for circ_result in experiment_data.data():
            target_bs.append(circ_result["metadata"]["target_bitstring"])
            trial_depth = circ_result["metadata"]["depth"]
            self.ntrials += 1
            if self.depth is None:
                self.depth = trial_depth
            elif trial_depth != self.depth:
                raise AnalysisError(
                    "QuantumVolume circuits do not all have the same depth."
                )

        num_qubits = self.depth

        self.set_options(
            data_processor=DataProcessor(
                input_key="counts",
                data_actions=[
                    _ComputeQuantities(
                        analyzed_quantity=self.options.analyzed_quantity,
                        num_qubits=num_qubits,
                        target_bs=target_bs,
                    )
                ],
            )
        )

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            ax(AxesSubplot): Optional. A matplotlib axis object to draw.
        """
        options = super()._default_options()
        options.plot = False
        options.ax = None
        options.calc_hop = True

        # By default, effective polarization is plotted (see arXiv:2112.09853). We can
        # also plot success probability or adjusted success probability (see PyGSTi).
        # Do this by setting options to "Success Probability" or "Adjusted Success Probability"
        options.analyzed_quantity = "Effective Polarization"

        options.set_validator(
            field="analyzed_quantity",
            validator_value=[
                "Success Probability",
                "Adjusted Success Probability",
                "Effective Polarization",
            ],
        )

        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ):
        results = []
        artifacts = []

        # Prepare for fitting
        self._initialize(experiment_data)

        processed = self.options.data_processor(experiment_data.data())
        yvals = unp.nominal_values(processed).flatten()

        success_prob_result = AnalysisResultData(
            "mean_success_probability",
            value=ufloat(nominal_value=np.mean(yvals), std_dev=np.std(yvals)),
            quality="good",
            extra={
                "depth": self.depth,
                "trials": self.ntrials,
            },
        )

        artifacts.append(
            ArtifactData(
                name="data",
                data=yvals,
            )
        )

        if self.options.plot:
            # figure out what to do
            figures = None
        else:
            figures = None

        results.append(success_prob_result)

        return results + artifacts, figures
