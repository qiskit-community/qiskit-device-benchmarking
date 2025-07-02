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

import numpy as np
import copy
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, CZGate, ECRGate, RZGate, SXGate
from qiskit.quantum_info import Clifford, Pauli, SparsePauliOp, random_unitary
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from qiskit_ibm_runtime import SamplerV2
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit_experiments.framework.experiment_data import ExperimentData

import qiskit_device_benchmarking.utilities.layer_fidelity_utils as lfu


"""Utilities for running large Clifford and XEB circuits"""


class SingleDecompPass(TransformationPass):
    """
    Transformation pass that converts any
    single qubit gate in the circuit into parameterized U3 form
    """

    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        single_layer = True
        idl = 0
        idq = 0
        for node in dag.op_nodes(include_directives=False):
            if node.op.num_qubits > 1:
                if single_layer:
                    single_layer = False
                    idl += 1
                    idq = 0
                continue
            if not single_layer:
                single_layer = True
            gate_blocks = SingleDecompPass.rz_sx_gate_blocks(idl, idq)
            dag.substitute_node_with_dag(
                node=node,
                wires=node.qargs,
                input_dag=SingleDecompPass.par_clifford_dag(
                    gate_blocks=gate_blocks,
                    qubits=node.qargs,
                ),
            )
            idq += 1

        return dag

    @staticmethod
    def rz_sx_gate_blocks(idl, idq):
        def pars(zi):
            return Parameter(f"l_{idl}_q_{idq}_z_{zi}")

        return [RZGate(pars(0)), SXGate(), RZGate(pars(1)), SXGate(), RZGate(pars(2))]

    @staticmethod
    def par_clifford_dag(
        gate_blocks,
        qubits=None,
        qreg=None,
        num_qubits=None,
    ):
        dag = DAGCircuit()
        if qubits:
            dag.add_qubits(qubits)
        elif qreg:
            dag.add_qreg(qreg)
        elif num_qubits:
            dag.add_qreg(QuantumRegister(num_qubits))
        else:
            dag.add_qreg(QuantumRegister(1))

        for wire in dag.wires:
            for op in gate_blocks:
                dag.apply_operation_back(op, qargs=(wire,))

        return dag


class Cliffordize:
    """
    A class that takes a circuit and converts it to a cliffordized version of itself

    As written assumes the template circuit is in the form of 2Q gates as cliffords
    and the singles can all be replaced
    """

    decomp = OneQubitEulerDecomposer(basis="ZSX")
    pauli_preps = [
        Clifford([[0, 1, 0], [1, 0, 0]]),
        Clifford([[1, 0, 1], [1, 1, 0]]),
        Clifford([[1, 0, 0], [0, 1, 0]]),
    ]

    def __init__(self, circuit, nsamples=1):
        """
        Init class

        Args:
            circuit: base template circuit
            nsamples: The number of samples
        Returns:
        """

        single_cliffs = []
        stab_base = [[1, 0], [0, 1], [1, 1]]
        ph_base = [[0, 0], [1, 0], [0, 1], [1, 1]]
        for iz in range(3):
            for ix in range(2):
                for iph in range(4):
                    cliff_table = np.zeros([2, 3], dtype=int)
                    cliff_table[0, 0:2] = stab_base[iz]
                    cliff_table[1, 0:2] = stab_base[(iz + ix + 1) % 3]
                    cliff_table[:, 2] = ph_base[iph]
                    single_cliffs.append(Clifford(cliff_table))

        self._single_cliffs = single_cliffs
        self._cliff_params = [Cliffordize._cliff_angles(c) for c in single_cliffs]

        self.nq = len(circuit.qregs[0])
        self.nsamples = nsamples
        self.base_circuit = circuit

        # initialize but will be populated by construct
        self.nlayers = None
        self.bindings = None
        self.observables = None
        self.cliff_inds = None
        self.construct()

    def get_circuit(self):
        return self.circuit

    def circ_to_layers(self, circ):
        return circ

    def random_cliff(
        self, input_support=None, output_support=None, rand_outputs_only=False
    ):
        """
        Output a set of random cliffords with pauli's and observables

        Args:
            input_support: bitstring, only put pauli's on the supported qubits
            output_support: bitstring, only put pauli's on the supported qubits
            rand_outputs_only: randomize only the outputs...otherwise both outputs and
            clifford circuit are randomized together (number set by self.nsamples)
        Returns:
            cliff_inds: list of list of clifford indices
            bindings: list of proper bindings arrays for the circuit
            observables: list of observable paulis
        """

        if input_support is not None and output_support is not None:
            raise ValueError(
                "Input support and output support cannot be simultaneously specified"
            )

        if rand_outputs_only:
            # only 1 cliffordization, so repeat nsamples times

            self._random_cliff_bindings(nrands=1)
            self.cliff_inds = [self.cliff_inds[0] for i in range(self.nsamples)]

        else:
            self._random_cliff_bindings()

        # Construct the input and output pauli's for direct fidelity estimation

        if output_support is None:
            # if the output support is not specified, work from random input pauli's and evolve to the output

            input_Paulis = [
                Cliffordize._random_pauli(self.nq, input_support)
                for i in range(self.nsamples)
            ]
            output_Paulis = []
            for samp_idx in range(self.nsamples):
                if self.nsamples == 1 or rand_outputs_only:
                    idx = ()
                else:
                    idx = (samp_idx,)

                pind = samp_idx

                cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))
                pauli = Pauli(input_Paulis[pind])

                output_P = pauli.evolve(cliff, frame="s")
                pstring = ""
                evolved_pauli = pauli.evolve(cliff, frame="s")
                for x, z in zip(evolved_pauli.x, evolved_pauli.z):
                    if x and z:
                        ptemp = "Y"
                    elif x:
                        ptemp = "X"
                    elif z:
                        ptemp = "Z"
                    else:
                        ptemp = "I"
                    pstring = ptemp + pstring

                if output_P.phase == 2:
                    pstring = "-" + pstring
                output_Paulis.append(pstring)

        else:
            # output support is specified, so specify the output pauli's and work back

            output_Paulis = [
                Cliffordize._random_pauli(self.nq, output_support)
                for i in range(self.nsamples)
            ]
            input_Paulis = []
            for samp_idx in range(self.nsamples):
                if self.nsamples == 1 or rand_outputs_only:
                    idx = ()
                else:
                    idx = (samp_idx,)

                pind = samp_idx
                cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))

                pauli = Pauli(output_Paulis[pind])
                input_Paulis.append(str(pauli.evolve(cliff.adjoint(), frame="s")))

        # Input and output pauli's are now determined
        # Mix the input and output pauli's with the first and last clifford layers
        # update input bindings
        self.input_support_all = []
        for s in range(self.nsamples):
            pauli = input_Paulis[s]
            if input_Paulis[s][0] == "-":
                pauli = input_Paulis[s][1:]
                output_Paulis[s] = "-" + output_Paulis[s]
            else:
                pauli = input_Paulis[s]
            for q, p in enumerate(reversed(pauli)):
                if p == "I" or p == "Z":
                    continue
                elif p == "X":
                    precliff = Cliffordize.pauli_preps[0]
                elif p == "Y":
                    precliff = Cliffordize.pauli_preps[1]
                postcliff = self._single_cliffs[self.cliff_inds[s][0][q]]
                newcliff = precliff.compose(postcliff)
                self.cliff_inds[s][0][q] = self._single_cliffs.index(newcliff)
            input_supp = []
            for p in reversed(pauli):
                if p == "I":
                    input_supp.append(0)
                else:
                    input_supp.append(1)
            self.input_support_all.append(input_supp)

        # update output bindings and construct observables
        for s in range(self.nsamples):
            for q, p in enumerate(reversed(output_Paulis[s])):
                if p == "I" or p == "Z" or p == "-":
                    continue
                elif p == "X":
                    postcliff = Cliffordize.pauli_preps[0]
                elif p == "Y":
                    postcliff = Cliffordize.pauli_preps[1]
                precliff = self._single_cliffs[self.cliff_inds[s][-1][q]]
                newcliff = precliff.compose(postcliff)
                self.cliff_inds[s][-1][q] = self._single_cliffs.index(newcliff)
            out = ""
            for p in output_Paulis[s]:
                if p == "X" or p == "Y" or p == "Z":
                    out += "Z"
                else:
                    out += p
            output_Paulis[s] = out

        _, _ = self._random_cliff_bindings(cliff_inds=self.cliff_inds)

        # construct observables
        self.observables = np.empty(self.nsamples, dtype=SparsePauliOp)
        for samp in range(self.nsamples):
            pauli_str = output_Paulis[samp]

            self.observables[samp] = SparsePauliOp(pauli_str)

        return self.cliff_inds, self.bindings, self.observables

    def _random_cliff_bindings(self, cliff_inds=None, nrands=None):
        """
        Output a set of random binding arrays to make this a clifford circuit

        Args:
            cliff_inds: if these are passed in then it will just redo the bindings
            nrands: if none use nsamples, but can override
        Returns:
            cliff_inds: list of list of clifford indices
            bindings: list of proper bindings arrays for the circuit
        """

        if nrands is None:
            nrands = self.nsamples

        # sample random singles
        # build a list of random cliffords
        if cliff_inds is None:
            cliff_inds = [
                [
                    [np.random.randint(24) for q in range(self.nq)]
                    for nl in range(self.nlayers)
                ]
                for s in range(nrands)
            ]

        # convert the list of random cliffords into a bindings array for the circuit
        bindings = BindingsArray(
            data={
                par: np.array(
                    [self._clifford_param_val(str(par), ci) for ci in cliff_inds]
                )
                for par in self.circuit.parameters
            }
        )

        self.bindings = bindings
        self.cliff_inds = cliff_inds

        return cliff_inds, bindings

    def random_haar(self):
        """
        Output a set of random binding arrays to make this a haar random circuit

        Args:
        Returns:
            haar_angles: list of haar angles
            bindings: list of proper bindings arrays for the circuit
        """

        # sample random singles
        haar_angles = [
            [
                [Cliffordize._haar_angles_single() for q in range(self.nq)]
                for nl in range(self.nlayers)
            ]
            for s in range(self.nsamples)
        ]

        bindings = BindingsArray(
            data={
                par: np.array(
                    [Cliffordize._haar_param_val(str(par), ha) for ha in haar_angles]
                )
                for par in self.circuit.parameters
            }
        )

        self.bindings = bindings

        return haar_angles, bindings

    def construct(self):
        """
        Convert the circuit into U3 parameteried form using the
        SingleDecompPass class

        Caveats:
        For now the circuit coming in must be in a layered form with a layer of singles
        at the beginning and end of the circuit. To do is add checking that this is the case
        """

        # In a future version of the code will make sure the 
        # circuit ordering is structured in such a way that the 
        # layering will be more obvious
        self.circuit = self.circ_to_layers(self.base_circuit)

        #convert a circuit which is singles and 2Q gates into
        #parameterized form
        pm = PassManager([SingleDecompPass()])
        self.circuit = pm.run(self.circuit)

        # number of layers in the circuit
        self.nlayers = np.max(
            [int(str(p).split("_")[1]) + 1 for p in self.circuit.parameters]
        )

    def to_pub(self, **kwargs):
        """
        Turn the cliffordized object into a pub that can
        be used for the estimator

        Args:
            kwargs which will be passed through to the transpiler
            should include
            backend: backend to run on
            initial_layout: layout on the device
        Returns:
            pub that can be sent to the estimator

        """

        circuit = self.circuit

        if self.observables is None or self.bindings is None:
            raise ValueError("Cannot run to_pub without first running random_cliff")

        if 'optimization_level' not in kwargs:
            kwargs['optimization_level'] = 1

        circuit = transpile(circuit, **kwargs)

        obs = ObservablesArray(
            [observable.apply_layout(circuit.layout) for observable in self.observables]
        )

        estimator_pub = EstimatorPub(
            circuit=circuit, observables=obs, parameter_values=self.bindings
        )

        return estimator_pub

    def to_sampler(self, **kwargs):
        """
        Turn the cliffordized object with haar bindings to a sampler object

        Args:
            kwargs which will be passed through to the transpiler
            should include
            backend: backend to run on
            initial_layout: layout on the device
        Returns:
            pub that can be sent to the sampler

        """

        # add measurements to the internal circuit
        circuit_internal = copy.deepcopy(self.circuit)
        circuit_internal.measure_all()

        if 'optimization_level' not in kwargs:
            kwargs['optimization_level'] = 1

        circuit_internal = transpile(circuit_internal, **kwargs)
        sampler_input = (circuit_internal, self.bindings)

        return sampler_input

    def _clifford_param_val(self, par_str, cliff_inds):
        vec = par_str.split("_")
        nl = int(vec[1])
        nq = int(vec[3])
        nz = int(vec[5])
        cliff_ind = cliff_inds[nl][nq]
        return self._cliff_params[cliff_ind][nz]

    @staticmethod
    def _haar_param_val(par_str, haar_angles):
        vec = par_str.split("_")
        nl = int(vec[1])
        nq = int(vec[3])
        nz = int(vec[5])
        return haar_angles[nl][nq][nz]

    @staticmethod
    def _cliff_angles(cliff):
        angles = Cliffordize.decomp.angles(cliff.to_matrix())
        return [angles[2], angles[0] + np.pi, angles[1] + np.pi]

    @staticmethod
    def _haar_angles_single():
        haar_op = random_unitary(2)
        angles = Cliffordize.decomp.angles(haar_op.to_matrix())
        return [angles[2], angles[0] + np.pi, angles[1] + np.pi]

    @staticmethod
    def _random_pauli(n, support=None):
        """
        Get a random Pauli

        Args:
            n: number of qubits
            support: bitstring of qubits with support '1' or not '0' (return I for not in support)
        Returns:
            random pauli string, e.g., 'XIYX'

        """
        out = ""
        if support is None:
            for i in range(n):
                out = np.random.choice(["I", "X", "Y", "Z"]) + out

        else:
            for s in support:
                if s:
                    out = np.random.choice(["X", "Y", "Z"]) + out
                else:
                    out = "I" + out

        if out == "I" * n:
            out = Cliffordize._random_pauli(n, support)
        return out


def readout_circuit(nq, chain, depth, output=None):
    """
    Make readout circuit that does a ramsey with cz gates
    before doing a readout test

    Args:
        nq: number of qubits for the circuit
        chain: chain of qubits
        depth: depth of circuit (must be even)
        output: target bitstring
    Returns:
        readout circuit
    """

    if np.mod(depth, 2) != 0:
        raise ValueError("Depth must be an even number")

    if output is not None:
        if len(output) != nq:
            raise ValueError("Output must be the size of nq")

    circuit = QuantumCircuit(nq, nq)

    # sx on all qubits in the chain
    for q in range(nq):
        circuit.sx(q)

    # do cz gates on all the 'even' gates of the chain
    circuit.barrier()
    for d in range(depth):
        for pair in chain:
            circuit.cz(pair[0], pair[1])
        circuit.barrier()

    # do sxd to return back to |0>
    for q in range(nq):
        circuit.rz(-np.pi,q)
        circuit.sx(q)
        circuit.rz(-np.pi,q)

    # if there is a target bitstring then do pi pulses
    # on the relevant qubits
    if output is not None:
        for q, b in enumerate(reversed(output)):
            if bool(int(b)):
                circuit.x(q)

    # measure all qubits
    for q in range(nq):
        circuit.measure(q, q)

    return circuit


def spam_circs(nq, layers, depths=[0, 2]):
    """
    Make a set of readout circuits for investigating spam

    Args:
        nq: number of qubits for the circuit
        layers: set of chains
        depth: list of depths
    Returns:
        spam circuit list
    """

    # two bitstrings are the all 0's and the all 1's
    all_bitstrings = ["0" * nq]
    all_bitstrings.append("1" * nq)

    all_circs = []
    for d in depths:
        for layer in layers:
            for bitstring in all_bitstrings:
                all_circs.append(readout_circuit(nq, layer, d, output=bitstring))

    return all_circs


def bricklayer_circ(nq, depth, gate="cz", singles="x", onlylayer=None):
    """
    Make bricklayer circuits on 0->(nq-1) circuits as a primitive to be used
    later

    Args:
        nq: number of qubits for the circuit
        depth: number of layers
        gate: 2Q gate
        singles: single qubit gate for the rotation block
        onlyLayer: index to either layer 0 or layer 1. Default None includes both
        layers in the circuit
    Returns:
        Circuit of 'depth' with TwoLocal entanglers on each edge

    """

    if gate == "cz":
        gate = CZGate
    elif gate == "ecr":
        gate = ECRGate
    else:
        raise ValueError("Gate %s not defined in bricklayer_circ" % gate)

    qubits = [a for a in range(nq)]
    layer_chains = [
        [(a, b) for a, b in zip(qubits[0::2], qubits[1::2])],
        [(a, b) for a, b in zip(qubits[1::2], qubits[2::2])],
    ]
    if onlylayer is not None:
        layer_chains = [layer_chains[onlylayer]]

    circ = TwoLocal(
        num_qubits=nq,
        rotation_blocks=singles,
        entanglement_blocks=gate(),
        entanglement=layer_chains,
        reps=depth,
        insert_barriers=True,
    ).decompose()

    return circ


def run_eplg_and_spam(
    grid_chain_flt,
    layers,
    backend,
    twoq_gate,
    oneq_gates,
    batch_run=False,
    lf_samples=6,
    lf_lengths=[1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400],
):
    """
    Run the EPLG grid circuits and SPAM

    Args:
        grid_chain_flt: list of chain of qubits in the grid
        layers: the 2Q gates to run
        backend: backend to run these one
        twoq_gate: two qubit gate
        oneq_gates: one qubit gate
        batch_run: Run as a batch (to do: not implemented )
        lf_samples: number of samples to use for the grid run
        lf_lengths: lengths to use for the lf runs

    Returns:
        lfexps: the list of lf experiments
        eplg_job_ids: job ids from lf experiments
        spam_job_id: job id from the spam
    """

    if batch_run:
        print("Batch run not implemented yet")

    lfexps = []

    for i in range(len(grid_chain_flt)):
        lfexps.append(
            LayerFidelity(
                physical_qubits=grid_chain_flt[i],
                two_qubit_layers=layers[2 * i : (2 * i + 2)],
                lengths=lf_lengths,
                backend=backend,
                num_samples=lf_samples,
                seed=60,
                two_qubit_gate=twoq_gate,
                one_qubit_basis_gates=oneq_gates,
            )
        )

        # set the synthesis method so that we match the brickwork circuits
        lfexps[-1].experiment_options.clifford_synthesis_method = "1Q_fixed"
        lfexps[-1].experiment_options.max_circuits = lf_samples * len(lf_lengths) * 2

    nshots = 200

    # Run the LF experiment (generate circuits and submit the job)
    exp_job_id_lst = []

    for i in range(len(grid_chain_flt)):
        exp_data = lfexps[i].run(shots=nshots)
        print(
            f"Run experiment: ID={exp_data.experiment_id} with jobs {exp_data.job_ids}]"
        )
        exp_job_id_lst.append(exp_data.job_ids)

    # use this sampler for the EPLG/spam runs
    spam_circ_lst = spam_circs(backend.num_qubits, layers)
    sampler = SamplerV2(backend)
    job_spam = sampler.run(spam_circ_lst, shots=2046)
    spam_id = job_spam.job_id()

    return lfexps, exp_job_id_lst, spam_id


def eplg_analysis(lfexps, eplg_ids, service, backend, twoq_gate, layers):
    """
    Analyze the eplg runs

    Args:
        lfexps: the lf experiment objs
        eplg_ids: the job ids
        service: the loaded qiskit runtime service containing the backend
        backend: backend we ran these one
        twoq_gate: two qubit gate
        layers: the grid layers

    Returns:
        updated error dictionary from the backend
    """

    # reload the job data using the IDs
    exp_data_lst = []
    for i in range(len(lfexps)):
        exp_data_lst.append(ExperimentData(experiment=lfexps[i]))
        exp_data_lst[-1].add_jobs([service.job(ii) for ii in eplg_ids[i]])
        lfexps[i].analysis.run(exp_data_lst[-1], replace_results=True)

    print("retrieving elpg data")
    # make an updated error map
    updated_err_dicts = []
    err_dict = lfu.make_error_dict(backend, twoq_gate)

    for i in range(len(lfexps)):
        # get the results from the experiment
        df = exp_data_lst[i].analysis_results(dataframe=True)
        for j in range(2):
            updated_err_dicts.append(lfu.df_to_error_dict(df, layers[2 * i + j]))

    err_dict = lfu.update_error_dict(err_dict, updated_err_dicts)
    return err_dict


def spam_analysis(nq, service, spam_id, layers):
    """
    Analyze the spam runs

    Args:
        nq: number of qubits
        service: the loaded qiskit runtime service containing the backend
        spam_id: job id for the spam job
        layers: the grid layers

    Returns:
        readout fidelities
    """

    depths = [0, 2]
    print("retrieving spam data")
    job = service.job(spam_id)
    res = job.result()
    errors = []
    raw = raw_bits(res)
    for q in range(nq):
        err = 0
        survivals0, survivals1 = binned_survivals(raw, depths, layers, q)
        for nl, layer in enumerate(layers):
            err += (
                (
                    2
                    - np.mean(np.array(survivals0[nl]), axis=1)
                    - np.mean(np.array(survivals1[nl]), axis=1)
                )
                / 2
                / len(layers)
            )
        errors.append(err)
    errors = np.array(errors)
    readout_fids = 1 - errors[:, 1]
    return readout_fids


def raw_bits(res):
    """
    Pull out bitstrings from sampler result

    Args:
        res: sampler result

    Returns:
        raw bitstring
    """

    raw_bitarrays = []
    # strip off annoying container stuff
    for r in res:
        test = r.data
        raw_bitarrays.append([test[d] for d in test][0])

    return np.array(raw_bitarrays)


def binned_survivals(raw_bitarrays, depths, layers, qubit):
    """
    Qubit by qubit binning of the readout results P01, P10
    by depth and layers

    Args:
        raw_bitarrays: the full bitstrings
        depths: number of cz's
        layers: layout of cz
        qubit: which qubit to analyze

    Returns:
        survival0/1 for qubit
    """

    bitstrings = ["0", "1"]
    survivals0 = [[[] for i, _ in enumerate(depths)] for j, _ in enumerate(layers)]
    survivals1 = [[[] for i, _ in enumerate(depths)] for j, _ in enumerate(layers)]

    ind = 0
    for indd, d in enumerate(depths):
        for indl, _ in enumerate(layers):
            for _, bit in enumerate(bitstrings):
                sliced = raw_bitarrays[ind].slice_bits([qubit])

                counts = sliced.get_counts()
                shots = np.sum([val for val in counts.values()])

                if bit == "0":
                    if "0" in counts.keys():
                        correct = counts["0"]
                    else:
                        correct = 0

                    survivals0[indl][indd].append(correct / shots)
                else:
                    if "1" in counts.keys():
                        correct = counts["1"]
                    else:
                        correct = 0

                    survivals1[indl][indd].append(correct / shots)
                ind += 1

    return survivals0, survivals1


def xeb(nq, counts, ideal_counts):
    """
    Calculate the cross entropy

    Args:
        nq: number of qubits
        counts: count dictionary
        ideal_counts: ideal counts dictionary

    Returns:
        xeb, xeb_ideal
    """

    p_0 = 0
    p_0_ideal = 0

    total_shots = 0

    for i in counts:
        p_0 += counts[i] * ideal_counts.get(i, 0)
        total_shots += counts[i]

    for i in ideal_counts:
        p_0_ideal += ideal_counts[i] ** 2

    return 2**nq * p_0 / total_shots - 1, 2**nq * p_0_ideal - 1
