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
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import (TwoLocal, CZGate, ECRGate,
                                    RZGate, SXGate) 
from qiskit.quantum_info import (Clifford, Pauli, SparsePauliOp, 
                                 random_unitary)
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis import OneQubitEulerDecomposer

"""Utilities for running large Clifford and XEB circuits"""

class Single_Decomp_pass(TransformationPass):

    """
    Transformation pass that converts any 
    single qubit gate in the circuit into parameterized U3 form
    """

    def __init__(self):
        super().__init__()
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        
        single_layer=True
        idl=0
        idq=0
        for node in dag.op_nodes(include_directives=False):
            if node.op.num_qubits > 1:
                if single_layer:
                    single_layer=False
                    idl+=1
                    idq = 0    
                continue
            if not single_layer:
                single_layer=True
            gate_blocks = Single_Decomp_pass.rz_sx_gate_blocks(idl,idq)
            dag.substitute_node_with_dag(
                node=node,
                wires=node.qargs,
                input_dag=Single_Decomp_pass.par_clifford_dag(
                    gate_blocks=gate_blocks,
                    qubits=node.qargs,
                ),
            )
            idq+=1

        return dag
    
    @staticmethod
    def rz_sx_gate_blocks(idl,idq):
        def pars(zi):
            return Parameter(f"l_{idl}_q_{idq}_z_{zi}")
        return [RZGate(pars(0)), SXGate(), RZGate(pars(1)), SXGate(), RZGate(pars(2))]            

    @staticmethod        
    def par_clifford_dag(
        gate_blocks,
        qubits = None,
        qreg = None,
        num_qubits = None,
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

   
    decomp = OneQubitEulerDecomposer(basis='ZSX')
    pauli_preps = [Clifford([[0,1,0],[1,0,0]]),Clifford([[1,0,1],[1,1,0]]),Clifford([[1,0,0],[0,1,0]])]


    def __init__(self, circuit, nsamples=1):

        """
        Init class

        Args:
            circuit: base template circuit
        Returns:
        """

        single_cliffs = []
        stab_base = [[1,0],[0,1],[1,1]]
        ph_base=[[0,0],[1,0],[0,1],[1,1]]
        for iz in range(3):
            for ix in range(2):
                for iph in range(4):
                    cliff_table=np.zeros([2,3],dtype=int)
                    cliff_table[0,0:2] = stab_base[iz]
                    cliff_table[1,0:2] = stab_base[(iz+ix+1)%3]
                    cliff_table[:,2] = ph_base[iph]
                    single_cliffs.append(Clifford(cliff_table))

        self._single_cliffs = single_cliffs
        self._cliff_params = [Cliffordize._cliff_angles(c) for c in single_cliffs]

        self.nq = len(circuit.qregs[0])
        self.nsamples = nsamples
        self.base_circuit=circuit
       
        #initialize but will be populated by construct
        self.nlayers = None
        self.bindings = None
        self.observables = None
        self.construct()
        
    def get_circuit(self):
        return self.circuit
    
    def circ_to_layers(self,circ):
        return circ
    
    def random_cliff(self, input_support=None, output_support=None, rand_outputs_only=False):

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
            raise ValueError('Input support and output support cannot be simultaneously specified')

        if rand_outputs_only:

            #only 1 cliffordization, so repeat nsamples times

            cliff_inds, _  = self._random_cliff_bindings(nrands=1)
            cliff_inds = [cliff_inds[0] for i in range(self.nsamples)]

        else:
            cliff_inds, _  = self._random_cliff_bindings()

        #Construct the input and output pauli's for direct fidelity estimation
        
        if output_support is None:

            #if the output support is not specified, work from random input pauli's and evolve to the output

            input_Paulis = [Cliffordize._random_pauli(self.nq, input_support) for i in range(self.nsamples)]
            output_Paulis=[]
            for samp_idx in range(self.nsamples):
       
                if self.nsamples==1 or rand_outputs_only:
                    idx = ()
                else:
                    idx = (samp_idx,)
                    
                pind = samp_idx

                cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))
                pauli = Pauli(input_Paulis[pind])
                
                output_P = pauli.evolve(cliff,frame='s')
                pstring=''
                for x, z in zip(pauli.evolve(cliff,frame='s').x,pauli.evolve(cliff,frame='s').z):
                    if x and z:
                        ptemp = 'Y'
                    elif x:
                        ptemp = 'X'
                    elif z:
                        ptemp= 'Z'
                    else:
                        ptemp = 'I'
                    pstring  = ptemp+pstring    
                
                
                if output_P.phase==2:
                    pstring='-'+pstring
                output_Paulis.append(pstring)
                
        else:

            #output support is specified, so specify the output pauli's and work back

            output_Paulis = [Cliffordize._random_pauli(self.nq, output_support) for i in range(self.nsamples)]
            input_Paulis=[]
            for samp_idx in range(self.nsamples):

                if self.nsamples==1 or rand_outputs_only:
                    idx = ()
                else:
                    idx = (samp_idx,)
                    
                pind = samp_idx
                cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))

                pauli = Pauli(output_Paulis[pind])
                input_Paulis.append(str(pauli.evolve(cliff.adjoint(),frame='s')))
         
        #Input and output pauli's are now determined
        #Mix the input and output pauli's with the first and last clifford layers
        #update input bindings
        self.input_support_all=[]
        for s in range(self.nsamples):
            
            pauli = input_Paulis[s]
            if input_Paulis[s][0]=='-':
                pauli=input_Paulis[s][1:]
                output_Paulis[s]='-'+output_Paulis[s]
            else: 
                pauli=input_Paulis[s]
            for q,p in enumerate(reversed(pauli)):
                if p=='I' or p=='Z' :
                    continue
                elif p=='X':
                    precliff = Cliffordize.pauli_preps[0]
                elif p=='Y':
                    precliff = Cliffordize.pauli_preps[1]
                postcliff = self._single_cliffs[self.cliff_inds[s][0][q]]
                newcliff = precliff.compose(postcliff)
                cliff_inds[s][0][q] = self._single_cliffs.index(newcliff)
            input_supp=[]
            for p in reversed(pauli):
                if p=='I':
                    input_supp.append(0)
                else:
                    input_supp.append(1)
            self.input_support_all.append(input_supp)

        #update output bindings and construct observables
        for s in range(self.nsamples):
            
            for q,p in enumerate(reversed(output_Paulis[s])):
                if p=='I' or p=='Z' or p =='-' :
                    continue
                elif p=='X':
                    postcliff = Cliffordize.pauli_preps[0]
                elif p=='Y':
                    postcliff = Cliffordize.pauli_preps[1]
                precliff = self._single_cliffs[self.cliff_inds[s][-1][q]]
                newcliff = precliff.compose(postcliff)
                cliff_inds[s][-1][q] = self._single_cliffs.index(newcliff)
            out=''
            for p in output_Paulis[s]:
                if p=='X' or p=='Y' or p=='Z':
                    out+='Z'
                else:
                    out+=p
            output_Paulis[s] = out

        cliff_inds,_ = self._random_cliff_bindings(cliff_inds=cliff_inds)

        #construct observables
        self.observables = np.empty(self.nsamples, dtype=SparsePauliOp)
        for samp in range(self.nsamples):
            pauli_str = output_Paulis[samp]
            
            self.observables[samp] = SparsePauliOp(pauli_str)


        return cliff_inds, self.bindings, self.observables

    
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
      
        #sample random singles
        #build a list of random cliffords
        if cliff_inds is None:
            cliff_inds = [[[np.random.randint(24) for q in range(self.nq)]
                                for l in range(self.nlayers)]
                            for s in range(nrands)] 
            
        #convert the list of random cliffords into a bindings array for the circuit
        bindings = BindingsArray(data={par: np.array([self._clifford_param_val(str(par),ci) for ci in cliff_inds])
            for par in self.circuit.parameters
        })

        self.bindings = bindings

        return cliff_inds, bindings
    
        
    def random_haar(self):
        """
        Output a set of random binding arrays to make this a haar random circuit
        
        Args:
        Returns:
            haar_angles: list of haar angles 
            bindings: list of proper bindings arrays for the circuit
        """
        
        #sample random singles
        haar_angles = [[[Cliffordize._haar_angles_single() for q in range(self.nq)]
                            for l in range(self.nlayers)]
                        for s in range(self.nsamples)] 

        bindings = BindingsArray(data={par: np.array([Cliffordize._haar_param_val(str(par),ha) for ha in haar_angles])
            for par in self.circuit.parameters
        })

        self.bindings = bindings

        return haar_angles, bindings
    
    
    def construct(self): 

        """
        Convert the circuit into U3 parameteried form using the 
        Single_Decomp_pass class

        Caveats:
        For now the circuit coming in must be in a layered form with a layer of singles
        at the beginning and end of the circuit. To do is add checking that this is the case
        """

        #layerize, TODO, make this
        self.circuit = self.circ_to_layers(self.base_circuit)
        
        pm = PassManager([Single_Decomp_pass()])
        self.circuit = pm.run(self.circuit)

        #number of layers in the circuit
        self.nlayers = np.max([int(str(p).split('_')[1])+1 for p in self.circuit.parameters])


    def _clifford_param_val(self, par_str, cliff_inds):
        vec = par_str.split('_')
        nl=int(vec[1])
        nq=int(vec[3])
        nz=int(vec[5])
        cliff_ind = cliff_inds[nl][nq]
        return self._cliff_params[cliff_ind][nz]
    
    @staticmethod
    def _haar_param_val(par_str, haar_angles):
        vec = par_str.split('_')
        nl=int(vec[1])
        nq=int(vec[3])
        nz=int(vec[5])
        return haar_angles[nl][nq][nz]  
    
    @staticmethod            
    def _cliff_angles(cliff):
        angles = Cliffordize.decomp.angles(cliff.to_matrix())
        return [angles[2] ,angles[0]+np.pi ,angles[1]+np.pi ]
    
    @staticmethod
    def _haar_angles_single():
        haar_op = random_unitary(2)
        angles = Cliffordize.decomp.angles(haar_op.to_matrix())
        return [angles[2] ,angles[0]+np.pi ,angles[1]+np.pi ]
    
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
        out=''
        if support is None:
            for i in range(n):
                out=np.random.choice(['I','X','Y','Z'])+out
            
        else:
            for s in support:
                if s:
                    out=np.random.choice(['X','Y','Z'])+out
                else:
                    out='I'+out
                    
        if out == 'I'*n:
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

    if np.mod(depth,2)!=0:
        raise ValueError('Depth must be an even number')
    
    if output is not None:
        if len(output)!=nq:
            raise ValueError('Output must be the size of nq')

    circuit = QuantumCircuit(nq,nq)
    
    #sx on all qubits in the chain
    for q in range(nq):
        circuit.sx(q)
    
    #do cz gates on all the 'even' gates of the chain
    circuit.barrier()
    for d in range(depth):
        for pair in chain:
            circuit.cz(pair[0],pair[1])
        circuit.barrier()
    
    #do sxd to return back to |0>
    for q in range(nq):
        circuit.sxdg(q)   
        
    #if there is a target bitstring then do pi pulses 
    #on the relevant qubits
    if output is not None:
        for q,b in enumerate(reversed(output)):
            if bool(int(b)):
                circuit.x(q)
    
    #measure all qubits
    for q in range(nq):
        circuit.measure(q,q)
    
        
    return circuit

def spam_circs(nq, layers, depths=[0,2]):

    """
    Make a set of readout circuits for investigating spam

    Args:
        nq: number of qubits for the circuit
        layers: set of chains
        depth: list of depths
    Returns:
        pam circuit list
    """

    #two bitstrings are the all 0's and the all 1's
    all_bitstrings = ['0'*nq]
    all_bitstrings.append(['1'*nq])

    all_circs = []
    for d in depths:
        for layer in layers:
            for bitstring in all_bitstrings:
                all_circs.append(readout_circuit(nq,layer,d,output=bitstring))
   
    return all_circs


def bricklayer_circ(nq, depth, gate = 'cz', singles='x', onlylayer=None):

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

    if gate == 'cz':
        gate = CZGate
    elif gate == 'ecr':
        gate = ECRGate
    else:
        raise ValueError('Gate %s not defined in bricklayer_circ'%gate)
        
    qubits = [a for a in range(nq)]
    layer_chains = [[(a,b) for a, b in zip(qubits[0::2],qubits[1::2])],
                    [(a,b) for a, b in zip(qubits[1::2],qubits[2::2])]]
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



