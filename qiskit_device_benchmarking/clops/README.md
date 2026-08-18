# CLOPS Benchmark

This benchmark measures Circuit Layer Operations Per Seconds (CLOPS) of
parameterized utility scale  hardware efficient circuits. 
 CLOPS measures the steady state throughput of a large quantity of 
 these parameterized circuits that are 
 of width 100 qubits with 100 layers of gates. 
 Each layer consists of two qubit gates across as many qubits
 as possible that can be done in parallel, followed by a single qubit
 gate(s) on every qubit to allow any arbitrary rotation, with those 
 rotations being parameterized.
 Parameters are applied to the circuit to generate a large number of 
 instantiated circuits to be executed on the quantum computer. It is
 up to the vendor on how to optimally execute these circuits for 
 maximal throughput. 

CLOPS supports the `execution_path` option, which passes a value through to the
Sampler's experimental `execution_path` option. Use this when a device offers a
special execution path worth benchmarking; which paths exist changes over time.

## Qubit set and layer structure

CLOPS is measured on a connected 1D chain of qubits, selected in this order:

1. An explicit chain passed as `qubits=[...]`.
2. The layer fidelity chain reported by the backend (`lf_<width>`), i.e. the
   best chain found during EPLG characterization. This is the default whenever
   the backend reports one, so CLOPS is measured on the same qubits as the
   companion layer fidelity measurement and the two numbers can be interpreted
   together.
3. A topology only heuristic (longest connected path), used when the backend
   reports no such chain. This must be declared when reporting a result;
   `job_attributes['qubit_set_source']` records which of the three was used.

The circuit is built from the canonical decomposition of that chain into two
qubit disjoint sublayers, L1 on the even bonds and L2 on the odd bonds, cycled
L1, L2, L1, L2, ... over the requested depth. This is the same decomposition the
layer fidelity experiment uses. Every layer has a fixed structure; only the
single qubit gate parameters vary between depth positions. Note a 100 qubit
chain has 99 bonds, so L1 has 50 gates covering every qubit, while L2 has 49 and
leaves both chain endpoints idle.

An unqualified CLOPS number means the canonical operating point of 100 qubits,
100 layers and 100 shots. Any other operating point is labelled
`CLOPS_h(N, D, S)`; see `job_attributes['clops_label']`.

CLOPS does not verify layer fidelity operating conditions internally, so pass
`lfoc_declared=True` (or `False`) to record whether the run satisfied them.

In the `twirled` method the circuit sent to the Sampler is unparameterized, so
its single qubit layers are one fixed SX per qubit; the Sampler's gate twirling
supplies the parameterization, inserting an rz-sx-rz-sx-rz frame ahead of the 2Q
gates in each twirling box. That frame is the fully twirled single qubit layer the
protocol calls for, so the fixed SX layer is additional to it. It is kept on
purpose: gate twirling only covers qubits taking part in a box, so a qubit idling
in a sublayer would otherwise receive no single qubit gate at all. On a 100 qubit
chain L1 covers every qubit, while L2 leaves both chain endpoints idle. The extra
gates cost a small amount of speed and ensure the executed circuit is never
thinner than the protocol specifies.

The `layer_order` argument controls where the barrier sits within each layer.
The default `"2q_first"` keeps the barrier between the 2Q and 1Q gates, which is
the twirling box the Sampler's gate twirling expects. `"1q_first"` instead orders
each layer as 1Q gates, 2Q gates, barrier, terminating each physical layer with
the barrier. Both build the same gates and the same number of layers.

## Example

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_device_benchmarking.clops.clops_benchmark import clops_benchmark

service = QiskitRuntimeService()

# Run clops with default settings (twirled circuits, 5000 circuits in run,
# 100 wide by 100 layers, etc)  Note this is done in a session and currently
# takes about 10 minutes to run
my_clops_run = clops_benchmark(service, "your-favorite-ibm-quantum-computer")

# If a device offers a special execution path, pass it through with
# execution_path. On a faster path you may need to increase the number of
# circuits to reach steady state
my_clops_run = clops_benchmark(service, "ibm_pittsburgh", execution_path='some-path', num_circuits = 5000)

# We can check the attributes of the benchmark run
print(my_clops_run.job_attributes)
{'backend_name': 'ibm_pittsburgh', 'width': 100, 'layers': 100, 'shots': 100, 'rep_delay': 0.00025, 'num_circuits': 5000, 'circuit_type': 'twirled', 'batch_size': None, 'pipelines': 1, 'qubits': [116, 101, 102, 103, 104, ..., 153, 154, 155], 'qubit_set_source': 'eplg_reported', 'externally_specified': True, 'two_qubit_layers': [[[116, 101], [102, 103], [104, 105], ..., [154, 155]], [[101, 102], [103, 104], [105, 106], ..., [153, 154]]], 'num_sublayers': 2, 'two_qubit_gate': 'cz', 'layer_order': '2q_first', 'twirling': {'enable_gates': True, 'num_randomizations': 5000, 'shots_per_randomization': 100}, 'parameterization': 'sampler_gate_twirling', 'execution_interface': 'qiskit-ibm-runtime SamplerV2', 'execution_path': None, 'canonical_operating_point': True, 'clops_label': 'CLOPS', 'lfoc_declared': None}

# The report method returns just the fields a CLOPS result must specify.
# Note this blocks until the result is ready, since it calls clops()
print(my_clops_run.report())

# There is a standard qiskit job and we can check its status, job_id, etc
print(my_clops_run.job.status())
QUEUED

# The clops method will calculate the clops value for the run
# Note this call will block until the result is ready
print("Measured clops of", my_clops_run.job_attributes['backend_name'], "is", my_clops_run.clops())
Measured clops of ibm_pittsburgh is 332154
```



## Variations


The benchmark code provides several 
 ways to measure CLOPS depending on the capability of the quantum computer. 

 The "twirling" method uses the native parameterization of the Sampler
 primitive to parameterize the circuit, and optimal batching of the 
 circuits is assumed to be done by the Sampler, freeing the user
 from having to optimize the batch size. The only requirement is
 that the total number of circuits executed needs to be chosen to
 get the system into a steady state to measure CLOPS.

 The "parameterized" method is similar, but instead sends an already
 parameterized circuit to the Sampler primitive, along with enough 
 parameters to execute the specified number of circuits. Batching 
 again is handled by the Sampler. This method requires larger bandwidth
 to send in all of the necessary parameters. Currently on IBM systems
 you will need to limit the number of circuits to approximately 160 to 
 fit within API job input limits.

 The "instantiated" method (not yet implemented) is for systems that cannot natively 
 handle parameterized circuits. In this case the circuit parameters
 are bound locally and then sent to the quantum computer for execution.
 This method requires the user to specify the desired size of each
 batch of circuits (so that they can be sent together the quantum computer) 
 as well as the number of local parallel pipelines to bind parameters and
 create payloads in parallel. The user will need to tune both of these
 parameters to try and optimize performance of the system. This will 
 tend to be much slower than on systems that natively support parameterized
 circuits
