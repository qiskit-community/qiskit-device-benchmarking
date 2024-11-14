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

CLOPS now supports the new `gen3-turbo` flag for execution path available
on some of our devices.

## Example

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_device_benchmarking.clops.clops_benchmark import clops_benchmark

service = QiskitRuntimeService(channel="ibm_quantum",
                               instance="your-hub/group/project")

# Run clops with default settings (twirled circuits, 1000 circuits in run,
# 100 wide by 100 layers, etc)  Note this is done in a session and currently
# takes about 10 minutes to run
my_clops_run = clops_benchmark(service, "your-favorite-ibm-quantum-computer")

# To run clops with the new new `gen3-turbo` path, you can specify the 
# execution path. For the new faster path you should increase the number
# of circuits to 5,000
my_clops_run = clops_benchmark(service, "machine supporting gen3-turbo", execution_path='gen3-turbo', num_circuits = 5000)

# We can check the attributes of the benchmark run
print(my_clops_run.job_attributes)
{'backend_name': 'ibm_brisbane', 'width': 100, 'layers': 100, 'shots': 100, 'rep_delay': 0.00025, 'num_circuits': 1000, 'circuit_type': 'twirled', 'batch_size': None, 'pipelines': 1}

# There is a standard qiskit job and we can check its status, job_id, etc
print(my_clops_run.job.status())
QUEUED

# The clops method will calculate the clops value for the run
# Note this call will block until the result is ready
print("Measured clops of", my_clops_run.job_attributes['backend_name'], "is", my_clops_run.clops())
Measured clops of ibm_brisbane is 30256
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
