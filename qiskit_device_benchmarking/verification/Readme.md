# Fast Benchmarking

The file fast_bench.py is a command line invocation to run a mirror qv suite on device(s) `python fast_bench.py`. Requires a config file (default is `config.yaml`) of the form
```
hgp: X/X/X
backends: [ibm_sherbrooke, ibm_brisbane, ibm_torino]
nrand: 10
depths: [4,6,8,10,12]
he: True
dd: True
opt_level: 1
trials: 10
shots: 200
```
Generates an output yaml (timestamped) with the results. There are two types of circuits for the benchmarking. 
Mirror QV circuits which are all-to-all and HE (hardware-efficient) Mirror QV circuits which are layers of random SU(4) assuming nearest neighbor on a chain.

The output can be turned into plots with `bench_analyze.py`, e.g. `python bench_analyze.py -f MQV_2024-04-27_06_19_32.yaml -v max --plot` will product a plot of all the maximum results over the sets from the listed file. Plots are generated as pdf.

Similarly, the file `fast_layer_fidelity.py` is a command line invocation to run a single layer fidelity experiment on specified device(s) `python fast_layer_fidelity.py`. The qubit chain selected for this is the reported 100Q on qiskit for each device(s). This file also requires a config file (default is `config.yaml`) of the form (unless overwritten by command line arguments)
```
hgp: X/X/X
backends: [ibm_fez]
channel: 'ibm_quantum' or 'imb_cloud'
```

Circuits are based on the code written for https://arxiv.org/abs/2303.02108 which was based on the earlier work by Proctor et al [Phys. Rev. Lett. 129, 150502 (2022)](https://doi.org/10.48550/arXiv.2112.09853). 
