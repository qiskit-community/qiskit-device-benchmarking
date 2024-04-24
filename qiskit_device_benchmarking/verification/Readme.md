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

Circuits are based on the code written for https://arxiv.org/abs/2303.02108 which was based on the earlier work by Proctor et al [Phys. Rev. Lett. 129, 150502 (2022)](https://doi.org/10.48550/arXiv.2112.09853). 
