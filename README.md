## Masters in Statistics and Data Science Thesis
### Normalized Laplacian for Random Graphs: Spectral properties and Applications

Repository for simulation work for normalized Laplacians for random graphs and degree based eigenvalue correction for Graphical Neural Networks.

Simulations are for the following cases, in both Python and Julia:
1. Dense regime with fixed $p, np\to\infty$
2. Dense regime with $p\to 0, np\to\infty$
3. Sparse regime with $p\sim \frac{\lambda}{n}, np\to\lambda$
4. Sparse regime with $p=\frac{\lambda}{n}$, $\lambda$ between 1 and 100, s.t. $np<<\log{n}$
5. Block matrix with fixed $p$ within each block

This includes both unscaled and uncentered spectra as well as the scaled and centered spectra. Kernel density estimates are done using [`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity). 

GNN modelling can be found in the GNN_Corrections directory. Note that this was initially run on an AMD machine, and so `sdnext` is currently a submodule. There is some work involved to connect `sdnext` on AMD to `torch_geometric` and other `torch` adjacent libraries, and this is quite an involved process that is heavily machine dependent. `training_fix.py` is the main run file to generate results for the three tested correction methodologies (no corretion, uniform correction, degree-based correction). If the code is to be run on either a CPU or on an Nvidia card, some alterations will need to be done to drop `local_amd_setup` from the relevant files.
