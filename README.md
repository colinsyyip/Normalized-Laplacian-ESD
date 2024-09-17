## Masters in Statistics and Data Science Thesis
### Normalized Laplacian for Random Graphs: Spectral properties and Applications

Repository for simulation work for normalized Laplacians for random graphs.

Simulations for the following cases:
1. Dense regime with fixed $p, np\to\infty$
2. Dense regime with $p\to 0, np\to\infty$
3. Sparse regime with $p\sim \frac{\lambda}{n}, np\to\lambda$
4. Sparse regime with $p=\frac{\lambda}{n}$, $\lambda$ between 1 and 100, s.t. $np<<\log{n}$
5. Block matrix with fixed $p$ within each block

This includes both unscaled and uncentered spectra as well as the scaled and centered spectra. Kernel density estimates are done using [`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity). 