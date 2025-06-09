# Simulating for fixed alphas, st np = alpha, and exp(-alpha) > 0, of course. Drop all eigenvalues that are near 0 (lambda < 0.0005)
# NO NEAR 0 REMOVAL
# alpha < 1

using LinearAlgebra
using Plots
using LaTeXStrings
include("support_functions.jl")

# Params
n = 10000
alpha = 0.5
p = alpha / n
figure_path = "sparse_alpha_lt1_figures/"

# Matrix setup
matrices = matrix_generator(n, p)
A_n = matrices[1]
delta_n = matrices[2]
L_n = matrices[3]

# A_n spectra
An_normed_spectra = calculate_An_spectra(A_n, n, p)
histogram(An_normed_spectra, legend=false, length=51)
title!(L"ESD of Normalized $\textbf{A}_n$, Sparse")
savefig(string(figure_path, "jl_normed_An.png"))

# Delta_n spectra
normed_centered_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p)
histogram(normed_centered_delta_n_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Centered $\Delta_n$, Sparse")
savefig(string(figure_path, "jl_normedcentered_deltan.png"))

normed_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p, false)
histogram(normed_delta_n_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Uncentered $\Delta_n$, Sparse")
savefig(string(figure_path, "jl_normed_deltan.png"))

# L_n spectra
normed_centered_L_spectra = calculate_Ln_spectra(L_n, n, p)
histogram(normed_centered_L_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Centered $\textbf{L}_n$, Sparse")
savefig(string(figure_path, "jl_normedcentered_Ln.png"))

normed_L_spectra = calculate_Ln_spectra(L_n, n, p, false)
histogram(normed_L_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Uncentered $\textbf{L}_n$, Sparse")
savefig(string(figure_path, "jl_normed_Ln.png"))

print("Plotted ESD of Ln\n")