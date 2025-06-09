# Simulation for p to 0, np to infinity, np > logn

# Run for the sparse regimes
# Run for sparse

using LinearAlgebra
using Plots
using LaTeXStrings
include("support_functions.jl")

n = 10000
p = 0.3
figure_path = "dense_fix_p_figures/"

# Matrix setup
matrices = matrix_generator(n, p)
A_n = matrices[1]
delta_n = matrices[2]
L_n = matrices[3]

# A_n
An_normed_spectra = calculate_An_spectra(A_n, n, p)
histogram(An_normed_spectra, legend=false, length=51)
title!(L"ESD of Normalized $\textbf{A}_n$")
savefig(string(figure_path, "jl_normed_An.png"))

# Delta_n spectra
normed_centered_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p)
histogram(normed_centered_delta_n_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Centered $\Delta_n$")
savefig(string(figure_path, "jl_normedcentered_deltan.png"))

normed_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p, false)
histogram(normed_delta_n_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Uncentered $\Delta_n$")
savefig(string(figure_path, "jl_normed_deltan.png"))

# L_n spectra
normed_centered_L_spectra = calculate_Ln_spectra(L_n, n, p)
histogram(normed_centered_L_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Centered $\textbf{L}_n$")
savefig(string(figure_path, "jl_normedcentered_Ln.png"))

normed_L_spectra = calculate_Ln_spectra(L_n, n, p, false)
histogram(normed_L_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Uncentered $\textbf{L}_n$")
savefig(string(figure_path, "jl_normed_Ln.png"))

print("Plotted ESD of Ln\n")
