# Simulating for fixed alphas, st np = alpha, and exp(-alpha) > 0, of course. Drop all eigenvalues that are near 0 (lambda < 0.0005)
# PRUNE NEAR 0 VALUES FOR ALL
# alpha < 1

using LinearAlgebra
using Plots
using LaTeXStrings
include("support_functions.jl")

# Params
n = 10000
alpha = 0.5
p = alpha / n
figure_path = "sparse_alpha_lt1_pruned_figures/"
pruning_theshold = 0.0005

# Matrix setup
matrices = matrix_generator(n, p)
A_n = matrices[1]
delta_n = matrices[2]
L_n = matrices[3]

# A_n spectra
An_normed_spectra = calculate_An_spectra(A_n, n, p)
thresholded_check_An_normed_spectra = findall(iszero, abs.(An_normed_spectra) .<= pruning_theshold)
An_pruned_normal_spectra = An_normed_spectra[thresholded_check_An_normed_spectra]
histogram(An_pruned_normal_spectra, legend=false, length=51)
title!(L"ESD of Normalized $\textbf{A}_n$, Sparse, $|\lambda_i(\textbf{A}_n)|\geq0.0005$")
savefig(string(figure_path, "jl_normed_An.png"))

# Delta_n spectra
normed_centered_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p)
thresholded_check_delta_n_normed_centered_spectra = findall(iszero, abs.(normed_centered_delta_n_spectra) .<= pruning_theshold)
normed_centered_pruned_delta_n_spectra = normed_centered_delta_n_spectra[thresholded_check_delta_n_normed_centered_spectra]
histogram(normed_centered_pruned_delta_n_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Centered $\Delta_n$, Sparse, $|\lambda_i(\Delta_n)|\geq0.0005$")
savefig(string(figure_path, "jl_normedcentered_deltan.png"))

normed_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p, false)
thresholded_check_delta_n_normed_spectra = findall(iszero, abs.(normed_delta_n_spectra) .<= pruning_theshold)
normed_pruned_delta_n_spectra = normed_delta_n_spectra[thresholded_check_delta_n_normed_spectra]
histogram(normed_pruned_delta_n_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Uncentered $\Delta_n$, Sparse, $|\lambda_i(\Delta_n)|\geq0.0005$")
savefig(string(figure_path, "jl_normed_deltan.png"))

# L_n spectra
normed_centered_L_spectra = calculate_Ln_spectra(L_n, n, p)
thresholded_check_L_normed_centered_spectra = findall(iszero, abs.(normed_centered_L_spectra) .<= pruning_theshold)
normed_centered_pruned_L_spectra = normed_centered_L_spectra[thresholded_check_L_normed_centered_spectra]
histogram(normed_centered_pruned_L_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Centered $\textbf{L}_n$, Sparse, $|\lambda_i(\textbf{L}_n)|\geq0.0005$")
savefig(string(figure_path, "jl_normedcentered_Ln.png"))

normed_L_spectra = calculate_Ln_spectra(L_n, n, p, false)
thresholded_check_L_normed_spectra = findall(iszero, abs.(normed_L_spectra) .<= pruning_theshold)
normed_pruned_L_spectra = normed_L_spectra[thresholded_check_L_normed_spectra]
histogram(normed_pruned_L_spectra, legend=false, length=51)
title!(L"ESD of Normalized/Uncentered $\textbf{L}_n$, Sparse, $|\lambda_i(\textbf{L}_n)|\geq0.0005$")
savefig(string(figure_path, "jl_normed_Ln.png"))

print("Plotted ESD of Ln\n")