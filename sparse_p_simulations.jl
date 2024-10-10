using Distributions 
using LinearAlgebra
using Plots

# Params
n = 25000
p = 1.11 * 10 ^ -4

# Matrix setup
A_n_raw = rand(Distributions.Bernoulli(p), n, n)
A_n = Symmetric(A_n_raw)
A_n[diagind(A_n)] .= 0
A_n = Float32.(A_n)

print("Generated A_n\n")

D_n_vals = sum(A_n, dims = 2)
D_n = zeros(n, n)
D_n[diagind(D_n)] = D_n_vals

print("Calculated D_n\n")

delta_n = D_n - A_n

print("Calculated delta_n\n")

inv_sqrt_D_n = zeros(n, n)
inv_sqrt_D_n_vals = 1 ./ sqrt.(D_n_vals)
inv_sqrt_D_n[diagind(inv_sqrt_D_n)] = inv_sqrt_D_n_vals

L = I - inv_sqrt_D_n * A_n * inv_sqrt_D_n

print("Calculated L_n\n")

# A_n spectra
centered_An = A_n .- p
normed_centered_An = centered_An .* 1/sqrt(n * p * (1 - p))
An_normed_spectra = eigen(normed_centered_An).values
histogram(An_normed_spectra)
title!("ESD of Normalized A_n")
savefig("jl_normed_An.png")

print("Plotted ESD of An\n")

# Delta_n spectra
centering_delta_matrix = -p .* ones(n, n)
centering_delta_matrix[diagind(centering_delta_matrix)] .= (n - 2) * p
centered_delta_n = (delta_n - centering_delta_matrix) ./ sqrt(n * p)
normed_centered_delta_n_spectra = eigen(centered_delta_n).values

histogram(normed_centered_delta_n_spectra)
title!("ESD of Normalized/Centered delta_n")
savefig("jl_normedcentered_deltan.png")

normed_delta_n = delta_n ./ sqrt(n * p)
normed_delta_n_spectra = eigen(normed_delta_n).values
histogram(normed_delta_n_spectra)
title!("ESD of Normalized/Uncentered delta_n")
savefig("jl_normed_deltan.png")

print("Plotted ESD of deltan\n")

# L_n spectra
centering_L_matrix = (0 - 1/(n-1)) .* ones(n, n)
centering_L_matrix[diagind(centering_L_matrix)] .= 1
centered_L_matrix= (L - centering_L_matrix) .* sqrt((n * p)/(1 - p))
normed_centered_L_spectra = eigen(centered_L_matrix).values

histogram(normed_centered_L_spectra)
title!("ESD of Normalized/Centered L_n")
savefig("jl_normedcentered_Ln.png")

normed_L_matrix = L .* sqrt((n * p)/(1 - p))
normed_L_spectra = eigen(normed_L_matrix).values

histogram(normed_L_spectra)
title!("ESD of Normalized/Uncentered L_n")
savefig("jl_normed_Ln.png")

print("Plotted ESD of Ln\n")
