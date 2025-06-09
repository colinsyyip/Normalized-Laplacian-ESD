# Universal simulation script which takes p down over a series of steps
# to avoid many scripts doing almost the same thing

using LinearAlgebra
using Plots
using LaTeXStrings
include("support_functions.jl")

n = 10000
n_blocks = 2
m = fill(Int.(n/n_blocks), n_blocks)
p_vec = reverse!(collect(range(5 * 10 ^ -5, stop=0.5, length=10)))
# p_vec = [0.5]
q_vec = p_vec ./ 2
figure_path = "inhomogeneous_esd_figures/"
file_counter = 0

for (p, q) in zip(p_vec, q_vec)

    global file_counter

    print("Simulating ESD at p=$p\n")

    # Matrix setup
    matrices = block_matrix_generator(n, m, fill(p, n_blocks), q)
    A_n = matrices[1]
    delta_n = matrices[2]
    L_n = matrices[3]

    # A_n
    An_normed_spectra = calculate_An_spectra(A_n, n, p)
    histogram(An_normed_spectra, legend=false, length=51)
    title!(L"$\mu(\textbf{A}_n)$, $n=$%$(n), $p=$%$(p), Number of Blocks: %$(n_blocks)")
    savefig(string(figure_path, "normed_An_$file_counter.png"))

    # Delta_n spectra
    normed_centered_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p)
    histogram(normed_centered_delta_n_spectra, legend=false, length=51)
    title!(L"$\mu(\Delta_n)$, $n=$%$(n), $p=$%$(p), Number of Blocks: %$(n_blocks)")
    savefig(string(figure_path, "jl_normedcentered_deltan_$file_counter.png"))

    normed_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p, false)
    histogram(normed_delta_n_spectra, legend=false, length=51)
    title!(L"$\mu(\Delta_n)$, $n=$%$(n), $p=$%$(p), Number of Blocks: %$(n_blocks)")
    savefig(string(figure_path, "jl_normed_deltan_$file_counter.png"))

    # L_n spectra
    normed_centered_L_spectra = calculate_Ln_spectra(L_n, n, p)
    histogram(normed_centered_L_spectra, legend=false, length=51)
    title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p), Number of Blocks: %$(n_blocks)")
    savefig(string(figure_path, "jl_normedcentered_Ln_$file_counter.png"))

    normed_L_spectra = calculate_Ln_spectra(L_n, n, p, false)
    histogram(normed_L_spectra, legend=false, length=51)
    title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p), Number of Blocks: %$(n_blocks)")
    savefig(string(figure_path, "jl_normed_Ln_$file_counter.png"))

    file_counter += 1

end