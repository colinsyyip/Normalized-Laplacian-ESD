# Universal simulation script which takes p down over a series of steps
# to avoid many scripts doing almost the same thing

using LinearAlgebra
using Plots
using LaTeXStrings
include("support_functions.jl")

n = 10000
# p_vec = reverse!(collect(range(5 * 10 ^ -5, stop=0.5, length=10)))
p_vec = [5 * 10 ^ -5]
figure_path = "homogeneous_esd_figures/"
file_counter = 0

### TO DO - prune out some of the extraneous plots to see if theyre contributing to LAPACK errors

for p in p_vec

    if n * p > 1000
        
        continue
    
    end

    global file_counter

    print("Simulating ESD at p=$p\n")

    # Matrix setup
    matrices = matrix_generator(n, p)
    A_n = matrices[1]
    delta_n = matrices[2]
    L_n = matrices[3]

    print("Matrix sim. done")

    # A_n
    # An_normed_spectra = calculate_An_spectra(A_n, n, p)
    # histogram(An_normed_spectra, legend=false, length=51)
    # title!(L"ESD of Normalized $\textbf{A}_n$, $n=$%$(n), $p=$%$(p)")
    # savefig(string(figure_path, "normed_An_$file_counter.png"))

    # if n * p <= log(n)
 
    #     histogram(An_normed_spectra, legend=false, length=51, ylim=(0, 1000))
    #     title!(L"ESD of Normalized $\textbf{A}_n$, $n=$%$(n), $p=$%$(p)")
    #     savefig(string(figure_path, "normed_An_truncated_$file_counter.png"))
        
    #     histogram(An_normed_spectra, legend=false, length=51, ylim=(0, 50))
    #     title!(L"ESD of Normalized $\textbf{A}_n$, $n=$%$(n), $p=$%$(p)")
    #     savefig(string(figure_path, "normed_An_truncated_small_$file_counter.png"))
    
    # end

    # Delta_n spectra
    # normed_centered_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p)
    # histogram(normed_centered_delta_n_spectra, legend=false, length=51)
    # title!(L"ESD of Normalized/Centered $\Delta_n$, $n=$%$(n), $p=$%$(p)")
    # savefig(string(figure_path, "jl_normedcentered_deltan_$file_counter.png"))

    # if n * p <= log(n)

    #     histogram(normed_centered_delta_n_spectra, legend=false, length=51, ylim=(0, 1000))
    #     title!(L"ESD of Normalized/Centered $\Delta_n$, $n=$%$(n), $p=$%$(p)")
    #     savefig(string(figure_path, "jl_normedcentered_deltan_truncated_$file_counter.png"))

    #     histogram(normed_centered_delta_n_spectra, legend=false, length=51, ylim=(0, 50))
    #     title!(L"ESD of Normalized/Centered $\Delta_n$, $n=$%$(n), $p=$%$(p)")
    #     savefig(string(figure_path, "jl_normedcentered_deltan_truncated_small_$file_counter.png"))

    # end

    # normed_delta_n_spectra = calculate_Deltan_spectra(delta_n, n, p, false)
    # histogram(normed_delta_n_spectra, legend=false, length=51)
    # title!(L"ESD of Normalized/Uncentered $\Delta_n$, $n=$%$(n), $p=$%$(p)")
    # savefig(string(figure_path, "jl_normed_deltan_$file_counter.png"))

    # if n * p <= log(n)

    #     histogram(normed_delta_n_spectra, legend=false, length=51, ylim=(0, 1000))
    #     title!(L"ESD of Normalized/Uncentered $\Delta_n$, $n=$%$(n), $p=$%$(p)")
    #     savefig(string(figure_path, "jl_normed_deltan_truncated_$file_counter.png"))

    #     histogram(normed_delta_n_spectra, legend=false, length=51, ylim=(0, 50))
    #     title!(L"ESD of Normalized/Uncentered $\Delta_n$, $n=$%$(n), $p=$%$(p)")
    #     savefig(string(figure_path, "jl_normed_deltan_truncated_small_$file_counter.png"))

    # end

    # L_n spectra
    normed_centered_L_spectra = calculate_Ln_spectra(L_n, n, p)
    histogram(normed_centered_L_spectra, legend=false, length=51)
    title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p)")
    savefig(string(figure_path, "jl_normedcentered_Ln_$file_counter.png"))

    print("jl_normedcentered_Ln_$file_counter.png done")

    if n * p <= log(n)

        histogram(normed_centered_L_spectra, legend=false, length=51, ylim=(0, 1000))
        title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p)")
        savefig(string(figure_path, "jl_normedcentered_Ln_truncated_$file_counter.png"))

        print("jl_normedcentered_Ln_truncated_$file_counter.png done")

        histogram(normed_centered_L_spectra, legend=false, length=51, ylim=(0, 50))
        title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p)")
        savefig(string(figure_path, "jl_normedcentered_Ln_truncated_small_$file_counter.png"))

        print("jl_normedcentered_Ln_truncated_small_$file_counter.png done")

    end

    normed_L_spectra = calculate_Ln_spectra(L_n, n, p, false)
    histogram(normed_L_spectra, legend=false, length=51)
    title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p)")
    savefig(string(figure_path, "jl_normed_Ln_$file_counter.png"))

    print("jl_normed_Ln_$file_counter.png done")

    if n * p <= log(n)

        histogram(normed_L_spectra, legend=false, length=51, ylim=(0, 1000))
        title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p)")
        savefig(string(figure_path, "jl_normed_Ln_truncated_$file_counter.png"))

        print("jl_normed_Ln_truncated_$file_counter.png done")

        histogram(normed_L_spectra, legend=false, length=51, ylim=(0, 200))
        title!(L"$\mu(\textbf{L}_n)$, $n=$%$(n), $p=$%$(p)") 
        savefig(string(figure_path, "jl_normed_Ln_truncated_small_$file_counter.png"))

        print("jl_normed_Ln_truncated_small_$file_counter.png done")
    
    end

    file_counter += 1

end