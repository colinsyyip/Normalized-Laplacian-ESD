using LinearAlgebra
using Distributions 
using BlockDiagonals

function matrix_generator(n, p)
    A_n_raw = rand(Distributions.Bernoulli(p), n, n)
    A_n = Symmetric(A_n_raw)
    A_n[diagind(A_n)] .= 0
    A_n = Float32.(A_n)

    D_n_vals = sum(A_n, dims = 2)
    D_n = zeros(n, n)
    D_n[diagind(D_n)] = D_n_vals

    delta_n = D_n - A_n

    inv_sqrt_D_n = zeros(n, n)
    inv_sqrt_D_n_vals = 1 ./ sqrt.(D_n_vals)
    inv_sqrt_D_n[diagind(inv_sqrt_D_n)] = inv_sqrt_D_n_vals
    inv_sqrt_D_n = replace!(inv_sqrt_D_n, Inf => 0)

    L = I - inv_sqrt_D_n * A_n * inv_sqrt_D_n

    return [A_n, delta_n, L]
end


function calculate_An_spectra(A_n, n, p) 
    centered_An = A_n .- p
    # Try with no 1-p for sparse alpha case
    normed_centered_An = centered_An .* 1/sqrt(n * p * (1 - p))
    An_normed_spectra = eigen(normed_centered_An).values

    return An_normed_spectra
end


function calculate_Deltan_spectra(delta_n, n, p, center=true) 
    if center
        centering_delta_matrix = -p .* ones(n, n)
        centering_delta_matrix[diagind(centering_delta_matrix)] .= (n - 2) * p
        adj_delta_n = (delta_n - centering_delta_matrix) ./ sqrt(n * p)
    else 
        adj_delta_n = delta_n ./ sqrt(n * p)
    end

    adj_delta_n_spectra = eigen(adj_delta_n).values

    return adj_delta_n_spectra
end


function calculate_Ln_spectra(L_n, n, p, center=true) 
    if center
        centering_L_matrix = (0 - 1/(n-1)) .* ones(n, n)
        centering_L_matrix[diagind(centering_L_matrix)] .= 1
        adj_L_n = (L_n - centering_L_matrix) .* sqrt((n * p)/(1 - p))
    else 
        adj_L_n = L_n .* sqrt((n * p)/(1 - p))
    end

    adj_L_n_spectra= eigen(adj_L_n).values

    return adj_L_n_spectra
end

function block_matrix_generator(n, m_vec, p_vec, q)
    if sum(m_vec) != n
        throw("m_vec must sum to n")
    end

    if any(p_vec .<= q)
        throw("All p_vec must be > q")
    end

    A_n_components = Matrix{Float64}[]

    for (p, m) in zip(p_vec, m_vec)
        A_n_raw = rand(Distributions.Bernoulli(p), m, m)
        A_n = Symmetric(A_n_raw)
        A_n[diagind(A_n)] .= 0
        A_n = Float32.(A_n)
        
        push!(A_n_components, A_n)
    end

    A_n = Matrix(BlockDiagonal(A_n_components))

    start_x_idx = 1
    start_y_idx = 1
    end_x_idx = 1
    end_y_idx = 1
    m_push_idx_start = 2
    
    # Walks down "columns" 
    for m in m_vec
        end_y_idx = start_y_idx + m - 1
        start_x_idx += m
        row_start_x_idx = start_x_idx
        
        # Walks across "rows"
        for m_push in m_vec[m_push_idx_start:length(m_vec)]
            end_x_idx = row_start_x_idx + m_push - 1

            A_n[start_y_idx:end_y_idx, row_start_x_idx:end_x_idx] = rand(Distributions.Bernoulli(q), m,m_push)

            row_start_x_idx += m_push
        end
        start_y_idx += m
        m_push_idx_start += 1
    end

    A_n = Symmetric(A_n)

    D_n_vals = sum(A_n, dims = 2)
    D_n = zeros(n, n)
    D_n[diagind(D_n)] = D_n_vals

    delta_n = D_n - A_n

    inv_sqrt_D_n = zeros(n, n)
    inv_sqrt_D_n_vals = 1 ./ sqrt.(D_n_vals)
    inv_sqrt_D_n[diagind(inv_sqrt_D_n)] = inv_sqrt_D_n_vals
    inv_sqrt_D_n = replace!(inv_sqrt_D_n, Inf => 0)

    L = I - inv_sqrt_D_n * A_n * inv_sqrt_D_n

    return [A_n, delta_n, L]

    return A_n
end