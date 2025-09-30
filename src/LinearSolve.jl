using LinearAlgebra

export safe_linear_solve, safe_linear_solve!

"""
    safe_linear_solve(A, b; warn_on_singular=true)

Solves the linear system Ax = b with robust handling of ill-conditioned matrices.

# Arguments
- `A`: Coefficient matrix
- `b`: Right-hand side vector or matrix  
- `warn_on_singular`: Whether to issue warnings when ill-conditioned matrices are detected

# Algorithm
1. **Primary**: Standard solve `A \\ b` (when matrix is well-conditioned)
2. **Fallback**: Pseudoinverse solve using normal equations or Moore-Penrose pseudoinverse
"""
function safe_linear_solve(
    A::AbstractMatrix,
    b::AbstractVecOrMat;
    warn_on_singular = true,
    max_cond = 1e12,
)
    cond_A = cond(A)
    is_ill_conditioned = cond_A > max_cond
    is_ill_conditioned || return A \ b

    warn_on_singular && @warn "Ill-conditioned matrix detected (cond=$cond_A). Using pseudoinverse solve."
    return pinv(A) * b
end
"""
    safe_linear_solve!(x, A, b; kwargs...)

In-place version of `safe_linear_solve` that writes into `x`.
Errors if shapes don’t match.
"""
function safe_linear_solve!(
    x::AbstractVecOrMat,
    A::AbstractMatrix,
    b::AbstractVecOrMat;
    kwargs...
)
    n = size(A, 2)
    ncols = ndims(b) == 1 ? 1 : size(b, 2)
    if size(x) != (n, ncols)
        throw(DimensionMismatch("x has size $(size(x)), expected ($(n), $(ncols))"))
    end
    x .= safe_linear_solve(A, b; kwargs...)
    return x
end
