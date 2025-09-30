using LinearAlgebra

export safe_linear_solve, safe_linear_solve!, add_diagonal_regularization!

"""
    safe_linear_solve(A, b; warn_on_singular=true)

Solves the linear system Ax = b with robust handling of ill-conditioned matrices.

# Arguments
- `A`: Coefficient matrix
- `b`: Right-hand side vector or matrix  
- `warn_on_singular`: Whether to issue warnings when ill-conditioned matrices are detected

"""
function safe_linear_solve(A::AbstractMatrix, b::AbstractVecOrMat; warn_on_singular = true)
    try
        return A \ b
    catch e
        if e isa SingularException
            warn_on_singular && @warn "Ill-conditioned matrix detected (cond=$(cond(A))). Using pseudoinverse solve."
            return pinv(A) * b
        else
            rethrow(e)
        end
    end
end
"""
    safe_linear_solve!(x, A, b; kwargs...)

In-place version of `safe_linear_solve` that writes into `x`.
Errors if shapes don’t match.
"""
function safe_linear_solve!(x::AbstractVecOrMat, A::AbstractMatrix, b::AbstractVecOrMat; kwargs...)
    n = size(A, 2)
    ncols = ndims(b) == 1 ? 1 : size(b, 2)
    if size(x) != (n, ncols)
        throw(DimensionMismatch("x has size $(size(x)), expected ($(n), $(ncols))"))
    end
    x .= safe_linear_solve(A, b; kwargs...)
    return x
end

"""
    add_diagonal_regularization!(cov_matrix; regularization_factor=sqrt(eps(eltype(cov_matrix))))

Adds diagonal regularization to a covariance matrix to prevent singular matrix issues.

# Arguments
- `cov_matrix`: The covariance matrix to regularize (modified in-place)
- `regularization_factor`: Regularization amount.
"""
function add_diagonal_regularization!(
    cov_matrix::AbstractMatrix;
    regularization_factor::Union{Real, Nothing} = sqrt(eps(eltype(cov_matrix))),
)
    cov_matrix[diagind(cov_matrix)] .+= regularization_factor
    return cov_matrix
end
