using Distributions, StatsBase, Random, LinearAlgebra
using QuadGK, Optim
using SpecialFunctions

export median_constraint, quartile_constraint, quartile_constraint

# -----------------------------------------------------------------
# Coefficients for 3-point transformations, from median

function get_xform_coeffs(median::FT, min::FT, max::FT) where {FT<: Real}
    @assert min < median < max
    # no way to dispatch method on runtime values of isinf()?
    return _get_xform_coeffs(median, (isinf(min) ? nothing : min), (isinf(max) ? nothing : max)) 
end

function _get_xform_coeffs(median::FT, min::FT, ::Nothing) where {FT<: Real}
    # min > -Infinity, max = + Infinity
    return (-min, 1., (median - min), 0.)
end

function _get_xform_coeffs(median::FT, ::Nothing, max::FT) where {FT<: Real}
    # min = -Infinity, max < + Infinity
    return ((median - max), 0., -max, 1.)
end

function _get_xform_coeffs(median::FT, min::FT, max::FT) where {FT<: Real}
    # general case
    return (
        -min * (median - max),
        (median - max),
        -max * (median - min),
        (median - min)
    )
end

# 3-point transform
function w_from_z(z::FT, c::NTuple{4,FT}) where {FT <: Real}
    return (c[1] + z*c[2])/(c[3] + z*c[4])
end

function w_jacobian(z::FT, c::NTuple{4,FT}) where {FT <: Real}
    # dlog(w_from_z) / dz
    return (c[2]*c[3] - c[1]*c[4])/(
        (c[1] + z*c[2])*(c[3] + z*c[4])
    )
end

function z_from_w(w::FT, c::NTuple{4,FT}) where {FT <: Real}
    return -(c[1] - w*c[3])/(c[2] - w*c[4])
end

# -----------------------------------------------------------------
# Coefficients for 5-point transformations, from quantiles

# Expressions for when min and/or max are +/- Infinity were evaluated with a 
# computer algebra package as appropriate limits of the general case (last method in
# this section), on the grounds that this gives more manageable code than coding for
# these conditions by dropping appropriate subleading terms in the determinant
# evaluations in the general case. 

function quantile(p::FT) where {FT<: Real}
    # returns x such that integral of unit normal from -Infinity to x is 1/q.
    p = p > 0.5 ? 1-p : p
    return sqrt(2) * erfinv(2*p - 1)
end

function get_xform_coeffs(qs::AbstractVector{FT}, min::FT, max::FT, quantile_::FT=0.25) where {FT<: Real}
    @assert length(qs) == 3
    @assert min < qs[1] < qs[2] < qs[3] < max
    # no way to dispatch method on runtime values of isinf()?
    exp_zp = exp(quantile(quantile_))
    return _get_xform_coeffs(qs, (isinf(min) ? nothing : min), (isinf(max) ? nothing : max), exp_zp) 
end
            
function _get_xform_coeffs(qs::AbstractVector{FT}, ::Nothing, ::Nothing, exp_zp::FT) where {FT<: Real}
    # min = -Infinity, max = + Infinity
    # get a 3-point transformation in this limit
    return (
        qs[2]*qs[3]*exp_zp + qs[1]*(qs[2] - qs[3]*(1 + exp_zp)),
        qs[3] + qs[1]*exp_zp - qs[2]*(1 + exp_zp),
        qs[2]*qs[3] + qs[1]*qs[2]*exp_zp - qs[1]*qs[3]*(1 + exp_zp),
        qs[1] + qs[3]*exp_zp - qs[2]*(1 + exp_zp)
    )
end
    
function _get_xform_coeffs(qs::AbstractVector{FT}, min::FT, ::Nothing, exp_zp::FT) where {FT<: Real}
    # min > -Infinity, max = + Infinity
    return (
        min*(
            (min - qs[1])*qs[1]*(qs[2] - qs[3]) 
            + qs[2]*(-min + qs[2])*(qs[1] - qs[3])*exp_zp 
            + (qs[1] - qs[2])*(min - qs[3])*qs[3]*exp_zp^2
        ),(
        qs[1]^2*(qs[2] - qs[3]) + qs[2]^2*(-qs[1] + qs[3])*exp_zp 
            + (qs[1] - qs[2])*qs[3]^2*exp_zp^2 
            + min^2*(-1 + exp_zp)*(qs[2] - qs[3] - qs[1]*exp_zp + qs[2]*exp_zp)
        ),
        (-1 + exp_zp)*(
            qs[2]*qs[3]*exp_zp + min*(qs[3] + qs[1]*exp_zp - qs[2]*(1 + exp_zp)) 
            + qs[1]*(qs[2] - qs[3]*(1 + exp_zp))
        ),(
            (min - qs[1])*(min - qs[2])*(-qs[1] + qs[2])*qs[3] 
            + (min - qs[1])*qs[2]*(min - qs[3])*(qs[1] - qs[3])*exp_zp 
            - qs[1]*(min - qs[2])*(min - qs[3])*(qs[2] - qs[3])*exp_zp^2
        ),(
            (min - qs[1])*(min - qs[2])*(qs[1] - qs[2]) 
            - (min - qs[1])*(min - qs[3])*(qs[1] - qs[3])*exp_zp 
            + (min - qs[2])*(min - qs[3])*(qs[2] - qs[3])*exp_zp^2
        ),
        0.
    )
end
        
function _get_xform_coeffs(qs::AbstractVector{FT}, ::Nothing, max::FT, exp_zp::FT) where {FT<: Real}
    # min = -Infinity, max < + Infinity
    return (
        (
            qs[1]*(max - qs[2])*(max - qs[3])*(qs[2] - qs[3]) 
            + (-max + qs[1])*qs[2]*(max - qs[3])*(qs[1] - qs[3])*exp_zp 
            + (max - qs[1])*(max - qs[2])*(qs[1] - qs[2])*qs[3]*exp_zp^2
        ),(
            -(max - qs[2])*(max - qs[3])*(qs[2] - qs[3])
            + (max - qs[1])*(max - qs[3])*(qs[1] - qs[3])*exp_zp 
            + (max - qs[1])*(max - qs[2])*(-qs[1] + qs[2])*exp_zp^2
        ),
        0.,
        max*(
            (qs[1] - qs[2])*qs[3]*(-max + qs[3]) 
            + (max - qs[2])*qs[2]*(qs[1] - qs[3])*exp_zp 
            + qs[1]*(-max + qs[1])*(qs[2] - qs[3])*exp_zp^2
        ),(
            (-qs[1] + qs[2])*qs[3]^2 + qs[2]^2*(qs[1] - qs[3])*exp_zp 
            + qs[1]^2*(-qs[2] + qs[3])*exp_zp^2 
            + max^2*(-1 + exp_zp)*(-qs[1] + qs[2] + qs[2]*exp_zp - qs[3]*exp_zp)
        ),
        (-1 + exp_zp)*(
            (qs[1] - qs[2])*(max - qs[3]) 
            + (-max + qs[1])*(qs[2] - qs[3])*exp_zp       
        )  
    )
end

# from https://discourse.julialang.org/t/minor-matrix/18328/3
struct OmitOneRange{IT} <: AbstractVector{IT where {IT <: Integer}}
    start::IT
    stop::IT
    omit::IT
    function OmitOneRange(start::IT, stop::IT, omit::IT) where {IT <: Integer}
        # inner constructor: always do bounds checks
        @assert start <= omit <= stop
        new{IT}(start, stop, omit)
    end
end
OmitOneRange(start::IT, stop::IT, nothing) where {IT <: Integer} = start:stop

Base.length(s::OmitOneRange)= s.stop - s.start
Base.size(s::OmitOneRange) = (length(s),)
Base.getindex(s::OmitOneRange, i) = s.start + i - 1 < s.omit ? s.start + i - 1 : s.start + i

function _submatrix(m::AbstractMatrix{FT}, r::Union{IT, Nothing}, c::Union{IT,Nothing}) where {FT<: Real, IT <: Integer}
    nrow, ncol = size(m)
    return m[OmitOneRange(1, nrow, r), OmitOneRange(1, ncol, c)]
end

function _get_xform_coeffs(qs::AbstractVector{FT}, min::FT, max::FT, exp_zp::FT) where {FT<: Real}
    # general case; evaluate determinant
    function _m_row(z::FT, w::FT) where {FT<: Real}
        return [1.0 z z^2 -w -w*z -w*z^2]
    end
    
    function _get_coeff(m::AbstractMatrix{FT}, k::IT) where {FT<: Real, IT <: Integer}
        # compute determinant of minor, taking correct limit of w-> + Infinity
        # for w in first row of m.
        mm = _submatrix(m, nothing, k)
        j_start = (k <= 3 ? 3 : 4)
        return (-1)^(k+1) * sum(
            (j -> (-1)^(1+j) * mm[1,j] * det(_submatrix(mm, 1, j))),
            j_start:size(m,1)
        )
    end

    m = vcat(
        _m_row(max, 1.0),       # max; 1.0 is "infinity"
        _m_row(qs[3], 1/exp_zp), # (1-p)th quantile
        _m_row(qs[2], 1.0),     # median
        _m_row(qs[1], exp_zp),   # pth quantile
        _m_row(min, 0.0)        # min
    )
    return Tuple([_get_coeff(m, k) for k in 1:6])
end

function w_from_z(z::FT, c::NTuple{6,FT}) where {FT <: Real}
    return (c[1] + z*c[2] + z^2*c[3])/(c[4] + z*c[5] + z^2*c[6])
end

function w_jacobian(z::FT, c::NTuple{6,FT}) where {FT <: Real}
    # dlog(w_from_z) / dz
    return (1/z)*(
        (2*c[4] + z*c[5])/(c[4] + z*c[5] + z^2*c[6])
        -(2*c[1] + z*c[2])/(c[1] + z*c[2] + z^2*c[3])
    )
end

# TODO: verify we chose right branch
function z_from_w(w::FT, c::NTuple{6,FT}) where {FT <: Real}
    return -((c[2] - w*c[5]) - sqrt(
            (c[2] - w*c[5])^2 - 4(c[1] - w*c[4])*(c[3] - w*c[6])
            ))/(2*(c[3] - w*c[6]))
end

# -----------------------------------------------------------------

"""
    median_constraint(median::FT, lower_bound::FT, upper_bound::FT)

Constructor for a Constraint mapping the median of a distribution to a 
specified value.
"""
function median_constraint(median::FT, lower_bound::FT, upper_bound::FT) where {FT<: Real}
    cs = get_xform_coeffs(median, lower_bound, upper_bound)
    c_to_u = (z -> log(w_from_z(z, cs)))
    jacobian = (z -> w_jacobian(z, cs))
    u_to_c = (u -> z_from_w(exp(u), cs))
    return Constraint(c_to_u, jacobian, u_to_c)
end

"""
    quartile_constraint (median::FT, lower_bound::FT, upper_bound::FT)

Constructor for a Constraint mapping three quartiles of a distribution to
specified values.
"""
function quartile_constraint(qs::AbstractVector{FT}, lower_bound::FT, upper_bound::FT) where {FT<: Real}
    cs = get_xform_coeffs(qs, lower_bound, upper_bound)
    c_to_u = (z -> log(w_from_z(z, cs)))
    jacobian = (z -> w_jacobian(z, cs))
    u_to_c = (u -> z_from_w(exp(u), cs))
    return Constraint(c_to_u, jacobian, u_to_c)
end

"""
    quantile_constraint (median::FT, lower_bound::FT, upper_bound::FT, p::FT)

Constructor for a Constraint mapping the ``p``th, median, and ``1-p``th
quantiles of a distribution to specified values.
"""
function quartile_constraint(qs::AbstractVector{FT}, lower_bound::FT, upper_bound::FT, p::FT) where {FT<: Real}
    cs = get_xform_coeffs(qs, lower_bound, upper_bound, p)
    c_to_u = (z -> log(w_from_z(z, cs)))
    jacobian = (z -> w_jacobian(z, cs))
    u_to_c = (u -> z_from_w(exp(u), cs))
    return Constraint(c_to_u, jacobian, u_to_c)
end
