using DocStringExtensions
using LinearAlgebra
using Statistics
using Random
import Base: size
export Observation, Minibatcher, FixedMinibatcher, RandomFixedSizeMinibatcher, ObservationSeries, SVDplusD, DminusTall
export get_samples,
    get_covs,
    get_cov_size,
    get_diag_cov,
    get_svd_cov,
    get_tall_cov,
    inv_cov,
    get_inv_covs,
    get_names,
    get_indices,
    get_metadata,
    combine_observations,
    get_obs_noise_cov,
    get_obs_noise_cov_inv,
    get_obs,
    create_new_epoch!,
    get_minibatches,
    get_method,
    get_rng,
    get_minibatch_size,
    get_length_epoch,
    get_observations,
    get_minibatch_index,
    get_current_minibatch_index,
    get_minibatcher,
    update_minibatch!,
    get_minibatch,
    get_current_minibatch,
    no_minibatcher,
    tsvd_mat,
    tsvd_cov_from_samples,
    lmul_without_build

## A new wrapper for a sum-of-covariances type
abstract type SumOfCovariances end

# svd plus diagonal
"""
$(TYPEDEF)

Storage for a covariance matrix of the form `D + USV'` for Diagonal D, and SVD decomposition USV'.
Note the inverse of this type (as computed through `inv_cov(...)`) will be stored compactly as a `DminusTall` type.

$(TYPEDFIELDS)
"""
struct SVDplusD <: SumOfCovariances
    "summand of covariance matrix stored with SVD decomposition"
    svd_cov::SVD
    "summand of covariance matrix stored as a diagonal matrix"
    diag_cov::Diagonal

    function SVDplusD(s_in::SVD, d_in::Diagonal)
        mat_sizes = (get_cov_size(s_in), get_cov_size(d_in))
        if !(mat_sizes[2] == mat_sizes[1])
            throw(
                ArgumentError(
                    "all covariances provided must have the same size (as they are to be summed), instead recieved different sizes: $(mat_sizes)",
                ),
            )
        end

        return new(s_in, d_in)

    end
end

get_svd_cov(spd::SpD) where {SpD <: SVDplusD} = spd.svd_cov
get_diag_cov(spd::SpD) where {SpD <: SVDplusD} = spd.diag_cov
get_cov_size(spd::SpD) where {SpD <: SVDplusD} = size(get_diag_cov(spd), 1)
Base.size(spd::SpD) where {SpD <: SVDplusD} = size(get_diag_cov(spd))

function Base.:(==)(spd1::SpD1, spd2::SpD2) where {SpD1 <: SVDplusD, SpD2 <: SVDplusD}
    fn = unique([fieldnames(SpD1)...; fieldnames(SpD2)...])
    x = [false for f in fn]
    for (i, f) in enumerate(fn)
        x[i] = (getfield(spd1, Symbol(f)) == getfield(spd2, Symbol(f)))
    end
    return all(x)
end


# the inverse of SVD plus diagonal is stored like this:
"""
$(TYPEDEF)

Storage for a covariance matrix of the form `D - RR'` for Diagonal D, and (tall) matrix R.
Primary use case for this matrix is to compactly store the inverse of the `SVDplusD` type.

$(TYPEDFIELDS)
"""
struct DminusTall{D <: Diagonal, AM <: AbstractMatrix} <: SumOfCovariances
    "summand of covariance matrix stored as a diagonal matrix"
    diag_cov::D
    "summand of covariance matrix stored as an abstract matrix"
    tall_cov::AM
end

get_tall_cov(dmt::DmT) where {DmT <: DminusTall} = dmt.tall_cov
get_diag_cov(dmt::DmT) where {DmT <: DminusTall} = dmt.diag_cov
get_cov_size(dmt::DmT) where {DmT <: DminusTall} = size(get_diag_cov(dmt), 1)
Base.size(dmt::DmT) where {DmT <: DminusTall} = size(get_diag_cov(dmt))
function Base.:(==)(dmt1::DmT1, dmt2::DmT2) where {DmT1 <: DminusTall, DmT2 <: DminusTall}
    fn = unique([fieldnames(DmT1)...; fieldnames(DmT2)...])
    x = [false for f in fn]
    for (i, f) in enumerate(fn)
        x[i] = (getfield(dmt1, Symbol(f)) == getfield(dmt2, Symbol(f)))
    end
    return all(x)
end


function SVDplusD(s_in::SVD, us_in::US) where {US <: UniformScaling}
    return SVDplusD(s_in, us_in(size(s_in.U, 1))) # make US into Diag

end

function SVDplusD(s_in::SVD, am_in::AM) where {AM <: AbstractMatrix}
    @warn("SVDplusD requires Diagonal matrix type, converting input X to Diagonal(X)")
    return SVDplusD(s_in, Diagonal(am_in))
end



# wrapper for tsvd - putting solution in SVD object
"""
$(TYPEDSIGNATURES)

For a given matrix `X` and rank `r`, return the truncated SVD for X as a LinearAlgebra.jl `SVD` object. Setting `return_inverse=true` also return it's psuedoinverse X⁺.
"""
function tsvd_mat(X, r::Int; return_inverse = false, quiet = false, kwargs...)

    if isa(X, UniformScaling)
        if return_inverse
            return svd(X(r)), svd(inv(X)(r))
        else
            return svd(X(r))
        end
    else
        rx = rank(X)
        mindim = minimum(size(X))
        SS = svd(X)

        if rx <= r
            if rx < r && !quiet
                @warn(
                    "Requested truncation to rank $(r) for an input matrix of rank $(rx). Performing (truncated) SVD for rank $(rx) matrix."
                )
            end
        end
        trunc = min(r, rx)

        # Note this following call scales poorly with N. So if r << N AND r=O(1-20), then we could replace this with TSVD.jl "tsvd(X,r=...)", poor accuracy with ~ r>=20. Leaving for now.
        U, s, Vt = SS.U[:, 1:trunc], SS.S[1:trunc], SS.Vt[1:trunc, :]

        if return_inverse
            return SVD(U, s, Vt), SVD(permutedims(Vt, (2, 1)), 1 ./ s, permutedims(U, (2, 1)))
        else
            return SVD(U, s, Vt)
        end
    end
end

function tsvd_mat(X; return_inverse = false, quiet = false, kwargs...)
    if isa(X, UniformScaling)
        throw(
            ArgumentError(
                "Cannot perform a low rank decomposition on `UniformScaling` type without providing a rank in the second argument.",
            ),
        )
    end
    return tsvd_mat(X, rank(X); return_inverse = return_inverse, quiet = quiet, kwargs...)
end

"""
$(TYPEDSIGNATURES)

For a given `sample_mat`, (with `data_are_columns = true`), rank "r" is optionally provided. Returns the SVD objects corresponding to the matrix `cov(sample_mat; dims=2)`. Efficient representation when `size(sample_mat,1) << size(sample_mat,2)`. Setting `return_inverse=true` also returns its psuedoinverse.

Example usage to make a low-rank covariance:
```
# "data"
n_trials = 30
output_dim = 1_000_000
Y = randn(output_dim, n_trials);

# the noise estimated from the samples (will have rank n_trials-1)
internal_cov = tsvd_cov_from_samples(Y)
internal_cov_lower_rank = tsvd_cov_from_samples(Y, n_trials-5)
```

If one also wishes to add a Diagonal matrix to `internal cov` to increase the rank in a compact fashion, use the `SVDplusD` object type
```
diag_cov = 1e-6*Diagonal(1:output_dim)
full_cov = SVDplusD(internal_cov, diag_cov)
```
Either can be passed in the `covariances` entry of an Observation
"""
function tsvd_cov_from_samples(
    sample_mat::AM,
    r::Int;
    data_are_columns::Bool = true,
    return_inverse = false,
    quiet = false,
    kwargs...,
) where {AM <: AbstractMatrix}
    mat = data_are_columns ? sample_mat : permutedims(sample_mat, (2, 1))
    FT = eltype(mat)
    N = size(mat, 2)
    if N > size(mat, 1) && !quiet
        @warn(
            "SVD representation is efficient when estimating high-dimensional covariance with few samples. \n here # samples is $(N), while the space dimension is $(size(mat,1)), and representation will be inefficient."
        )
    elseif N == 1
        throw(ArgumentError("Require multiple samples to estimate covariance matrix, only 1 provided"))
    end
    debiased_scaled_mat = FT(1 ./ sqrt(N - 1)) * (mat .- mean(mat, dims = 2))
    rk = min(rank(debiased_scaled_mat), r)

    if return_inverse
        A, Ainv = tsvd_mat(debiased_scaled_mat, rk; return_inverse = return_inverse, kwargs...)
        # A now represents the sqrt of the covariance, so we square the singular values 
        return SVD(A.U, A.S .^ 2, permutedims(A.U, (2, 1))), SVD(permutedims(Ainv.Vt, (2, 1)), Ainv.S .^ 2, Ainv.Vt)
    else
        A = tsvd_mat(debiased_scaled_mat, rk; return_inverse = return_inverse, kwargs...)
        # A now represents the sqrt of the covariance, so we square the singular values 
        return SVD(A.U, A.S .^ 2, permutedims(A.U, (2, 1)))
    end
end

function tsvd_cov_from_samples(
    sample_mat::AM;
    data_are_columns::Bool = true,
    return_inverse = false,
    quiet = false,
    kwargs...,
) where {AM <: AbstractMatrix}
    # (need to compute this to get the rank as debiasing can change it)
    mat = data_are_columns ? sample_mat : permutedims(sample_mat, (2, 1))
    rk = rank(mat .- mean(mat, dims = 2))

    return tsvd_cov_from_samples(
        sample_mat,
        rk;
        data_are_columns = data_are_columns,
        return_inverse = return_inverse,
        quiet = quiet,
        kwargs...,
    )
end

# TODO: Define == and copy for these structs
"""
$(TYPEDEF)

Structure that contains a (possibly stacked) observation. Defined by sample(s), noise covariance(s), and name(s)

Typical Constructors:
```
Observation(
    Dict(
        "samples" => [1,2,3],
        "covariances" => I(3),
        "names" => "one_two_three",
    ),
)
```
or
```
Observation([1,2,3], I(3), "one_two_three")
```
One can stack up multiple observations with `combine_observations`, (recommended), or by providing vectors of samples, covariances and names to the dictionary.

# Fields

$(TYPEDFIELDS)
"""
struct Observation{
    AV1 <: AbstractVector,
    AV2 <: AbstractVector,
    AV3 <: AbstractVector,
    AV4 <: AbstractVector,
    AV5 <: AbstractVector,
    MD,
}
    "A (vector of) observation vectors"
    samples::AV1
    "A (vector of) observation covariance matrices"
    covs::AV2
    "A (vector of) inverses of observation covariance matrices"
    inv_covs::AV3
    "A (vector of) name strings"
    names::AV4
    "A (vector of) indices of the contained observation blocks"
    indices::AV5
    "Metadata of any type that the user can group with the Observation"
    metadata::MD

end

"""
$(TYPEDSIGNATURES)

gets the `samples` field from the `Observation` object
"""
get_samples(o::Observation) = o.samples

"""
$(TYPEDSIGNATURES)

gets the `covs` field from the `Observation` object
"""
get_covs(o::Observation) = o.covs

"""
$(TYPEDSIGNATURES)

gets the `inv_covs` field from the `Observation` object
"""
get_inv_covs(o::Observation) = o.inv_covs

"""
$(TYPEDSIGNATURES)

gets the `names` field from the `Observation` object
"""
get_names(o::Observation) = o.names

"""
$(TYPEDSIGNATURES)

gets the `indices` field from the `Observation` object
"""
get_indices(o::Observation) = o.indices

"""
$(TYPEDSIGNATURES)

gets the `metadata` field from the `Observation` object
"""
get_metadata(o::Observation) = o.metadata

function Observation(obs_dict::Dict; metadata = nothing)
    if !all(["samples", "names", "covariances"] .∈ [collect(keys(obs_dict))])
        throw(
            ArgumentError(
                "input dictionaries must contain the keys: \"samples\", \"names\", \"covariances\", and optionally: \"inv_covariances\". Got $(keys(obs_dict))",
            ),
        )
    end
    samples = obs_dict["samples"]
    covariances = obs_dict["covariances"]
    names = obs_dict["names"]

    if !isa(samples, AbstractVector) # 1 -> [[1]]
        snew = [[samples]]
    else
        T = eltype(samples)
        samples_tmp = [convert(T, s) for s in samples] # to re-infer eltype  (if the user makes an eltype of "Any")
        if !isa(samples_tmp, AbstractVector{<:AbstractVector}) # [1,2] -> [[1,2]] 
            snew = [samples]
        else
            snew = samples_tmp  # [[1,2,3]]
        end
    end

    if !isa(covariances, AbstractVector) # [2 1;1 2] -> [[2 1;1 2]]
        ctmp = [covariances]
    else
        ctmp = covariances
    end
    # additionally provide a dimension for UniformScalings for covariances
    ctmp2 = []
    for (id, c) in enumerate(ctmp)
        if isa(c, UniformScaling)
            push!(ctmp2, Diagonal(c.λ * ones(length(snew[id])))) # get dim from samples
        elseif isa(c, Real) # treat number as uniform scaling
            push!(ctmp2, Diagonal(c * ones(length(snew[id]))))
        else
            push!(ctmp2, c)
        end
    end
    # then promote
    T = promote_type((typeof(c) for c in ctmp2)...)
    cnew = [convert(T, c) for c in ctmp2] # to re-infer eltype

    if !("inv_covariances" ∈ collect(keys(obs_dict)))
        inv_covariances = []
        for c in cnew # ensures its a vector
            push!(inv_covariances, inv_cov(c))
        end
    else
        inv_covariances = obs_dict["inv_covariances"]
    end
    if !isa(inv_covariances, AbstractVector) # [2 1;1 2] -> [[2 1;1 2]]
        ictmp = [inv_covariances]
    else
        ictmp = inv_covariances
    end
    # additionally provide a dimension for UniformScalings (if users provided inverse as US)
    ictmp2 = []
    for (id, c) in enumerate(ictmp)
        if isa(c, UniformScaling)
            push!(ictmp2, Diagonal(c.λ * ones(length(snew[id])))) # get dim from samples
        else
            push!(ictmp2, c)
        end
    end
    T = promote_type((typeof(c) for c in ictmp2)...)
    icnew = [convert(T, c) for c in ictmp2] # to re-infer eltype
    if !isa(names, AbstractVector) # "name" -> ["name"]
        nnew = [names]
    else
        T = promote_type((typeof(n) for n in names)...)
        nnew = [convert(T, n) for n in names] # to re-infer eltype
    end

    if !all([length(snew) == length(cnew), length(nnew) == length(cnew), length(icnew) == length(cnew)])
        throw(
            ArgumentError(
                "input dictionaries must contain the same number of objects. Got $(length(snew)) samples, $(length(cnew)) covs,  $(length(icnew)) inv_covs, and $(length(nnew)) names.",
            ),
        )
    end
    block_sizes = length.(snew)
    n_blocks = length(block_sizes)
    indices = [1:block_sizes[1]]
    if n_blocks > 1
        for i in 2:n_blocks
            push!(indices, (sum(block_sizes[1:(i - 1)]) + 1):sum(block_sizes[1:i]))
        end
    end

    # metadata - if multiple provided, dict-stored > keyword
    if "metadata" ∈ collect(keys(obs_dict))
        mnew = obs_dict["metadata"]
    else
        mnew = metadata
    end

    return Observation(snew, cnew, icnew, nnew, indices, mnew)

end

function Observation(
    sample::AV,
    obs_noise_cov::AMorUSorSVD,
    name::AS;
    kwargs...,
) where {
    AV <: AbstractVector,
    AMorUSorSVD <: Union{AbstractMatrix, UniformScaling, SVD, SumOfCovariances},
    AS <: AbstractString,
}
    return Observation(Dict("samples" => sample, "covariances" => obs_noise_cov, "names" => name); kwargs...)
end

function Observation(
    samples::AV1,
    obs_noise_covs::AV2,
    names::AV3;
    kwargs...,
) where {AV1 <: AbstractVector, AV2 <: AbstractVector, AV3 <: AbstractVector}
    return Observation(Dict("samples" => samples, "covariances" => obs_noise_covs, "names" => names); kwargs...)
end



"""
$(TYPEDSIGNATURES)

combines a vector of `Observation` objects into a single `Observation`
"""
function combine_observations(obs_vec::AV) where {AV <: AbstractVector}
    n_obs = length(obs_vec)

    snew = []
    cnew = []
    icnew = []
    nnew = []
    inew = []
    mnew = []
    shift = [0] # running shift to add to indexing 
    for obs in obs_vec
        @assert(nameof(typeof(obs)) == :Observation) # check it's a vector of Observations
        append!(snew, get_samples(obs))
        append!(cnew, get_covs(obs))
        append!(icnew, get_inv_covs(obs))
        append!(nnew, get_names(obs))
        indices = get_indices(obs)
        shifted_indices = [ind .+ shift[1] for ind in get_indices(obs)]
        append!(inew, shifted_indices)
        shift[1] = maximum(shifted_indices[end]) # increase the shift for the next "append"           
        md = get_metadata(obs)
        if isa(md, AbstractString) # strings are appendable but easier if pushed
            push!(mnew, md)
        elseif hasmethod(length, (typeof(md),)) # some types aren't appendable
            append!(mnew, md)
        else
            push!(mnew, md)
        end

    end

    #re-infer eltypes
    T = promote_type((typeof(s) for s in snew)...)
    snew2 = [convert(T, s) for s in snew]
    T = promote_type((typeof(c) for c in cnew)...)
    cnew2 = [convert(T, c) for c in cnew]
    T = promote_type((typeof(c) for c in icnew)...)
    icnew2 = [convert(T, c) for c in icnew]
    T = promote_type((typeof(n) for n in nnew)...)
    nnew2 = [convert(T, n) for n in nnew]
    T = promote_type((typeof(i) for i in inew)...)
    inew2 = [convert(T, i) for i in inew]

    return Observation(snew2, cnew2, icnew2, nnew2, inew2, mnew)
end

"""
$(TYPEDSIGNATURES)

if `build=true`, returns the stacked vector of observed samples `samples`(default), otherwise it calls `get_samples`
"""
function get_obs(o::Observation; build = true)
    if !build # return the blocks directly
        return get_samples(o)
    else
        indices = get_indices(o)
        samples = get_samples(o)
        obs_sample = zeros(maximum(indices[end]))
        for (idx, sample) in zip(indices, samples)
            obs_sample[idx] = sample
        end
        return obs_sample
    end
end


function build_cov!(out, idx, c::AM) where {AM <: AbstractMatrix}
    view(out, idx, idx) .= c
end

function build_cov!(out, idx, c::SVD)
    if size(c.U, 1) == size(c.U, 2) # then work with Vt
        view(out, idx, idx) .= c.Vt' * Diagonal(c.S) * c.Vt
    else
        view(out, idx, idx) .= c.U * Diagonal(c.S) * c.U'
    end
end


function build_cov!(out, idx, c::SpD) where {SpD <: SVDplusD}
    tmp = zeros(length(idx), length(idx))

    c_diag = get_diag_cov(c)
    build_cov!(tmp, 1:length(idx), c_diag)
    view(out, idx, idx) .+= tmp

    c_svd = get_svd_cov(c)
    build_cov!(tmp, 1:length(idx), c_svd)
    view(out, idx, idx) .+= tmp

end
function build_cov!(out, idx, c::DmT) where {DmT <: DminusTall}
    tmp = zeros(length(idx), length(idx))

    c_diag = get_diag_cov(c)
    build_cov!(tmp, 1:length(idx), c_diag)
    view(out, idx, idx) .+= tmp

    c_tall = get_tall_cov(c)
    view(out, idx, idx) .-= c_tall * c_tall'

end

function inv_cov(a::AM) where {AM <: AbstractMatrix}
    return inv(a)
end
function inv_cov(a::SVD)
    return SVD(permutedims(a.Vt, (2, 1)), 1 ./ a.S, permutedims(a.U, (2, 1)))
end

function inv_cov(a::SpD) where {SpD <: SVDplusD}

    # Compute the cholesky quantity useful for evaluating woodbury formula for computing inverse product (s_in + Din)^-1 * mat
    a_diag = get_diag_cov(a)
    a_svd = get_svd_cov(a)
    Dinv = inv(a_diag)
    S = Diagonal(a_svd.S)
    if size(a_svd.U, 1) == size(a_svd.U, 2)
        U = a_svd.Vt'
        Vt = a_svd.Vt
    else
        U = a_svd.U
        Vt = a_svd.U'
    end
    T_nonsym = inv(S + S * Vt * Dinv * U * S) # often non-symmetric from rounding error
    cholT = cholesky(0.5 * (T_nonsym + T_nonsym'))
    inv_Lfactor = Dinv * U * S * cholT.L
    # (s_in + Din)^-1 * Y = Dinv*Y - inv_Lfactor*inv_Lfactor'*Y

    return DminusTall(Dinv, inv_Lfactor)
end

function inv_cov(a::DmT) where {DmT <: DminusTall}
    @warn(
        "DminusTall is a convenience structure to handle the inverse of type `SVDplusD`\n it is not designed to be used as a fast implementation by itself. For low-rank+diagonal representations of the Covariance matrix, please create an `SVDplusD`, with help from utilities such as `tsvd_cov_from_samples()` "
    )

    a_diag = get_diag_cov(a)
    a_tall = get_tall_cov(a)
    return inv(a + a_tall * a_tall')

end

"""
$(TYPEDSIGNATURES)

if `build=true`, returns the block matrix of observation covariances `covs` (default), otherwise it calls `get_covs`
"""
function get_obs_noise_cov(o::Observation; build = true)

    if !build # return the blocks directly
        return get_covs(o)
    else # return the blocked matrix
        indices = get_indices(o)
        covs = get_covs(o)
        cov_full = zeros(maximum(indices[end]), maximum(indices[end]))
        for (idx, c) in zip(indices, covs)
            build_cov!(cov_full, idx, c)
        end

        return cov_full
    end
end

"""
$(TYPEDSIGNATURES)

if `build=true`, returns the block matrix of the inverses of the observation covariances `inv_covs` (default), otherwise it calls `get_inv_covs`
"""
function get_obs_noise_cov_inv(o::Observation; build = true)

    if !build # return the blocks directly
        return get_inv_covs(o)
    else # return the blocked matrix
        indices = get_indices(o)
        inv_covs = get_inv_covs(o)
        inv_cov_full = zeros(maximum(indices[end]), maximum(indices[end]))
        for (idx, c) in zip(indices, inv_covs)
            build_cov!(inv_cov_full, idx, c)
        end
        return inv_cov_full
    end
end

function Base.:(==)(ob_a::OB1, ob_b::OB2) where {OB1 <: Observation, OB2 <: Observation}
    fn = unique([fieldnames(OB1)...; fieldnames(OB2)...])
    x = [false for f in fn]
    for (i, f) in enumerate(fn)
        x[i] = (getfield(ob_a, Symbol(f)) == getfield(ob_b, Symbol(f)))
    end
    return all(x)
end



#####
# Batching methods for multiple observations
#####
abstract type Minibatcher end

create_new_epoch(mb::M, args...; kwargs...) where {M <: Minibatcher} = throw(
    MethodError(
        "current Minibatcher type has no function \"create_new_epoch!()\", please use one of the provided implementations, or create this method",
    ),
)

function Base.:(==)(m_a::M1, m_b::M2) where {M1 <: Minibatcher, M2 <: Minibatcher}
    fn = unique([fieldnames(M1)...; fieldnames(M2)...])
    x = [false for f in fn]
    for (i, f) in enumerate(fn)
        x[i] = (getfield(m_a, Symbol(f)) == getfield(m_b, Symbol(f)))
    end
    return all(x)
end

"""
$(TYPEDEF)

A `Minibatcher` that takes in a given epoch of batches. It creates a new epoch by either copying-in-order, or by shuffling, the provided batches.

# Fields

$(TYPEDFIELDS)

# Example epochs

```
given_batches = [[1,2,3], [4,5,6], [7,8,9]]
mb = FixedMinibatcher(given_batches)
# create_new_epoch(mb) = [[1,2,3],[4,5,6],[7,8,9]]

mb2 = FixedMinibatcher(given_batches, "random")
# create_new_epoch(mb2) = [[4,5,6],[1,2,3],[7,8,9]]
```
"""
struct FixedMinibatcher{AV1 <: AbstractVector, SS <: AbstractString, ARNG <: AbstractRNG} <: Minibatcher
    "explicit indices of the minibatched epoch"
    minibatches::AV1
    "method of selecting minibatches from the list for each epoch (\"order\" select in order, \"random\" generate a random selector)"
    method::SS
    "rng for sampling, if \"random\" method is selected"
    rng::ARNG
    function FixedMinibatcher(
        minibatches::AV1,
        method::SS,
        rng::ARNG,
    ) where {AV1 <: AbstractVector, SS <: AbstractString, ARNG <: AbstractRNG}
        if !isa(minibatches, AbstractVector) # 1 -> [[1]]
            mnew = [[minibatches]]
        else
            T = promote_type((typeof(m) for m in minibatches)...)
            minibatches_tmp = [convert(T, m) for m in minibatches] # to re-infer eltype  (if the user makes an eltype of "Any")
            if !isa(minibatches_tmp, AbstractVector{<:AbstractVector}) # [1,2] -> [[1,2]] 
                mnew = [minibatches]
            else
                mnew = minibatches_tmp  # [[1,2,3]]
            end
        end
        return new{typeof(mnew), typeof(method), typeof(rng)}(mnew, method, rng)
    end
end

function FixedMinibatcher(minibatches::AV) where {AV <: AbstractVector}
    # method 
    def_method = "order"
    def_rng = Random.default_rng()
    return FixedMinibatcher(minibatches, def_method, def_rng)
end
"""
$(TYPEDSIGNATURES)

constructs a `FixedMinibatcher` of given `epoch_size`, that generates an epoch of `1:epoch_size` and one minibatch that constitutes the whole epoch
"""
function no_minibatcher(epoch_size::Int = 1) #optional to provide size
    # method
    def_minibatch = [collect(1:epoch_size)]
    def_method = "order"
    def_rng = Random.default_rng()
    return FixedMinibatcher(def_minibatch, def_method, def_rng)
end

"""
$(TYPEDSIGNATURES)

gets the `minibatches` field from the `FixedMinibatcher` object
"""
get_minibatches(m::FM) where {FM <: FixedMinibatcher} = m.minibatches
"""
$(TYPEDSIGNATURES)

gets the `method` field from the `FixedMinibatcher` object
"""
get_method(m::FM) where {FM <: FixedMinibatcher} = m.method

"""
$(TYPEDSIGNATURES)

gets the `rng` field from the `FixedMinibatcher` object
"""
get_rng(m::FM) where {FM <: FixedMinibatcher} = m.rng

"""
$(TYPEDSIGNATURES)

updates the epoch by either copying ("order") the initialization minibatches, or by randomizing ("random") their order
"""
function create_new_epoch!(m::FM, args...; kwargs...) where {FM <: FixedMinibatcher}
    minibatches = get_minibatches(m)
    method = get_method(m)
    if method == "order"
        idx = 1:length(minibatches)
        new_epoch = minibatches[idx]
    elseif method == "random"
        rng = get_rng(m)
        idx = shuffle(rng, collect(1:length(minibatches)))
        new_epoch = minibatches[idx]
    else
        throw(
            ArgumentError(
                "method must be either \"order\" to select in order, or \"random\" to generate a random selector. Got $(method).",
            ),
        )
    end

    minibatches[:] = new_epoch # update the internal state
    return new_epoch


end

"""
$(TYPEDEF)

A `Minibatcher` that takes in a given epoch of batches. It creates a new epoch by either copying-in-order, or by shuffling, the provided batches.

# Fields

$(TYPEDFIELDS)

# Example epochs

```
for data = 1:10
batch_size = 3
mb = RandomFixedSizeMinibatcher(batch_size)
# create_new_epoch(mb) = [[6,7,5],[4,3,10],[9,2,8]] #  1 is trimmed

mb2 = RandomFixedSizeMinibatcher(batch_size, "extend")
# create_new_epoch(mb2) = [[2,9,1],[3,4,7],[10,5,1,6]] # last batch larger
```
"""
struct RandomFixedSizeMinibatcher{SS <: AbstractString, ARNG <: AbstractRNG, AV2 <: AbstractVector} <: Minibatcher
    "fixed size of minibatches"
    minibatch_size::Int
    "how to deal with remainder if minibatch-size doesn't divide the epoch size (\"trim\" - ignore trailing samples, \"extend\" - have a larger final minibatch)"
    method::SS
    "rng for sampling"
    rng::ARNG
    "explicit indices of the minibatched epoch"
    minibatches::AV2
end

function RandomFixedSizeMinibatcher(minibatch_size, method, rng)
    minibatches = Vector{Int}[]
    return RandomFixedSizeMinibatcher(minibatch_size, method, rng, minibatches)
end

RandomFixedSizeMinibatcher(minibatch_size::Int, method::SS) where {SS <: AbstractString} =
    RandomFixedSizeMinibatcher(minibatch_size, method, Random.default_rng())
RandomFixedSizeMinibatcher(minibatch_size::Int, rng::ARNG) where {ARNG <: AbstractRNG} =
    RandomFixedSizeMinibatcher(minibatch_size, "extend", rng)
RandomFixedSizeMinibatcher(minibatch_size::Int) = RandomFixedSizeMinibatcher(minibatch_size, Random.default_rng())

"""
$(TYPEDSIGNATURES)

gets the `minibatch_size` field from the `RandomFixesSizeMinibatcher` object
"""
get_minibatch_size(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.minibatch_size

"""
$(TYPEDSIGNATURES)

gets the `method` field from the `RandomFixesSizeMinibatcher` object
"""
get_method(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.method

"""
$(TYPEDSIGNATURES)

gets the `rng` field from the `RandomFixesSizeMinibatcher` object
"""
get_rng(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.rng

"""
$(TYPEDSIGNATURES)

gets the `minibatches` field from the `RandomFixesSizeMinibatcher` object
"""
get_minibatches(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.minibatches

"""
$(TYPEDSIGNATURES)

updates the epoch by randomizing the provided epoch indices. If the length of minibatches do not divide the length of the epoch, then the remainder is either ignored (default) or a final larger batch is created
"""
function create_new_epoch!(
    m::RFSM,
    epoch_in::AV,
    args...;
    kwargs...,
) where {RFSM <: RandomFixedSizeMinibatcher, AV <: AbstractVector}
    T = promote_type((typeof(e) for e in epoch_in)...)
    epoch = [convert(T, e) for e in epoch_in] # re-infer type
    if !(eltype(epoch) <: Int)
        throw(ArgumentError("the epoch must be a Vector{Int}, got eltype $(eltype(epoch))"))
    end
    epoch_size = length(epoch)
    rng = get_rng(m)
    indices = shuffle(rng, epoch)

    bs = get_minibatch_size(m)
    n_minibatches = Int(floor(epoch_size / bs))
    method = get_method(m)

    if method == "extend"
        new_epoch = [
            i < n_minibatches ? indices[((i - 1) * bs + 1):(i * bs)] : # bs sized minibatches
            indices[((n_minibatches - 1) * bs + 1):end] # final large minibatch < 2*bs sized
            for i in 1:n_minibatches
        ]
    elseif method == "trim"
        new_epoch = [indices[((i - 1) * bs + 1):(i * bs)] # bs sized minibatches
                     for i in 1:n_minibatches]
    else
        throw(
            ArgumentError(
                "method must be either \"trim\" (ignore trailing minibatches) or \"extend\" (have a larger final minibatch). Got $(method).",
            ),
        )
    end

    minibatches = get_minibatches(m)
    if length(minibatches) == 0
        append!(minibatches, new_epoch) # initial 
    else
        minibatches[:] = new_epoch
    end

    return new_epoch
end

#####
# Container for Multiple Observations and Minibatching
#####


"""
$(TYPEDEF)

Structure that contains multiple `Observation`s along with an optional `Minibatcher`. Stores all observations in `EnsembleKalmanProcess`, as well as defining the behavior of the `get_obs`, `get_obs_noise_cov`, and `get_obs_noise_cov_inv` methods

Typical Constructor
```
ObservationSeries(
    Dict(
        "observations" => vec_of_observations,
        "names" => names_of_observations,
        "minibatcher" => minibatcher,
    ),
)
```
# Fields

$(TYPEDFIELDS)
"""
struct ObservationSeries{AV1 <: AbstractVector, MM <: Minibatcher, AV2 <: AbstractVector, AV3 <: AbstractVector, MD}
    "A vector of `Observation`s to be used in the experiment"
    observations::AV1
    "A `Minibatcher` object used to define the minibatching"
    minibatcher::MM
    "A vector of string identifiers for the observations"
    names::AV2
    "The current index (epoch #, minibatch #) of the current minibatch, stored as a Dict"
    current_minibatch_index::Dict
    "The batch history (grouped by minibatch and epoch)"
    minibatches::AV3
    "Metadata of any type that the user can group with the ObservationSeries"
    metadata::MD
end

"""
$(TYPEDSIGNATURES)

gets the `observations` field from the `ObservationSeries` object
"""
get_observations(os::OS) where {OS <: ObservationSeries} = os.observations

"""
$(TYPEDSIGNATURES)

gets the `minibatches` field from the `ObservationSeries` object
"""
get_minibatches(os::OS) where {OS <: ObservationSeries} = os.minibatches

"""
$(TYPEDSIGNATURES)

gets the `names` field from the `ObservationSeries` object
"""
get_names(os::OS) where {OS <: ObservationSeries} = os.names

"""
$(TYPEDSIGNATURES)

gets the `current_minibatch_index` field from the `ObservationSeries` object
"""
get_current_minibatch_index(os::OS) where {OS <: ObservationSeries} = os.current_minibatch_index

"""
$(TYPEDSIGNATURES)

gets the `minibatcher` field from the `ObservationSeries` object
"""
get_minibatcher(os::OS) where {OS <: ObservationSeries} = os.minibatcher

"""
$(TYPEDSIGNATURES)

gets the `metadata` field from the `ObservationSeries` object
"""
get_metadata(os::OS) where {OS <: ObservationSeries} = os.metadata


function ObservationSeries(
    obs_vec_in::AV,
    minibatcher::MM,
    names_in::AV2,
    epoch::AV3;
    metadata = nothing,
) where {AV <: AbstractVector, MM <: Minibatcher, AV2 <: AbstractVector, AV3 <: AbstractVector}
    T = promote_type((typeof(o) for o in obs_vec_in)...)
    obs_vec = [convert(T, o) for o in obs_vec_in]

    T = promote_type((typeof(n) for n in names_in)...)
    names = [convert(T, n) for n in names_in]

    minibatches = [create_new_epoch!(minibatcher, epoch)]
    current_minibatch_index = Dict("epoch" => 1, "minibatch" => 1)
    return ObservationSeries(obs_vec, minibatcher, names, current_minibatch_index, minibatches, metadata)
end

function ObservationSeries(
    obs_vec::AV,
    minibatcher::MM,
    epoch_or_names::AV2;
    kwargs...,
) where {AV <: AbstractVector, MM <: Minibatcher, AV2 <: AbstractVector}
    T = promote_type((typeof(en) for en in epoch_or_names)...)
    epoch_or_names = [convert(T, en) for en in epoch_or_names]

    if eltype(epoch_or_names) <: Int
        epoch = epoch_or_names
        names = ["series_$(string(i))" for i in 1:length(obs_vec)]
    elseif eltype(epoch_or_names) <: AbstractString
        names = epoch_or_names
        epoch = collect(1:length(obs_vec))
    else
        throw(
            ArgumentError(
                "the third argument must be a Vector of Int's (if defining the epoch) or Strings (if defining the names of the observations. Got Vector of $(eltype(epoch_or_names))",
            ),
        )
    end

    return ObservationSeries(obs_vec, minibatcher, names, epoch; kwargs...)

end

function ObservationSeries(obs_vec::AV, minibatcher::MM; kwargs...) where {AV <: AbstractVector, MM <: Minibatcher}
    names = ["series_$(string(i))" for i in 1:length(obs_vec)]
    epoch = collect(1:length(obs_vec))
    return ObservationSeries(obs_vec, minibatcher, names, epoch; kwargs...)
end

function ObservationSeries(obs_vec::AV; kwargs...) where {AV <: AbstractVector}
    len_epoch = length(obs_vec)
    minibatcher = no_minibatcher(len_epoch)
    names = ["series_$(string(i))" for i in 1:len_epoch]
    epoch = collect(1:len_epoch)
    return ObservationSeries(obs_vec, minibatcher, names, epoch; kwargs...)
end

function ObservationSeries(obs::O, args...; kwargs...) where {O <: Observation}
    return ObservationSeries([obs], no_minibatcher(), args...; kwargs...)
end

function ObservationSeries(obs_series_dict::Dict)

    if !("observations" ∈ collect(keys(obs_series_dict)))
        throw(
            ArgumentError(
                "input dictionaries must contain the key: \"observations\". Got $(collect(keys(obs_series_dict)))",
            ),
        )
    end

    # First remove kwarg values   
    if "metadata" ∈ collect(keys(obs_series_dict))
        metadata = obs_series_dict["metadata"]
        keys_osd = filter(x -> x != "metadata", collect(keys(obs_series_dict)))
    else
        metadata = nothing
        keys_osd = collect(keys(obs_series_dict))
    end

    # call different constructors
    if issetequal(["observations"], keys_osd)
        return ObservationSeries(obs_series_dict["observations"], metadata = metadata)
    elseif issetequal(["observations", "minibatcher"], keys_osd)
        return ObservationSeries(obs_series_dict["observations"], obs_series_dict["minibatcher"], metadata = metadata)
    elseif issetequal(["observations", "minibatcher", "epoch"], keys_osd)
        return ObservationSeries(
            obs_series_dict["observations"],
            obs_series_dict["minibatcher"],
            obs_series_dict["epoch"],
            metadata = metadata,
        )
    elseif issetequal(["observations", "minibatcher", "names"], keys_osd)
        return ObservationSeries(
            obs_series_dict["observations"],
            obs_series_dict["minibatcher"],
            obs_series_dict["names"],
            metadata = metadata,
        )
    elseif issetequal(["observations", "minibatcher", "epoch", "names"], keys_osd)
        return ObservationSeries(
            obs_series_dict["observations"],
            obs_series_dict["minibatcher"],
            obs_series_dict["names"],
            obs_series_dict["epoch"],
            metadata = metadata,
        )
    else
        throw(
            ArgumentError(
                "input dictionaries must contain a subset of keys: [\"observations\", \"minibatcher\", \"names\", \"epoch\", \"metadata\"]. Got $(keys(obs_series_dict))",
            ),
        )
    end
end

"""
$(TYPEDSIGNATURES)

gets the number of minibatches in an epoch
"""
function get_length_epoch(os::OS) where {OS <: ObservationSeries}
    return length(get_minibatches(os)[1])
end

"""
$(TYPEDSIGNATURES)

returns the minibatch_index `Dict("epoch"=> x, "minibatch" => y)`, for a given `iteration` 
"""
function get_minibatch_index(os::OS, iteration::Int) where {OS <: ObservationSeries}
    len_epoch = get_length_epoch(os)
    return Dict("epoch" => ((iteration - 1) ÷ len_epoch) + 1, "minibatch" => ((iteration - 1) % len_epoch) + 1)
end

"""
$(TYPEDSIGNATURES)

Within an epoch: iterates the current minibatch index by one.
At the end of an epoch: obtains a new epoch of minibatches from the `Minibatcher` updates the epoch index by one, and minibatch index to one.
"""
function update_minibatch!(os::OS) where {OS <: ObservationSeries}
    index = get_current_minibatch_index(os)
    len_epoch = get_length_epoch(os)

    # take new batch
    if len_epoch >= index["minibatch"] + 1
        index["minibatch"] += 1 #update by 1
    else
        index["minibatch"] = 1 # set to 1
        index["epoch"] += 1 # update by 1
        minibatches = get_minibatches(os)
        minibatcher = get_minibatcher(os)
        new_epoch = create_new_epoch!(minibatcher, collect(1:length(get_observations(os)))) # create a new sweep of minibatches
        push!(minibatches, new_epoch)
    end
end

"""
$(TYPEDSIGNATURES)

get the minibatch for a given minibatch index (`Dict("epoch"=> x, "minibatch" => y)`), or iteration `Int`. If `nothing` is provided as an iteration then the current minibatch is returned
"""
function get_minibatch(os::OS, it_or_mbi::IorDorN) where {OS <: ObservationSeries, IorDorN <: Union{Int, Dict, Nothing}}
    if isnothing(it_or_mbi)
        return get_current_minibatch(os)
    else
        index = isa(it_or_mbi, Dict) ? it_or_mbi : get_minibatch_index(os, it_or_mbi)
        minibatches = get_minibatches(os)
        epoch = index["epoch"]
        mini = index["minibatch"]
        return minibatches[epoch][mini]
    end
end

# stored as vector of vectors
"""
$(TYPEDSIGNATURES)

get the current minibatch that is pointed to by the `current_minibatch_indices` field
"""
function get_current_minibatch(os::OS) where {OS <: ObservationSeries}
    minibatches = get_minibatches(os)
    epoch = get_current_minibatch_index(os)["epoch"]
    return minibatches[epoch][get_current_minibatch_index(os)["minibatch"]]
end

"""
$(TYPEDSIGNATURES)

if `build=true` then gets the observed sample, stacked over the minibatch at `iteration`. `build=false` lists the `samples` for all observations. If `isnothing(iteration)` or not defined then the current iteration is used.
"""
function get_obs(os::OS, iteration::IorN; build = true) where {OS <: ObservationSeries, IorN <: Union{Int, Nothing}}
    minibatch = get_minibatch(os, iteration)
    minibatch_length = length(minibatch)
    observations_vec = get_observations(os)[minibatch] # gives observation objects
    if minibatch_length == 1
        return get_obs(observations_vec[1], build = build)
    end

    if !build # return y as vec of vecs
        return get_obs.(observations_vec, build = false)
    else # stack y
        sample_lengths = [length(get_obs(ov, build = true)) for ov in observations_vec]
        minibatch_samples = zeros(sum(sample_lengths))
        for (i, observation) in enumerate(observations_vec)
            idx = (sum(sample_lengths[1:(i - 1)]) + 1):sum(sample_lengths[1:i])
            minibatch_samples[idx] = get_obs(observation, build = true)
        end
        return minibatch_samples
    end

end

"""
$(TYPEDSIGNATURES)

if `build=true` then gets the observed sample, stacked over the current minibatch. `build=false` lists the `samples` for all observations. 
"""
get_obs(os::OS; kwargs...) where {OS <: ObservationSeries} = get_obs(os, nothing; kwargs...)


"""
$(TYPEDSIGNATURES)

if `build=true` then gets the observation covariance matrix, blocked over the minibatch at `iteration`. `build=false` lists the `covs` for all observations. If `isnothing(iteration)` or not defined then the current iteration is used.
"""
function get_obs_noise_cov(
    os::OS,
    iteration::IorN;
    build = true,
) where {OS <: ObservationSeries, IorN <: Union{Int, Nothing}}
    minibatch = get_minibatch(os, iteration)
    minibatch_length = length(minibatch)
    observations_vec = get_observations(os)[minibatch] # gives observation objects
    if minibatch_length == 1 # if only 1 sample then return it
        return get_obs_noise_cov(observations_vec[1], build = build)
    else
        minibatch_covs = []
        for observation in observations_vec
            push!(minibatch_covs, get_obs_noise_cov(observation, build = build))
        end
    end
    if !build # return the blocks directly
        return reduce(vcat, minibatch_covs)
    else # return the blocked matrix
        block_sizes = size.(minibatch_covs, 1) # square mats
        minibatch_cov_full = zeros(sum(block_sizes), sum(block_sizes))
        idx_min = [0]
        for (i, mc) in enumerate(minibatch_covs)
            idx = (idx_min[1] + 1):(idx_min[1] + block_sizes[i])
            minibatch_cov_full[idx, idx] .= mc
            idx_min[1] += block_sizes[i]
        end

        return minibatch_cov_full
    end

end

"""
$(TYPEDSIGNATURES)

if `build=true` then gets the observation covariance matrix, blocked over the current minibatch. `build=false` lists the `covs` for all observations 
"""
get_obs_noise_cov(os::OS; kwargs...) where {OS <: ObservationSeries} = get_obs_noise_cov(os, nothing; kwargs...)


"""
$(TYPEDSIGNATURES)

if `build=true` then gets the inverse of the observation covariance matrix, blocked over minibatch at `iteration`. `build=false` lists the `inv_covs` for all observations. If `isnothing(iteration)` or not defined then the current iteration is used.
"""
function get_obs_noise_cov_inv(
    os::OS,
    iteration::IorN;
    build = true,
) where {OS <: ObservationSeries, IorN <: Union{Int, Nothing}}
    minibatch = get_minibatch(os, iteration)
    minibatch_length = length(minibatch)
    observations_vec = get_observations(os)[minibatch] # gives observation objects
    if minibatch_length == 1 # if only 1 sample then return it
        return get_obs_noise_cov_inv(observations_vec[1], build = build)
    else
        minibatch_inv_covs = []
        for observation in observations_vec
            push!(minibatch_inv_covs, get_obs_noise_cov_inv(observation, build = build)) # 
        end
    end
    if !build # return the blocks directly
        return reduce(vcat, minibatch_inv_covs)
    else # return the blocked matrix
        block_sizes = size.(minibatch_inv_covs, 1) # square mats
        minibatch_inv_cov_full = zeros(sum(block_sizes), sum(block_sizes))
        idx_min = [0]
        for (i, mc) in enumerate(minibatch_inv_covs)
            idx = (idx_min[1] + 1):(idx_min[1] + block_sizes[i])
            minibatch_inv_cov_full[idx, idx] .= mc
            idx_min[1] += block_sizes[i]
        end

        return minibatch_inv_cov_full
    end

end

"""
$(TYPEDSIGNATURES)

if `build=true` then gets the inverse of the observation covariance matrix, blocked over the current minibatch. `build=false` lists the `inv_covs` for all observations. 
"""
get_obs_noise_cov_inv(os::OS; kwargs...) where {OS <: ObservationSeries} = get_obs_noise_cov_inv(os, nothing; kwargs...)


get_cov_size(am::AM) where {AM <: AbstractMatrix} = size(am, 1)
function get_cov_size(am::SVD)
    if size(am.U, 1) == size(am.U, 2)
        return size(am.Vt, 2)
    else
        return size(am.U, 1)
    end
end

function lmul_cov!(out, a::D, X, idx1, idx2) where {D <: Diagonal}
    view(out, idx1, :) .= a.diag .* X[idx2, :]
end
function lmul_cov!(out, a::AM, X, idx1, idx2) where {AM <: AbstractMatrix}
    view(out, idx1, :) .= a * X[idx2, :]
end
function lmul_cov!(out, a::SVD, X, idx1, idx2)
    if size(a.U, 1) == size(a.U, 2) # then work with Vt
        view(out, idx1, :) .= a.Vt' * (a.S .* a.Vt) * X[idx2, :]
    else
        view(out, idx1, :) .= a.U * (a.S .* a.U') * X[idx2, :]
    end
end


function lmul_cov!(out, a::SpD, X, idx1, idx2) where {SpD <: SVDplusD}
    tmp = zeros(length(idx1), size(out, 2))

    a_diag = get_diag_cov(a)
    lmul_cov!(tmp, a_diag, X, 1:size(tmp, 1), idx2)
    view(out, idx1, :) .+= tmp

    a_svd = get_svd_cov(a)
    lmul_cov!(tmp, a_svd, X, 1:size(tmp, 1), idx2)
    view(out, idx1, :) .+= tmp

end

function lmul_cov!(out, a::DmT, X, idx1, idx2) where {DmT <: DminusTall}
    tmp = zeros(length(idx1), size(out, 2))

    a_diag = get_diag_cov(a)
    lmul_cov!(tmp, a_diag, X, 1:size(tmp, 1), idx2)
    view(out, idx1, :) .+= tmp

    a_tall = get_tall_cov(a)
    view(out, idx1, :) .-= a_tall * (a_tall' * X[idx2, :])

end


## Most common operation
function lmul_without_build(A, X::AVorM) where {AVorM <: AbstractVecOrMat}
    a_sizes = zeros(Int, length(A))
    for (i, a) in enumerate(A)
        a_sizes[i] = get_cov_size(a)
    end
    Y = zeros(sum(a_sizes), size(X, 2)) # stores A * X
    Xmat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    shift = [0]
    for (γs, a) in zip(a_sizes, A)
        idx = (shift[1] + 1):(shift[1] + γs)
        lmul_cov!(Y, a, Xmat, idx, idx)
        shift[1] = maximum(idx)
    end
    return Y
end

function lmul_cov!(out, a::D, X, global_idx1, local_idx, global_idx2) where {D <: Diagonal}
    view(out, global_idx1, :) .= a.diag[local_idx] .* X[global_idx2, :]
end
function lmul_cov!(out, a::AM, X, global_idx1, local_idx, global_idx2) where {AM <: AbstractMatrix}
    view(out, global_idx1, :) .= a[local_idx, local_idx] * X[global_idx2, :]
end
function lmul_cov!(out, a::SVD, X, global_idx1, local_idx, global_idx2)
    if size(a.U, 1) == size(a.U, 2) # then work with Vt
        view(out, global_idx1, :) .= a.Vt[:, local_idx]' * (a.S .* a.Vt[:, local_idx]) * X[global_idx2, :]
    else
        view(out, global_idx1, :) .= a.U[local_idx, :] * (a.S .* a.U[local_idx, :]') * X[global_idx2, :]
    end
end

function lmul_cov!(out, a::SpD, X, global_idx1, local_idx, global_idx2) where {SpD <: SVDplusD}
    tmp = zeros(length(global_idx1), size(out, 2))

    a_diag = get_diag_cov(a)
    lmul_cov!(tmp, a_diag, X, 1:size(tmp, 1), local_idx, global_idx2)
    view(out, global_idx1, :) .= tmp

    a_svd = get_svd_cov(a)
    lmul_cov!(tmp, a_svd, X, 1:size(tmp, 1), local_idx, global_idx2)
    view(out, global_idx1, :) .+= tmp

end

function lmul_cov!(out, a::DmT, X, global_idx1, local_idx, global_idx2) where {DmT <: DminusTall}
    tmp = zeros(length(global_idx1), size(out, 2))

    a_diag = get_diag_cov(a)
    lmul_cov!(tmp, a_diag, X, 1:size(tmp, 1), local_idx, global_idx2)
    view(out, global_idx1, :) .= tmp

    a_tall = get_tall_cov(a)
    view(out, global_idx1, :) .-= a_tall[local_idx, :] * (a_tall[local_idx, :]' * X[global_idx2, :])

end

function lmul_without_build!(out, A, X::AVorM, idx_triple::AV) where {AVorM <: AbstractVecOrMat, AV <: AbstractVector}
    Xmat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    for (block_idx, local_idx, global_idx) in idx_triple
        a = A[block_idx]
        lmul_cov!(out, a, Xmat, global_idx, local_idx, global_idx)
    end
end

function lmul_obs_noise_cov(os::ObservationSeries, X::AVorM) where {AVorM <: AbstractVecOrMat}
    A = get_obs_noise_cov(os, build = false)
    return lmul_without_build(A, X)
end

function lmul_obs_noise_cov_inv(os::ObservationSeries, X::AVorM) where {AVorM <: AbstractVecOrMat}
    A = get_obs_noise_cov_inv(os, build = false)
    return lmul_without_build(A, X)
end

function lmul_obs_noise_cov!(
    out,
    os::ObservationSeries,
    X::AVorM,
    idx_set::AV,
) where {AVorM <: AbstractVecOrMat, AV <: AbstractVector}
    A = get_obs_noise_cov(os, build = false)
    idx_triple = generate_block_product_subindices(A, idx_set)
    if isa(X, AbstractVector)
        Xtrim = length(X) > length(idx_set) ? X[idx_set] : X
    else
        Xtrim = size(X, 1) > length(idx_set) ? X[idx_set, :] : X
    end
    return lmul_without_build!(out, A, Xtrim, idx_triple)
end

function lmul_obs_noise_cov_inv!(
    out,
    os::ObservationSeries,
    X::AVorM,
    idx_set::AV,
) where {AVorM <: AbstractVecOrMat, AV <: AbstractVector}
    A = get_obs_noise_cov_inv(os, build = false)
    idx_triple = generate_block_product_subindices(A, idx_set)
    # trim X if not already, (idx-triple expects it trimmed)
    if isa(X, AbstractVector)
        Xtrim = length(X) > length(idx_set) ? X[idx_set] : X
    else
        Xtrim = size(X, 1) > length(idx_set) ? X[idx_set, :] : X
    end
    return lmul_without_build!(out, A, Xtrim, idx_triple)
end

function generate_block_product_subindices(Ablocks, idx_set)
    A_sizes = zeros(Int, length(Ablocks))
    for (i, Ai) in enumerate(Ablocks)
        A_sizes[i] = get_cov_size(Ai)
    end

    idx_triple = []
    shift = 0
    int_shift = 0
    for (block_id, As) in enumerate(A_sizes)
        loc_idx = intersect(1:As, idx_set .- shift)
        if !(length(loc_idx) == 0)
            push!(idx_triple, (block_id, loc_idx, collect(1:length(loc_idx)) .+ int_shift))
        end
        shift += As
        int_shift += length(loc_idx)
    end
    return idx_triple
end


function Base.:(==)(os_a::OS1, os_b::OS2) where {OS1 <: ObservationSeries, OS2 <: ObservationSeries}
    fn = unique([fieldnames(OS1)...; fieldnames(OS2)...])
    x = [false for f in fn]
    for (i, f) in enumerate(fn)
        x[i] = (getfield(os_a, Symbol(f)) == getfield(os_b, Symbol(f)))
    end
    return all(x)
end
