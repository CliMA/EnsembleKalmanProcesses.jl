using DocStringExtensions
using LinearAlgebra
using Statistics
using Random
using TSVD

export Observation, Minibatcher, FixedMinibatcher, RandomFixedSizeMinibatcher, ObservationSeries
export get_samples,
    get_covs,
    get_inv_covs,
    get_names,
    get_indices,
    combine_observations,
    get_obs_noise_cov,
    get_obs_noise_cov_inv,
    get_obs,
    create_new_epoch!,
    get_minibatches,
    get_method,
    get_rng,
    get_minibatch_size,
    get_observations,
    get_current_minibatch_index,
    get_minibatcher,
    update_minibatch!,
    get_current_minibatch,
    no_minibatcher,
    tsvd_mat,
    tsvd_cov_from_samples,
    lmul_without_build

# wrapper for tsvd - putting solution in SVD object
"""
$(TYPEDSIGNATURES)

For a given matrix `X` and rank `r`, return the truncated SVD for X as a LinearAlgebra.jl `SVD` object. Setting `return_inverse=true` also return it's psuedoinverse X⁺.
"""
function tsvd_mat(X, r::Int; return_inverse = false, tsvd_kwargs...)
    # Note, must only use tsvd approximation when rank < minimum dimension of X or you get very poor approximation.
    if isa(X, UniformScaling)
        if return_inverse
            return svd(X(r)), svd(inv(X)(r))
        else
            return svd(X(r))
        end
    else
        rx = rank(X)
        mindim = minimum(size(X))
        if rx <= r
            if rx < r
                @warn(
                    "Requested truncation to rank $(r) for an input matrix of rank $(rx). Performing (truncated) SVD for rank $(rx) matrix."
                )
            end

            if rx < mindim
                U, s, V = tsvd(X, rx; tsvd_kwargs...)
            else # perform exact svd (do NOT use tsvd for this! very poor approximation)
                SS = svd(X)
                if return_inverse
                    return SS, SVD(permutedims(SS.Vt, (2, 1)), 1.0 ./ SS.S, permutedims(SS.U, (2, 1)))
                else
                    return SS
                end
            end
        else
            U, s, V = tsvd(X, r; tsvd_kwargs...)
        end
        if return_inverse
            return SVD(U, s, permutedims(V, (2, 1))), SVD(V, 1.0 ./ s, permutedims(U, (2, 1)))
        else
            return SVD(U, s, permutedims(V, (2, 1)))
        end
    end
end

function tsvd_mat(X; return_inverse = false, tsvd_kwargs...)
    if isa(X, UniformScaling)
        throw(
            ArgumentError(
                "Cannot perform a low rank decomposition on `UniformScaling` type without providing a rank in the second argument.",
            ),
        )
    end
    return tsvd_mat(X, rank(X); return_inverse = return_inverse, tsvd_kwargs...)
end

"""
$(TYPEDSIGNATURES)

For a given `sample_mat`, (with `data_are_columns = true`), rank "r" is optionally provided. Returns the SVDs corresponding to the matrix `cov(sample_mat; dims=2)`. Efficient representation when size(sample_mat,1) << size(sample_mat,2). Setting `return_inverse=true` also returns its psuedoinverse.
"""
function tsvd_cov_from_samples(
    sample_mat::AM,
    r::Int;
    data_are_columns::Bool = true,
    return_inverse = false,
    tsvd_kwargs...,
) where {AM <: AbstractMatrix}

    mat = data_are_columns ? sample_mat : permutedims(sample_mat, (2, 1))
    N = size(mat, 2)
    if N > size(mat, 1)
        @warn(
            "SVD representation is efficient when estimating high-dimensional covariance with few samples. \n here # samples is $(N), while the space dimension is $(size(mat,1)), and representation will be inefficient."
        )
    elseif N == 1
        throw(ArgumentError("Require multiple samples to estimate covariance matrix, only 1 provided"))
    end
    debiased_scaled_mat = 1.0 / sqrt(N - 1) * (mat .- mean(mat, dims = 2))
    rk = min(rank(debiased_scaled_mat), r)

    if return_inverse
        A, Ainv = tsvd_mat(debiased_scaled_mat, rk; return_inverse = return_inverse, tsvd_kwargs...)
        # A now represents the sqrt of the covariance, so we square the singular values 
        return SVD(A.U, A.S .^ 2, permutedims(A.U, (2, 1))), SVD(permutedims(Ainv.Vt, (2, 1)), Ainv.S .^ 2, Ainv.Vt)
    else
        A = tsvd_mat(debiased_scaled_mat, rk; return_inverse = return_inverse, tsvd_kwargs...)
        # A now represents the sqrt of the covariance, so we square the singular values 
        return SVD(A.U, A.S .^ 2, permutedims(A.U, (2, 1)))
    end
end

function tsvd_cov_from_samples(
    sample_mat::AM;
    data_are_columns::Bool = true,
    return_inverse = false,
    tsvd_kwargs...,
) where {AM <: AbstractMatrix}
    # (need to compute this to get the rank as debiasing can change it)
    mat = data_are_columns ? sample_mat : permutedims(sample_mat, (2, 1))
    rk = rank(mat .- mean(mat, dims = 2))

    return tsvd_cov_from_samples(
        sample_mat,
        rk;
        data_are_columns = data_are_columns,
        return_inverse = return_inverse,
        tsvd_kwargs...,
    )
end

# TODO: Define == and copy for these structs
"""
    Observation

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
One can stack up multiple observations with combine_observations, or by providing vectors of samples, covariances and names to the dictionary.

# Fields

$(TYPEDFIELDS)
"""
struct Observation{
    AV1 <: AbstractVector,
    AV2 <: AbstractVector,
    AV3 <: AbstractVector,
    AV4 <: AbstractVector,
    AV5 <: AbstractVector,
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

function Observation(obs_dict::Dict)
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
            if isa(c, SVD)
                push!(inv_covariances, SVD(permutedims(c.Vt, (2, 1)), 1.0 ./ c.S, permutedims(c.U, (2, 1))))
            else
                push!(inv_covariances, inv(c))
            end
        end
    else
        inv_covariances = obs_dict["inv_covariances"]
    end
    if !isa(inv_covariances, AbstractVector) # [2 1;1 2] -> [[2 1;1 2]]
        ictmp = [inv_covariances]
    else
        ictmp = inv_covariances
    end
    # additionally provide a dimension for UniformScalings
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

    return Observation(snew, cnew, icnew, nnew, indices)

end

function Observation(
    sample::AV,
    obs_noise_cov::AMorUS,
    name::AS,
) where {AV <: AbstractVector, AMorUS <: Union{AbstractMatrix, UniformScaling}, AS <: AbstractString}
    return Observation(Dict("samples" => sample, "covariances" => obs_noise_cov, "names" => name))
end

function Observation(
    samples::AV1,
    obs_noise_covs::AV2,
    names::AV3,
) where {AV1 <: AbstractVector, AV2 <: AbstractVector, AV3 <: AbstractVector}
    return Observation(Dict("samples" => samples, "covariances" => obs_noise_covs, "names" => names))
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

    return Observation(snew2, cnew2, icnew2, nnew2, inew2)
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
            if isa(c, SVD)
                if size(c.U, 1) == size(c.U, 2) # then work with Vt
                    cov_full[idx, idx] .= c.Vt' * Diagonal(c.S) * c.Vt
                else
                    cov_full[idx, idx] .= c.U * Diagonal(c.S) * c.U'
                end
            else
                cov_full[idx, idx] .= c
            end
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
            if isa(c, SVD)
                if size(c.U, 1) == size(c.U, 2) # then work with Vt
                    inv_cov_full[idx, idx] .= c.Vt' * Diagonal(c.S) * c.Vt
                else
                    inv_cov_full[idx, idx] .= c.U * Diagonal(c.S) * c.U'
                end
            else
                inv_cov_full[idx, idx] .= c
            end
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
    FixedMinibatcher <: Minibatcher

A `Minibatcher` that takes in a given epoch of batches. It creates a new epoch by either copying-in-order, or by shuffling, the provided batches.

# Fields

$(TYPEDFIELDS)
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
    RandomFixedSizeMinibatcher <: Minibatcher

A `Minibatcher` that takes in a given epoch of batches. It creates a new epoch by either copying-in-order, or by shuffling, the provided batches.

# Fields

$(TYPEDFIELDS)
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
    ObservationSeries

Structure that contains multiple `Observation`s along with an optional `Minibatcher`. Stores all observations in `EnsembleKalmanProcess`, as well as defining the behavior of the `get_obs`, `get_obs_noise_cov`, and `get_obs_noise_cov_inv` methods

Typical Constructor
```
ObservationSeries(
    Dict(
        "observations" => vec_of_observations,
        "names" => names_of_observations,
        "minibatcher" => minibatcher
    ),
)
```
# Fields

$(TYPEDFIELDS)
"""
struct ObservationSeries{AV1 <: AbstractVector, MM <: Minibatcher, AV2 <: AbstractVector, AV3 <: AbstractVector}
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

function ObservationSeries(
    obs_vec_in::AV,
    minibatcher::MM,
    names_in::AV2,
    epoch::AV3,
) where {AV <: AbstractVector, MM <: Minibatcher, AV2 <: AbstractVector, AV3 <: AbstractVector}
    T = promote_type((typeof(o) for o in obs_vec_in)...)
    obs_vec = [convert(T, o) for o in obs_vec_in]

    T = promote_type((typeof(n) for n in names_in)...)
    names = [convert(T, n) for n in names_in]

    minibatches = [create_new_epoch!(minibatcher, epoch)]
    current_minibatch_index = Dict("epoch" => 1, "minibatch" => 1)
    return ObservationSeries(obs_vec, minibatcher, names, current_minibatch_index, minibatches)
end

function ObservationSeries(
    obs_vec::AV,
    minibatcher::MM,
    epoch_or_names::AV2,
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

    return ObservationSeries(obs_vec, minibatcher, names, epoch)

end

function ObservationSeries(obs_vec::AV, minibatcher::MM) where {AV <: AbstractVector, MM <: Minibatcher}
    names = ["series_$(string(i))" for i in 1:length(obs_vec)]
    epoch = collect(1:length(obs_vec))
    return ObservationSeries(obs_vec, minibatcher, names, epoch)
end

function ObservationSeries(obs_vec::AV) where {AV <: AbstractVector}
    len_epoch = length(obs_vec)
    minibatcher = no_minibatcher(len_epoch)
    names = ["series_$(string(i))" for i in 1:len_epoch]
    epoch = collect(1:len_epoch)
    return ObservationSeries(obs_vec, minibatcher, names, epoch)
end

function ObservationSeries(obs::O, args...; kwargs...) where {O <: Observation}
    return ObservationSeries([obs], no_minibatcher(), args...; kwargs...)
end

function ObservationSeries(obs_series_dict::Dict)
    if !("observations" ∈ collect(keys(obs_series_dict)))
        throw(ArgumentError("input dictionaries must contain the key: \"observations\". Got $(keys(obs_series_dict))"))
    end
    # call different constructors
    if issetequal(["observations"], collect(keys(obs_series_dict)))
        return ObservationSeries(obs_series_dict["observations"])
    elseif issetequal(["observations", "minibatcher"], collect(keys(obs_series_dict)))
        return ObservationSeries(obs_series_dict["observations"], obs_series_dict["minibatcher"])
    elseif issetequal(["observations", "minibatcher", "epoch"], collect(keys(obs_series_dict)))
        return ObservationSeries(
            obs_series_dict["observations"],
            obs_series_dict["minibatcher"],
            obs_series_dict["epoch"],
        )
    elseif issetequal(["observations", "minibatcher", "names"], collect(keys(obs_series_dict)))
        return ObservationSeries(
            obs_series_dict["observations"],
            obs_series_dict["minibatcher"],
            obs_series_dict["names"],
        )
    elseif issetequal(["observations", "minibatcher", "epoch", "names"], collect(keys(obs_series_dict)))
        return ObservationSeries(
            obs_series_dict["observations"],
            obs_series_dict["minibatcher"],
            obs_series_dict["names"],
            obs_series_dict["epoch"],
        )
    else
        throw(
            ArgumentError(
                "input dictionaries must contain a subset of keys: [\"observations\", \"minibatcher\", \"names\", \"epoch\"]. Got $(keys(obs_series_dict))",
            ),
        )
    end
end

"""
$(TYPEDSIGNATURES)

Within an epoch: iterates the current minibatch index by one.
At the end of an epoch: obtains a new epoch of minibatches from the `Minibatcher` updates the epoch index by one, and minibatch index to one.
"""
function update_minibatch!(os::OS) where {OS <: ObservationSeries}
    index = get_current_minibatch_index(os)
    minibatches_in_epoch = get_minibatches(os)[index["epoch"]]

    # take new batch
    if length(minibatches_in_epoch) >= index["minibatch"] + 1
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

if `build=true` then gets the observed sample, stacked over the current minibatch. `build=false` lists the `samples` for all observations 
"""
function get_obs(os::OS; build = true) where {OS <: ObservationSeries}
    minibatch = get_current_minibatch(os) # gives the indices of the minibatch
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

if `build=true` then gets the observation covariance matrix, blocked over the current minibatch. `build=false` lists the `covs` for all observations 
"""
function get_obs_noise_cov(os::OS; build = true) where {OS <: ObservationSeries}
    minibatch = get_current_minibatch(os) # gives the indices of the minibatch
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

if `build=true` then gets the inverse of the observation covariance matrix, blocked over the current minibatch. `build=false` lists the `inv_covs` for all observations 
"""
function get_obs_noise_cov_inv(os::OS; build = true) where {OS <: ObservationSeries}
    minibatch = get_current_minibatch(os) # gives the indices of the minibatch
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

## Most common operation
function lmul_without_build(A, X::AVorM) where {AVorM <: AbstractVecOrMat}
    a_sizes = zeros(Int, length(A))
    for (i, a) in enumerate(A)
        if isa(a, SVD)
            if size(a.U, 1) == size(a.U, 2)
                a_sizes[i] = size(a.Vt, 2)
            else
                a_sizes[i] = size(a.U, 1)
            end
        else
            a_sizes[i] = size(a, 1)
        end
    end
    Y = zeros(sum(a_sizes), size(X, 2)) # stores A * X
    Xmat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    shift = [0]
    for (γs, a) in zip(a_sizes, A)
        idx = (shift[1] + 1):(shift[1] + γs)
        if isa(a, Diagonal)
            Y[idx, :] = a.diag .* Xmat[idx, :]
        elseif isa(a, SVD)
            if size(a.U, 1) == size(a.U, 2) # then work with Vt
                Y[idx, :] = a.Vt[:, idx]' * (a.S .* a.Vt[:, idx]) * Xmat[idx, :]
            else
                Y[idx, :] = a.U[idx, :] * (a.S .* a.U[idx, :]') * Xmat[idx, :]
            end
        else
            Y[idx, :] = a * Xmat[idx, :]
        end
        shift[1] = maximum(idx)
    end
    return Y
end

function lmul_without_build!(out, A, X::AVorM, idx_triple::AV) where {AVorM <: AbstractVecOrMat, AV <: AbstractVector}
    Xmat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    for (block_idx, local_idx, global_idx) in idx_triple
        a = A[block_idx]
        if isa(a, Diagonal)
            out[global_idx, :] = a.diag[local_idx] .* Xmat[global_idx, :]
        elseif isa(a, SVD)
            if size(a.U, 1) == size(a.U, 2) # then work with Vt
                out[global_idx, :] = a.Vt[:, local_idx]' * (a.S .* a.Vt[:, local_idx]) * Xmat[global_idx, :]
            else
                out[global_idx, :] = a.U[local_idx, :] * (a.S .* a.U[local_idx, :]') * Xmat[global_idx, :]
            end
        else # assume general matrix, much slower
            out[global_idx, :] = a[local_idx, local_idx] * Xmat[global_idx, :]
        end
    end
end

function lmul_sqrt_without_build(A, X::AVorM) where {AVorM <: AbstractVecOrMat}
    a_sizes = zeros(Int, length(A))
    for (i, a) in enumerate(A)
        if isa(a, SVD)
            if size(a.U, 1) == size(a.U, 2)
                a_sizes[i] = size(a.Vt, 2)
            else
                a_sizes[i] = size(a.U, 1)
            end
        else
            a_sizes[i] = size(a, 1)
        end
    end
    Y = zeros(sum(a_sizes), size(X, 2)) # stores A * X
    Xmat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    shift = [0]
    for (γs, a) in zip(a_sizes, A)
        idx = (shift[1] + 1):(shift[1] + γs)
        if isa(a, Diagonal)
            Y[idx, :] = sqrt.(a.diag) .* Xmat[idx, :]
        elseif isa(a, SVD)
            if size(a.U, 1) == size(a.U, 2) # then work with Vt
                Y[idx, :] = a.Vt[:, idx]' * (sqrt.(a.S) .* a.Vt[:, idx]) * Xmat[idx, :]
            else
                Y[idx, :] = a.U[idx, :] * (sqrt.(a.S) .* a.U[idx, :]') * Xmat[idx, :]
            end
        else # assume general matrix, much slower
            Y[idx, :] = sqrt(a) * Xmat[idx, :]
        end
        shift[1] = maximum(idx)
    end
    return Y
end

function lmul_sqrt_without_build!(out, A, X::AVorM, idx_triple::AV) where {AVorM <: AbstractVecOrMat, AV <: AbstractVector}
    Xmat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    for (block_idx, local_idx, global_idx) in idx_triple
        a = A[block_idx]
        if isa(a, Diagonal)
            out[global_idx, :] = sqrt.(a.diag[local_idx]) .* Xmat[global_idx, :]
        elseif isa(a, SVD)
            if size(a.U, 1) == size(a.U, 2) # then work with Vt
                out[global_idx, :] = a.Vt[:, local_idx]' * (sqrt.(a.S) .* a.Vt[:, local_idx]) * Xmat[global_idx, :]
            else
                out[global_idx, :] = a.U[local_idx, :] * (sqrt.(a.S) .* a.U[local_idx, :]') * Xmat[global_idx, :]
            end
        else # assume general matrix, much slower
            out[global_idx, :] = sqrt(a[local_idx, local_idx]) * Xmat[global_idx, :]
        end
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


function lmul_sqrt_obs_noise_cov!(
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
    return lmul_sqrt_without_build!(out, A, Xtrim, idx_triple)
end

function lmul_sqrt_obs_noise_cov_inv!(
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
    return lmul_sqrt_without_build!(out, A, Xtrim, idx_triple)
end



function generate_block_product_subindices(Ablocks, idx_set)
    A_sizes = zeros(Int, length(Ablocks))
    for (i, Ai) in enumerate(Ablocks)
        if isa(Ai, SVD)
            if size(Ai.U, 1) == size(Ai.U, 2)
                A_sizes[i] = size(Ai.Vt, 2)
            else
                A_sizes[i] = size(Ai.U, 1)
            end
        else
            A_sizes[i] = size(Ai, 1)
        end
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
