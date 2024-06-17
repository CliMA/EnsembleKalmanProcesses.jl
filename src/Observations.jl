using DocStringExtensions
using LinearAlgebra
using Statistics
using Random

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
    get_minibatches,
    get_current_minibatch_index,
    get_minibatcher,
    update_minibatch!,
    get_current_minibatch,
    no_minibatcher

# TODO: Define == and copy for these structs


"""
    Observation

Structure that contains a (possibly stacked) observation. Defined by sample(s), noise covariance(s), and name(s)

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

get_samples(o::Observation) = o.samples
get_covs(o::Observation) = o.covs
get_inv_covs(o::Observation) = o.inv_covs
get_names(o::Observation) = o.names
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
        T = promote_type((typeof(s) for s in samples)...)
        samples_tmp = [convert(T, s) for s in samples] # to re-infer eltype  (if the user makes an eltype of "Any")
        if !isa(samples_tmp, AbstractVector{<:AbstractVector}) # [1,2] -> [[1,2]] 
            snew = [samples]
        else
            snew = samples_tmp  # [[1,2,3]]
        end
    end
    if !isa(covariances, AbstractVector) # [2 1;1 2] -> [[2 1;1 2]]
        cnew = [covariances]
    else
        T = promote_type((typeof(c) for c in covariances)...)
        cnew = [convert(T, c) for c in covariances] # to re-infer eltype
    end

    if !("inv_covariances" ∈ collect(keys(obs_dict)))
        inv_covariances = []
        for c in cnew # ensures its a vector
            push!(inv_covariances, inv(c))
        end
    else
        inv_covariances = obs_dict["inv_covariances"]
    end
    if !isa(inv_covariances, AbstractVector) # [2 1;1 2] -> [[2 1;1 2]]
        icnew = [inv_covariances]
    else
        T = promote_type((typeof(c) for c in inv_covariances)...)
        icnew = [convert(T, c) for c in inv_covariances] # to re-infer eltype
    end

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

# combines observations into one long effective observation
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

function get_obs_noise_cov(o::Observation; build = true)

    if !build # return the blocks directly
        return get_covs(o)
    else # return the blocked matrix
        indices = get_indices(o)
        covs = get_covs(o)
        cov_full = zeros(maximum(indices[end]), maximum(indices[end]))
        for (idx, c) in zip(indices, covs)
            if isa(c, UniformScaling)
                for idxx in idx
                    cov_full[idxx, idxx] = c.λ
                end
            else
                cov_full[idx, idx] .= c
            end
        end

        return cov_full
    end
end

function get_obs_noise_cov_inv(o::Observation; build = true)

    if !build # return the blocks directly
        return get_inv_covs(o)
    else # return the blocked matrix
        indices = get_indices(o)
        inv_covs = get_inv_covs(o)
        inv_cov_full = zeros(maximum(indices[end]), maximum(indices[end]))
        for (idx, c) in zip(indices, inv_covs)
            if isa(c, UniformScaling)
                for idxx in idx
                    inv_cov_full[idxx, idxx] = c.λ
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

function no_minibatcher(size::Int = 1) #optional to provide size
    # method
    def_minibatch = [collect(1:size)]
    def_method = "order"
    def_rng = Random.default_rng()
    return FixedMinibatcher(def_minibatch, def_method, def_rng)
end


get_minibatches(m::FM) where {FM <: FixedMinibatcher} = m.minibatches
get_method(m::FM) where {FM <: FixedMinibatcher} = m.method
get_rng(m::FM) where {FM <: FixedMinibatcher} = m.rng
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

get_minibatch_size(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.minibatch_size
get_method(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.method
get_rng(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.rng
get_minibatches(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher} = m.minibatches

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

get_observations(os::ObservationSeries) = os.observations
get_minibatches(os::ObservationSeries) = os.minibatches
get_names(os::ObservationSeries) = os.names
get_current_minibatch_index(os::ObservationSeries) = os.current_minibatch_index
get_minibatcher(os::ObservationSeries) = os.minibatcher

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
function get_current_minibatch(os::OS) where {OS <: ObservationSeries}
    minibatches = get_minibatches(os)
    epoch = get_current_minibatch_index(os)["epoch"]
    return minibatches[epoch][get_current_minibatch_index(os)["minibatch"]]
end

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
        sample_length = length(get_obs(observations_vec[1], build = true))
        minibatch_samples = zeros(sample_length * minibatch_length)
        for (i, observation) in enumerate(observations_vec)
            idx = ((i - 1) * sample_length + 1):(i * sample_length)
            minibatch_samples[idx] = get_obs(observation, build = true)
        end
        return minibatch_samples
    end

end

function get_obs_noise_cov(os::OS; build = true) where {OS <: ObservationSeries}
    minibatch = get_current_minibatch(os) # gives the indices of the minibatch
    minibatch_length = length(minibatch)
    observations_vec = get_observations(os)[minibatch] # gives observation objects
    if minibatch_length == 1 # if only 1 sample then return it
        return get_obs_noise_cov(observations_vec[1], build = build)
    else
        minibatch_covs = []
        for observation in observations_vec
            push!(minibatch_covs, get_obs_noise_cov(observation, build = build)) # 
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




function Base.:(==)(os_a::OS1, os_b::OS2) where {OS1 <: ObservationSeries, OS2 <: ObservationSeries}
    fn = unique([fieldnames(OS1)...; fieldnames(OS2)...])
    x = [false for f in fn]
    for (i, f) in enumerate(fn)
        x[i] = (getfield(os_a, Symbol(f)) == getfield(os_b, Symbol(f)))
    end
    return all(x)
end
