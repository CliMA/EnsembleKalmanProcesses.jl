module DataContainers

using DocStringExtensions
## Imports
import Base: size #must import to add a definition to size

##Exports
export DataContainer, PairedDataContainer
export size
export get_data, get_inputs, get_outputs

## Objects
"""
$(TYPEDEF)

Container to store data samples as columns in an array.

# Fields

$(TYPEDFIELDS)

# Constructors
    DataContainer(data::AVorM; data_are_columns = true) where {AVorM <: AbstractVecOrMat}

"""
struct DataContainer{FT <: Real}
    "stored data, each piece of data is a column [data dimension × number samples]"
    data::AbstractMatrix{FT}
    #constructor with 2D arrays
    function DataContainer(data::AVorM; data_are_columns = true) where {AVorM <: AbstractVecOrMat}
        if isa(data, AbstractVector)
            @warn "input to DataContainer is vector and has ambiguous shape, continuing with interpretation as a matrix of size (1,$(length(data))). \n use `reshape(data, 1, :)`  (n samples of 1d data) or `reshape(data, :, 1)` (1 sample of n-d data) to set preferred orientation directly."
            dd = reshape(data, 1, :)
        else
            dd = data
        end
        FT = eltype(dd)
        if data_are_columns
            new{FT}(deepcopy(dd))
        else
            #Note: permutedims contains a deepcopy
            new{FT}(permutedims(dd, (2, 1)))
        end
    end
end

"""
$(TYPEDEF)

Stores input - output pairs as data containers, there must be an equal number of inputs and outputs.

# Fields

$(TYPEDFIELDS)

# Constructors
    PairedDataContainer(
        inputs::AVorM1,
        outputs::AVorM2;
        data_are_columns = true,
    ) where {AVorM1 <: AbstractVecOrMat, AVorM2 <: AbstractVecOrMat}

    PairedDataContainer(inputs::DataContainer, outputs::DataContainer)

"""
struct PairedDataContainer{FT <: Real}
    "container for inputs, each Container holds an array size [data/parameter dimension × number samples]"
    inputs::DataContainer{FT}
    "container for ouputs, each Container holds an array size [data/parameter dimension × number samples]"
    outputs::DataContainer{FT}

    #constructor with 2D Arrays
    function PairedDataContainer(
        inputs::AVorM1,
        outputs::AVorM2;
        data_are_columns = true,
    ) where {AVorM1 <: AbstractVecOrMat, AVorM2 <: AbstractVecOrMat}

        if isa(inputs, AbstractVector)
            @warn "input to DataContainer is vector and has ambiguous shape, continuing with interpretation as a matrix of size (1,$(length(inputs))). \n use `reshape(inputs, 1, :)`  (n samples of 1d inputs) or `reshape(inputs, :, 1)` (1 sample of n-d inputs) to set preferred orientation directly."
            in = reshape(inputs, 1, :)
        else
            in = inputs
        end
        if isa(outputs, AbstractVector)
            @warn "input to DataContainer is vector and has ambiguous shape, continuing with interpretation as a matrix of size (1,$(length(outputs))). \n use `reshape(outputs, 1, :)`  (n samples of 1d outputs) or `reshape(outputs, :, 1)` (1 sample of n-d outputs) to set preferred orientation directly."
            out = reshape(outputs, 1, :)
        else
            out = outputs
        end

        sample_dim = data_are_columns ? 2 : 1
        if !(size(in, sample_dim) == size(out, sample_dim))
            throw(
                DimensionMismatch(
                    "There must be the same number of samples of both inputs and outputs. Got $(size(in, sample_dim)) input samples and $(size(out, sample_dim)) output samples.",
                ),
            )
        end

        FT = promote_type(eltype(in), eltype(out))
        if !(FT == eltype(in)) || !(FT == eltype(out))
            @warn "inputs and outputs provided to PairedDataContainer have different types ($(eltype(in)), $(eltype(out))), Storing both in mutual type: $(FT). "
        end
        stored_inputs = DataContainer(convert(Matrix{FT}, in); data_are_columns = data_are_columns)
        stored_outputs = DataContainer(convert(Matrix{FT}, out); data_are_columns = data_are_columns)

        new{FT}(stored_inputs, stored_outputs)

    end

    # constructor with DataContainers
    function PairedDataContainer(inputs::DataContainer, outputs::DataContainer)

        if !(size(inputs, 2) == size(outputs, 2))
            throw(
                DimensionMismatch(
                    "There must be the same number of samples of both inputs and outputs. Got $(size(inputs, 2)) input samples and $(size(outputs, 2)) output samples.",
                ),
            )
        else

            FT = promote_type(eltype(get_data(inputs)), eltype(get_data(outputs)))
            if !(FT == eltype(get_data(inputs))) || !(FT == eltype(get_data(outputs)))
                @warn "inputs and outputs provided to PairedDataContainer have different types ($(eltype(get_data(inputs))), $(eltype(get_data(outputs)))), Storing both in mutual type: $(FT). "
                new_inputs = DataContainer(convert(Matrix{FT}, get_data(inputs)))
                new_outputs = DataContainer(convert(Matrix{FT}, get_data(outputs)))

                return new{FT}(new_inputs, new_outputs)
            end
            return new{FT}(inputs, outputs)
        end
    end

end



## Functions

"""
    size(dc::DataContainer, idx::IT) where {IT <: Integer}

Returns the size of the stored data. If `idx` provided, it returns the size along dimension `idx`.
"""
size(dc::DataContainer) = size(dc.data)

size(dc::DataContainer, idx::IT) where {IT <: Integer} = size(dc.data, idx)

size(pdc::PairedDataContainer) = size(pdc.inputs), size(pdc.outputs)

"""
    size(pdc::PairedDataContainer, idx::IT) where {IT <: Integer}

Returns the sizes of the inputs and ouputs along dimension `idx` (if provided).
"""
size(pdc::PairedDataContainer, idx::IT) where {IT <: Integer} = size(pdc.inputs, idx), size(pdc.outputs, idx)

"""
    get_data(pdc::PairedDataContainer)

Returns both input and output data stored in `pdc` as two matrices.
"""
get_data(dc::DataContainer) = deepcopy(dc.data)
get_data(pdc::PairedDataContainer) = get_inputs(pdc), get_outputs(pdc)

"""
    get_inputs(pdc::PairedDataContainer)

Returns input data stored in `pdc`.
"""
get_inputs(pdc::PairedDataContainer) = get_data(pdc.inputs)

"""
    get_outputs(pdc::PairedDataContainer)

Returns output data stored in `pdc`.
"""
get_outputs(pdc::PairedDataContainer) = get_data(pdc.outputs)

# Override ==
Base.:(==)(dc_a::DataContainer, dc_b::DataContainer) = get_data(dc_a) == get_data(dc_b)
Base.:(==)(pdc_a::PairedDataContainer, pdc_b::PairedDataContainer) =
    get_inputs(pdc_a) == get_inputs(pdc_b) && get_outputs(pdc_a) == get_outputs(pdc_b)



end # module
