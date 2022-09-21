module DataContainers

## Imports
import Base: size #must import to add a definition to size

##Exports
export DataContainer, PairedDataContainer
export size
export get_data, get_inputs, get_outputs

## Objects
"""
    DataContainer{FT <: Real}

Container to store data samples as columns in an array.
"""
struct DataContainer{FT <: Real}
    #stored data, each piece of data is a column [data dimension × number samples]
    stored_data::AbstractMatrix{FT}
    #constructor with 2D arrays
    function DataContainer(stored_data::AbstractMatrix{FT}; data_are_columns = true) where {FT <: Real}
        if data_are_columns
            new{FT}(deepcopy(stored_data))
        else
            #Note: permutedims contains a deepcopy
            new{FT}(permutedims(stored_data, (2, 1)))
        end
    end
end

"""
    PairedDataContainer{FT <: Real}

stores input - output pairs as data containers, there must be an equal number of inputs and outputs
"""
struct PairedDataContainer{FT <: Real}
    # container for inputs and ouputs, each Container holds an array
    # size [data/parameter dimension × number samples]
    inputs::DataContainer{FT}
    outputs::DataContainer{FT}

    #constructor with 2D Arrays
    function PairedDataContainer(
        inputs::AbstractMatrix{FT},
        outputs::AbstractMatrix{FT};
        data_are_columns = true,
    ) where {FT <: Real}

        sample_dim = data_are_columns ? 2 : 1
        if !(size(inputs, sample_dim) == size(outputs, sample_dim))
            throw(
                DimensionMismatch(
                    "There must be the same number of samples of both inputs and outputs. Got $(size(inputs, sample_dim)) input samples and $(size(outputs, sample_dim)) output samples.",
                ),
            )
        end

        stored_inputs = DataContainer(inputs; data_are_columns = data_are_columns)
        stored_outputs = DataContainer(outputs; data_are_columns = data_are_columns)
        new{FT}(stored_inputs, stored_outputs)

    end
    #constructor with DataContainers
    function PairedDataContainer(inputs::DataContainer, outputs::DataContainer)

        if !(size(inputs, 2) == size(outputs, 2))
            throw(
                DimensionMismatch(
                    "There must be the same number of samples of both inputs and outputs. Got $(size(inputs, 2)) input samples and $(size(outputs, 2)) output samples.",
                ),
            )
        else
            FT = eltype(get_data(inputs))
            new{FT}(inputs, outputs)
        end
    end

end

## functions
"""
    size(dc::DataContainer, idx::IT) where {IT <: Integer}

Returns the size of the stored data. If `idx` provided, it returns the size along dimension `idx`.
"""
size(dc::DataContainer) = size(dc.stored_data)

size(dc::DataContainer, idx::IT) where {IT <: Integer} = size(dc.stored_data, idx)

size(pdc::PairedDataContainer) = size(pdc.inputs), size(pdc.outputs)

"""
    size(pdc::PairedDataContainer, idx::IT) where {IT <: Integer}

Returns the sizes of the inputs and ouputs along dimension `idx` (if provided)
"""
size(pdc::PairedDataContainer, idx::IT) where {IT <: Integer} = size(pdc.inputs, idx), size(pdc.outputs, idx)

"""
    get_data(pdc::PairedDataContainer)

Returns both input and output data stored in pdc
"""
get_data(dc::DataContainer) = deepcopy(dc.stored_data)
get_data(pdc::PairedDataContainer) = get_inputs(pdc), get_outputs(pdc)

"""
    get_inputs(pdc::PairedDataContainer)

Returns input data stored in pdc
"""
get_inputs(pdc::PairedDataContainer) = get_data(pdc.inputs)

"""
    get_outputs(pdc::PairedDataContainer)

Returns output data stored in pdc
"""
get_outputs(pdc::PairedDataContainer) = get_data(pdc.outputs)

end # module
