# Wrapping up data

To provide a consistent form for data (such as observations, parameter ensembles, model evaluations) across the package, we store the data in simple wrappers internally.

Data is always stored as columns of `AbstractMatrix`. That is, we obey the format
```
[ data dimension x number of data samples ]
```

## The `DataContainer`

A `DataContainer` is constructed initially by copying and perhaps transposing matrix data
```julia
dc = DataContainer(abstract_matrix; data_are_columns = true)
```
!!! note
    Providing an n-vector will be interpreted as a [1xn] matrix

The flag `data_are_columns` indicates whether the provided data is stored column- or row-wise. The data is retrieved with
```julia
get_data(dc)
```

## The `PairedDataContainer`

A `PairedDataContainer` stores pairs of inputs and outputs in the form of `DataContainer`s. It is constructed from two data matrices, or from two `DataContainer`s.
```julia
pdc = PairedDataContainer(input_matrix, output_matrix; data_are_columns = true)
pdc = PairedDataContainer(input_data_container, output_data_container)
```
!!! note
    Providing inputs and outputs of different `eltype`s will lead to conversion of all data into the `eltype`: `T = promote_type(eltype(in_mat), eltype(out_mat))`

Data is retrieved with
```julia
get_data(pdc) # returns both inputs and outputs
get_inputs(pdc)
get_outputs(pdc)
```
