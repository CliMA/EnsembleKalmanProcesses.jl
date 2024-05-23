using JLD2, LinearAlgebra, Statistics
using CairoMakie, ColorSchemes
using EnsembleKalmanProcesses.DataContainers


function main()
output_directory = "output"

u_iters = JLD2.load(joinpath(output_directory,"parameter_storage.jld2"))["u_stored"]
g_iters = JLD2.load(joinpath(output_directory,"output_storage.jld2"))["g_stored"]

iters = 1:5
for iter in iters
    U = get_data(u_iters[iter])
    G = get_data(g_iters[iter])
    
    Cuu = cov(U,dims=2)
    Cgg = cov(G,dims=2)
    Um = mean(U,dims=2)
    Gm = mean(G,dims=2)
    Cug = 1/(size(G,2)-1)*((U .- Um)*(G .- Gm)')' # so u are rows
    
    
    fig = Figure(size=(3*450,450)) # size is backwards to what you expect
    
    auu = Axis(fig[1,1], title="Cuu")
    aug = Axis(fig[1,2], title="Cug")
    agg = Axis(fig[1,3], title="Cgg")
    
    heatmap!(auu,Cuu)
    heatmap!(aug,Cug)
    heatmap!(agg,Cgg)
    
    current_figure()
    
    save(joinpath(output_directory, "covs_iter_$iter.png"), fig, px_per_unit = 3)
end
end

main()
