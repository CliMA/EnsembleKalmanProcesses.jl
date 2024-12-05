
struct ProcessingWindow{FT <: AbstractFloat, SS <: AbstractString}
    "start time of processing window"
    t_start::FT
    "end time of processing window"
    t_end::FT
    "subwindow size"
    sw_size::Int
    "sliding subwindow, or batching subwindow"
    slide_or_batch::SS
end

function process_trajectory_to_data(pw::ProcessingWindow, xn, t)

    # Define averaging indices range - relative to trajectory start time
    indices = findall(x -> (x > pw.t_start) && (x < pw.t_end), t .- minimum(t))

    # Define the subwindows' starts and ends, and the number of them
    if pw.slide_or_batch == "batch"
        n_subwindow = Int64(floor(length(indices) / pw.sw_size))
        sw_start = 1:(pw.sw_size):(n_subwindow * pw.sw_size - 1)
        sw_end = (pw.sw_size):(pw.sw_size):(n_subwindow * pw.sw_size)
    elseif pw.slide_or_batch == "slide"
        n_subwindow = length(indices) - pw.sw_size
        sw_start = 1:n_subwindow
        sw_end = (pw.sw_size):(n_subwindow + pw.sw_size)
    else
        throw(
            ArgumentError,
            "ProcessingWindow.slide_or_batch must be \"slide\" or \"batch\", received $pw.slide_or_batch",
        )
    end

    # calculate first and second moments over the subwindows
    N = Int64(size(xn, 1) / 2)
    slow_id = 1:N
    fast_id = (N + 1):(2 * N)

    slow_mean_sw = [mean(vcat(xn[slow_id, indices[sw_start[i]:sw_end[i]]])) for i in 1:n_subwindow]
    fast_mean_sw = [mean(vcat(xn[fast_id, indices[sw_start[i]:sw_end[i]]])) for i in 1:n_subwindow]

    slow_meansq_sw = [mean(vcat(xn[slow_id, indices[sw_start[i]:sw_end[i]]] .^ 2)) for i in 1:n_subwindow]
    fast_meansq_sw = [mean(vcat(xn[fast_id, indices[sw_start[i]:sw_end[i]]] .^ 2)) for i in 1:n_subwindow]

    slowfast_mean_sw = [
        mean(vcat(xn[fast_id, indices[sw_start[i]:sw_end[i]]] .* xn[slow_id, indices[sw_start[i]:sw_end[i]]])) for
        i in 1:n_subwindow
    ]

    # Combine (<X>, <Y>, <X^2>, <Y^2>, <XY>)
    return vcat(slow_mean_sw..., fast_mean_sw..., slow_meansq_sw..., fast_meansq_sw..., slowfast_mean_sw...)
end
