using Distributed
using NCDatasets
using Dates
using TimeZones
using DataFrames
using CSV
using StatsPlots

addprocs(64)

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

@everywhere const MAX_GAP_S   = 2     # max gap (sec) between consecutive samples within a block
@everywhere const MIN_BLOCK_S = 600   # min block duration (sec) to keep

@everywhere using NCDatasets, Dates, TimeZones


# Scans ARM lidar .cdf files for Houston (Oct 2021 – Oct 2022), converts UTC
# timestamps to local time (America/Chicago, handling CST/CDT), and extracts
# continuous measurement blocks (gaps < 2 s, duration > 600 s). Files are
# processed in parallel and the resulting blocks are written to time_lidar.csv
# with columns [day, month, year, time_start_s, duration_s].


@everywhere function from_datetime_to_sec(set_time, file_date::Date, tz)
    # Use millisecond precision throughout to avoid sub-second rounding
    time_ms = hour.(set_time) .* 3600000 .+
              minute.(set_time) .* 60000 .+
              second.(set_time) .* 1000 .+
              millisecond.(set_time)

    base_datetime = DateTime(file_date)
    total_time_dt_utc = base_datetime .+ Millisecond.(time_ms)

    zoned    = ZonedDateTime.(total_time_dt_utc, tz"UTC")
    dt_local = DateTime.(astimezone.(zoned, tz))

    true_year  = year.(dt_local)
    true_month = month.(dt_local)
    true_day   = day.(dt_local)

    true_time_sec = Dates.value.(dt_local .- DateTime.(true_year, true_month, true_day)) ./ 1000

    return true_time_sec, true_day, true_month, true_year
end


@everywhere function block_division(set_time)
    length(set_time) < 2 && return Tuple{Int,Int,Int}[]

    pause_ind = vcat(findall(abs.(diff(set_time)) .> MAX_GAP_S), length(set_time))

    if pause_ind[1] != 1
        index_in  = [1; pause_ind[1:end-1] .+ 1]
        index_out = pause_ind 
    else 
        index_in  = pause_ind[1:end-1] .+ 1
        index_out = pause_ind[2:end]
    end

    blocks = Tuple{Int,Int,Int}[]
    for i in eachindex(index_in)
        block_size = index_out[i] - index_in[i]
        if block_size > MIN_BLOCK_S
            push!(blocks, (index_in[i], index_out[i], block_size)) 
        end
    end
    return blocks
end


@everywhere function read_lidar_data_for_one_file(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    all_info = Vector{Any}[]
    time_in_datetime = dataset["time"][:]
    time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, file_date, tz)
    set_blocks_ind = block_division(time_in_sec)

    for ind_t in set_blocks_ind
        time_start = time_in_sec[ind_t[1]]
        time_range = ind_t[3]
        push!(all_info, [day[ind_t[1]], month[ind_t[1]], year[ind_t[1]], time_start, time_range]) 
    end
    
    close(dataset) 
    return all_info
end


@everywhere function process_file(PATH_folder_lidar, file, tz)
    file_parts = split(basename(file), ".")
    year  = parse(Int, file_parts[3][1:4])
    month = parse(Int, file_parts[3][5:6])
    day   = parse(Int, file_parts[3][7:8])
    println(year, " ", month, " ", day)
    
    file_date = Date(year, month, day) 
    start_date = Date(2021, 10, 1)
    end_date   = Date(2022, 10, 1)   
    if start_date <= file_date <= end_date
        PATH_file = joinpath(PATH_folder_lidar, file)
        info_one_file = read_lidar_data_for_one_file(PATH_file, file_date, tz)
        return info_one_file
    end
end


function main()
    tz = tz"America/Chicago"  # Houston 
    
    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_lidar))
    time_from_one_file = pmap(file -> process_file(PATH_folder_lidar, file, tz), set_files)
    time_from_one_file = filter(x -> x !== nothing, time_from_one_file)

    all_blocks = reduce(vcat, time_from_one_file)

    df = DataFrame(
        day          = Int[r[1] for r in all_blocks],
        month        = Int[r[2] for r in all_blocks],
        year         = Int[r[3] for r in all_blocks],
        time_start_s = Float64[r[4] for r in all_blocks],
        duration_s   = Int[r[5] for r in all_blocks],
    )

    CSV.write(joinpath(PATH_output_txt, "time_lidar.csv"), df)
end


function duration_histogram()
    df = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"), DataFrame)
    
    count_above_850 = count(>(850), df.duration_s)
    println("Number of duration_s > 850: ", count_above_850)
    plt = histogram(
        df.duration_s,
        bins = 10000,
        xlabel = "Duration, sec",
        ylabel = "Frequency",
        legend = false,
        xlims = (700, 850)
    )
    return plt
end 


main()
duration_histogram()