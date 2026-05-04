using NCDatasets
using Dates
using TimeZones
using Statistics
using Interpolations
using DataFrames
using CSV
using DelimitedFiles
using Distributed
using Plots
using LaTeXStrings

addprocs(16)
# atexit(() -> rmprocs(workers()))

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

@everywhere using NCDatasets, Dates, TimeZones, Statistics, Interpolations, DataFrames, CSV, DelimitedFiles
@everywhere include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Sonde wind-profile parameters
@everywhere const MAX_H_SONDE  = 4000     # m, max sonde height kept
@everywhere const BIN_SIZE_S   = 900      # s, time-bin width for daily averaging (15 min)


# Builds wind-speed vertical profiles from the interpolated radiosonde
# product (houinterpolatedsondeM1) and projects them onto the lidar height
# grid.
#   main             : reads each sonde file, masks heights to <= MAX_H_SONDE,
#                      averages each day in BIN_SIZE_S bins, writes
#                      wind_profile.csv. Then calls extrapolated_data which
#                      linearly interpolates each profile onto the lidar
#                      height grid (NaN outside the sonde range) and writes
#                      wind_profile_interp.csv.
#
# NOTE on time storage:
#   time_s = bin start time in seconds since local midnight (Float64).
#   This is the canonical format used across the pipeline. Step 1 (flux_calc)
#   currently looks up wind by hour-of-day; that lookup will need to be
#   updated to use time_s when this file is regenerated.



@everywhere function from_datetime_to_sec(set_time, tz)
    zoned    = ZonedDateTime.(set_time, tz"UTC")
    dt_local = DateTime.(astimezone.(zoned, tz))

    true_year  = year.(dt_local)
    true_month = month.(dt_local)
    true_day   = day.(dt_local)

    midnight_local = DateTime.(true_year, true_month, true_day)
    true_time_sec  = Dates.value.(dt_local .- midnight_local) ./ 1000

    return true_time_sec, true_day, true_month, true_year
end

@everywhere function read_sonde_data_for_one_file(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    all_info = []
    try
        time_in_datetime = dataset["time"][:]
        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        height = dataset["height"] * 1000   # km -> m
        valid_j = findall(h -> h <= MAX_H_SONDE, height)
        wind_data = dataset["wspd"][valid_j, :]
        height = height[valid_j]

        min_time = floor(Int, minimum(time_in_sec))
        max_time = ceil(Int, maximum(time_in_sec))
        bins = min_time:BIN_SIZE_S:max_time

        for k in 1:length(bins)-1
            t_start = bins[k]
            t_end   = bins[k+1]

            inds = findall(t -> t_start <= t < t_end, time_in_sec)

            if !isempty(inds)
                for j in 1:length(height)
                    values = wind_data[j, inds]
                    if !all(ismissing, values)
                        avg = mean(skipmissing(values))
                        # bin start time in seconds since local midnight
                        push!(all_info, [day[inds[1]], month[inds[1]], year[inds[1]],
                                         Float64(t_start), height[j], avg])
                    end
                end
            end
        end
    finally
        close(dataset)
    end

    return all_info
end


@everywhere function extrapolated_data(info_sonde_all_days)
    new_height = readdlm(joinpath(PATH_output_txt, "height_lidar.txt"))[:, 1]

    day_w    = [x[1] for x in info_sonde_all_days]
    month_w  = [x[2] for x in info_sonde_all_days]
    year_w   = [x[3] for x in info_sonde_all_days]
    time_w   = [x[4] for x in info_sonde_all_days]   # seconds since midnight
    height_w = [x[5] for x in info_sonde_all_days]
    wind_w   = [x[6] for x in info_sonde_all_days]

    unique_times = unique([(day_w[i], month_w[i], year_w[i], time_w[i]) for i in eachindex(day_w)])

    new_wind = []

    for (d, m, y, t) in unique_times
        mask = (day_w .== d) .& (month_w .== m) .& (year_w .== y) .& (time_w .== t)
        heights = Float64.(height_w[mask])
        winds   = Float64.(wind_w[mask])

        # Filter out NaN values before interpolating to prevent NaN propagation
        valid = .!isnan.(winds)
        if sum(valid) < 2
            push!(new_wind, fill(NaN, length(new_height)))
            continue
        end
        heights = heights[valid]
        winds   = winds[valid]

        sort_idx = sortperm(heights)
        heights_sorted = heights[sort_idx]
        winds_sorted   = winds[sort_idx]

        # Linear interpolation onto the lidar height grid; NaN outside the sonde range
        interp_func = LinearInterpolation(heights_sorted, winds_sorted, extrapolation_bc=NaN)
        interpolated = [interp_func(h_val) for h_val in new_height]

        push!(new_wind, interpolated)
    end

    output = []
    for (i, (d, m, y, t)) in enumerate(unique_times)
        winds_interp = new_wind[i]
        for (j, height_val) in enumerate(new_height)
            wind_val = winds_interp[j]
            push!(output, [d, m, y, t, height_val, wind_val])
        end
    end

    df = DataFrame(
        day      = Int[r[1] for r in output],
        month    = Int[r[2] for r in output],
        year     = Int[r[3] for r in output],
        time_s   = Float64[r[4] for r in output],
        height_m = Float64[r[5] for r in output],
        wind     = Float64[r[6] for r in output],
    )
    output_path = joinpath(PATH_output_txt, "wind_profile_interp.csv")
    CSV.write(output_path, df)
    println(output_path)
end


@everywhere function process_sonde_file(file::String, tz)
    file_parts = split(basename(file), ".")
    year  = parse(Int, file_parts[3][1:4])
    month = parse(Int, file_parts[3][5:6])
    day   = parse(Int, file_parts[3][7:8])
    println("Processing day ", day, " month ", month, " year ", year)

    file_date = Date(year, month, day)
    file_path = joinpath(PATH_folder_sonde, file)
    return read_sonde_data_for_one_file(file_path, file_date, tz)
end


function main()
    tz = tz"America/Chicago"  # Houston

    set_files = filter(x -> endswith(x, ".nc"), readdir(PATH_folder_sonde))

    results = pmap(file -> process_sonde_file(file, tz), set_files)

    info_sonde_all_days = reduce(vcat, results)

    df = DataFrame(
        day      = Int[r[1] for r in info_sonde_all_days],
        month    = Int[r[2] for r in info_sonde_all_days],
        year     = Int[r[3] for r in info_sonde_all_days],
        time_s   = Float64[r[4] for r in info_sonde_all_days],
        height_m = Float64[r[5] for r in info_sonde_all_days],
        wind     = [r[6] for r in info_sonde_all_days],
    )
    output_file = joinpath(PATH_output_txt, "wind_profile.csv")
    CSV.write(output_file, df)

    extrapolated_data(info_sonde_all_days)
end


main()