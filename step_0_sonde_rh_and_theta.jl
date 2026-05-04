using NCDatasets
using Dates
using TimeZones
using Statistics
using DataFrames
using CSV
using Plots
using Measures
using Interpolations
using Distributed
using DelimitedFiles

addprocs(64)

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

@everywhere using NCDatasets, Dates, TimeZones, Statistics, Interpolations, DataFrames, CSV, DelimitedFiles
@everywhere include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Sonde processing parameters
@everywhere const H_MAX_SONDE = 4000           # m, max sonde height kept
@everywhere const BIN_WIDTH_H = 10 / 60        # h, time-bin width for daily averaging
@everywhere const TIME_GAP_H  = 1              # h, max time difference for sonde<->lidar match
@everywhere const TIME_GAP_S  = TIME_GAP_H * 3600

# Builds RH and potential-temperature profiles aligned to the lidar grid.
#   collect_files_from_sonde_all : reads houinterpolatedsondeM1, masks heights
#                                  to <= H_MAX_SONDE, averages each day in
#                                  BIN_WIDTH_H bins, writes sonde_rh_and_theta.csv.
#   align_data_with_lidar        : matches each lidar block timestamp to the
#                                  nearest sonde profile within TIME_GAP_H,
#                                  linearly interpolates RH and theta onto the
#                                  lidar height grid, writes
#                                  sonde_rh_and_theta_aligned_with_lidar.csv.
#
# NOTE on RH variable choice (per Petters et al. 2024 and ARM handbook DOE/SC-ARM-TR-183):
#   "rh"        - sonde-only relative humidity, interpolated between launches
#                 (with sondeadjust corrections for old Vaisala bias).
#   "rh_scaled" - rh additionally rescaled to match microwave radiometer PWV.
#   This script uses "rh" (matches Petters 2024).

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

@everywhere function read_sonde_data_PROFILE(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")
    all_info = []

    try
        time_in_datetime = dataset["time"][:]
        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        time_in_h = time_in_sec ./ 3600   # bin in hours, but write seconds at the end

        height_m = dataset["height"][:] .* 1000  # km -> m
        mask_height = height_m .<= H_MAX_SONDE
        height_m = height_m[mask_height]

        rh_data = dataset["rh"][mask_height, :]
        t_data  = dataset["potential_temp"][mask_height, :]

        date_keys = [(year[t], month[t], day[t]) for t in 1:length(time_in_h)]
        unique_dates = unique(date_keys)

        for d in unique_dates
            indices = findall(==(d), date_keys)

            t_subset    = time_in_h[indices]
            rh_subset   = rh_data[:, indices]
            temp_subset = t_data[:, indices]

            min_time = BIN_WIDTH_H * floor(Int, minimum(t_subset) / BIN_WIDTH_H)
            max_time = BIN_WIDTH_H * ceil(Int,  maximum(t_subset) / BIN_WIDTH_H)
            edges = collect(min_time:BIN_WIDTH_H:max_time)

            bin_indices = [searchsortedlast(edges, t) for t in t_subset]
            n_bins = maximum(bin_indices)

            binned_time = Float64[]
            binned_rh   = Array{Float64}(undef, length(height_m), n_bins)
            binned_temp = Array{Float64}(undef, length(height_m), n_bins)

            for i in 1:n_bins
                idxs = findall(==(i), bin_indices)
                if !isempty(idxs)
                    push!(binned_time, mean(t_subset[idxs]))
                    binned_rh[:, i]   = mapslices(x -> mean(skipmissing(x)), rh_subset[:, idxs];   dims=2)[:]
                    binned_temp[:, i] = mapslices(x -> mean(skipmissing(x)), temp_subset[:, idxs]; dims=2)[:]
                else
                    push!(binned_time, edges[i] + BIN_WIDTH_H / 2)
                    binned_rh[:, i]   .= NaN
                    binned_temp[:, i] .= NaN
                end
            end

            for h in 1:length(height_m)
                for t in 1:length(binned_time)
                    # Convert bin time back to seconds for canonical storage
                    bin_time_s = binned_time[t] * 3600
                    push!(all_info,
                        [d[3], d[2], d[1], bin_time_s, height_m[h], binned_rh[h, t], binned_temp[h, t]])
                end
            end
        end
    finally
        close(dataset)
    end

    return all_info
end


function collect_files_from_sonde_all()
    tz = tz"America/Chicago"  # Houston

    info_sonde_all_days = []
    set_files = filter(x -> endswith(x, ".nc"), readdir(PATH_folder_sonde))
    for file in set_files
        println(file)
        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])
        println("day ", day, " month ", month, " year ", year)

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_sonde, file)
        info_one_file = read_sonde_data_PROFILE(file_new, file_date, tz)
        append!(info_sonde_all_days, info_one_file)
    end

    println(size(info_sonde_all_days))

    df = DataFrame(
        day      = Int[r[1] for r in info_sonde_all_days],
        month    = Int[r[2] for r in info_sonde_all_days],
        year     = Int[r[3] for r in info_sonde_all_days],
        time_s   = Float64[r[4] for r in info_sonde_all_days],   # canonical: seconds
        height_m = Float64[r[5] for r in info_sonde_all_days],
        rh       = [r[6] for r in info_sonde_all_days],
        theta    = [r[7] for r in info_sonde_all_days],
    )
    output_file = joinpath(PATH_output_txt, "sonde_rh_and_theta.csv")
    CSV.write(output_file, df)
    println(output_file)
end


function align_data_with_lidar()
    input_data_height = readdlm(joinpath(PATH_output_txt, "height_lidar.txt"))[:, 1]
    input_data_time   = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"), DataFrame)
    input_data_rh     = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta.csv"), DataFrame)

    time_day_keys = [(input_data_time[i,:day], input_data_time[i,:month], input_data_time[i,:year]) for i in 1:nrow(input_data_time)]
    unique_days = unique(time_day_keys)
    interpolated_rh = []

    for day in unique_days
        day_lidar, m_lidar, y_lidar = day

        idxs_time = findall(x -> x[1]==day_lidar && x[2]==m_lidar && x[3]==y_lidar, time_day_keys)
        lidar_times = input_data_time[idxs_time, :time_start_s]   # seconds

        mask_day = (input_data_rh[!,:day]   .== day_lidar) .&
                   (input_data_rh[!,:month] .== m_lidar)   .&
                   (input_data_rh[!,:year]  .== y_lidar)

        rh_times    = input_data_rh[mask_day, :time_s]    # already seconds
        rh_heights  = input_data_rh[mask_day, :height_m]
        rh_values   = input_data_rh[mask_day, :rh]
        temp_values = input_data_rh[mask_day, :theta]

        if isempty(rh_times)
            for t_lidar in lidar_times
                for h in input_data_height
                    push!(interpolated_rh, [day_lidar, m_lidar, y_lidar, t_lidar, h, NaN, NaN])
                end
            end
            continue
        end

        for t_lidar in lidar_times
            time_diffs = abs.(rh_times .- t_lidar)
            if minimum(time_diffs) <= TIME_GAP_S
                idx_closest = argmin(time_diffs)
                t_rh = rh_times[idx_closest]

                # findall is needed here: pulls all heights at this matched timestamp
                idx_match = findall(rh_times .== t_rh)
                h_set1    = rh_heights[idx_match]
                rh_set1   = rh_values[idx_match]
                temp_set1 = temp_values[idx_match]

                # Filter out NaN values before interpolating to prevent NaN propagation.
                # rh and theta may have independent NaN locations.
                rh_valid   = .!isnan.(rh_set1)
                temp_valid = .!isnan.(temp_set1)

                if sum(rh_valid) >= 2
                    h_rh = h_set1[rh_valid]
                    sort_rh = sortperm(h_rh)
                    interp_rh = LinearInterpolation(h_rh[sort_rh], rh_set1[rh_valid][sort_rh],
                                                    extrapolation_bc = NaN)
                else
                    interp_rh = nothing
                end

                if sum(temp_valid) >= 2
                    h_t = h_set1[temp_valid]
                    sort_t = sortperm(h_t)
                    interp_t = LinearInterpolation(h_t[sort_t], temp_set1[temp_valid][sort_t],
                                                   extrapolation_bc = NaN)
                else
                    interp_t = nothing
                end

                for h in input_data_height
                    rh_val   = interp_rh === nothing  ? NaN : interp_rh(h)
                    temp_val = interp_t  === nothing  ? NaN : interp_t(h)
                    push!(interpolated_rh, [day_lidar, m_lidar, y_lidar, t_lidar, h, rh_val, temp_val])
                end
            else
                for h in input_data_height
                    push!(interpolated_rh, [day_lidar, m_lidar, y_lidar, t_lidar, h, NaN, NaN])
                end
            end
        end
    end

    df = DataFrame(
        day      = Int[r[1] for r in interpolated_rh],
        month    = Int[r[2] for r in interpolated_rh],
        year     = Int[r[3] for r in interpolated_rh],
        time_s   = Float64[r[4] for r in interpolated_rh],
        height_m = Float64[r[5] for r in interpolated_rh],
        rh       = Float64[r[6] for r in interpolated_rh],
        theta    = Float64[r[7] for r in interpolated_rh],
    )
    output_path = joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv")
    CSV.write(output_path, df)
    println(output_path)
end


collect_files_from_sonde_all()
align_data_with_lidar()