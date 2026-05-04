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


# Builds the height-normalized RH/theta record (z/PBL) on the lidar grid.
# Reads sonde_rh_and_theta_aligned_with_lidar.csv and divides each row's
# height by the corresponding PBL height from pbl_aligned_with_lidar_filtered.csv.
# The lookup uses an exact match on (day, month, year, time_s) since both
# files are aligned to the same lidar grid. Output:
# rh_norm_aligned_with_lidar.csv.



function norm_profiles()
    info_rh  = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"), DataFrame)
    pbl_info = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"),       DataFrame)

    # Build PBL lookup keyed on (day, month, year, time_s) for fast exact-match lookup.
    # Both files are aligned to the same lidar grid, so exact equality is the right test.
    pbl_lookup = Dict{Tuple{Int,Int,Int,Float64}, Union{Missing, Float64}}()
    for i in 1:nrow(pbl_info)
        key = (pbl_info.day[i], pbl_info.month[i], pbl_info.year[i], Float64(pbl_info.time_s[i]))
        pbl_lookup[key] = pbl_info.pbl_filtered[i]
    end

    day_rh    = info_rh[!,:day]
    month_rh  = info_rh[!,:month]
    year_rh   = info_rh[!,:year]
    time_rh   = info_rh[!,:time_s]
    height_rh = info_rh[!,:height_m]
    rh_val    = info_rh[!,:rh]
    temp_val  = info_rh[!,:theta]

    n = length(time_rh)
    info_pbl_height = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        key = (Int(day_rh[i]), Int(month_rh[i]), Int(year_rh[i]), Float64(time_rh[i]))
        info_pbl_height[i] = get(pbl_lookup, key, missing)
    end

    # Normalize height by PBL (missing/zero PBL -> missing height_n)
    height_n = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        pbl = info_pbl_height[i]
        if !ismissing(pbl) && pbl > 0
            height_n[i] = Float64(height_rh[i]) / Float64(pbl)
        end
    end

    df = DataFrame(
        day      = day_rh,
        month    = month_rh,
        year     = year_rh,
        time_s   = time_rh,
        height_n = height_n,
        rh       = rh_val,
        theta    = temp_val,
    )
    output_file_raw = joinpath(PATH_output_txt, "rh_norm_aligned_with_lidar.csv")
    CSV.write(output_file_raw, df)
    println(output_file_raw)
end


norm_profiles()