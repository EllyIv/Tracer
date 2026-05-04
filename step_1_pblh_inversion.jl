using DelimitedFiles
using Statistics
using DataFrames
using CSV
using Distributed
using Plots
using DSP   # for smoothing kernel in the alternate inversion detection
using Measures
using LaTeXStrings

addprocs(32)

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

@everywhere using DelimitedFiles, Statistics, DataFrames, CSV, DSP, LaTeXStrings, Measures
@everywhere include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Inversion-detection parameters
@everywhere const GRAD_THRESHOLD = 0.015              # K/m, gradient above which a level is flagged as inversion
@everywhere const TIME_TOL_S     = 0.5 * 3600         # s, profile time matching tolerance (= 0.5 h)
@everywhere const SMOOTH_WINDOW  = 5                  # points, smoothing window for the alternate method
@everywhere const PLOT_TIME_LO_S = 12 * 3600          # s, daytime window for the diagnostic plot
@everywhere const PLOT_TIME_HI_S = 15 * 3600


# Detects the planetary-boundary-layer top from the potential-temperature
# profile (column theta in sonde_rh_and_theta_aligned_with_lidar.csv),
# defined as the lowest height where the vertical gradient dT/dz first
# exceeds GRAD_THRESHOLD. Profiles within +/- TIME_TOL_S of each lidar
# timestamp are pooled and averaged at each height before the gradient
# is taken.
#
# Note: Petters et al. (2024) used PBL height from the NAM model. This
# script uses a sonde theta-inversion approach instead. The two will
# differ; downstream code should treat this PBL as a sonde-based
# diagnostic, not a NAM-equivalent.
#
#   main                              : runs the gradient-threshold
#                                       detection in parallel over every
#                                       unique profile timestamp; writes
#                                       pbl_aligned_with_lidar_temp_inversion.csv.
#   check_inversion                   : original method (first crossing of
#                                       the raw gradient).
#   check_inversion_smoothed          : alternate method using a smoothed
#                                       profile and the maximum gradient
#                                       location instead of the first
#                                       crossing. Same signature so it can
#                                       be swapped into main.
#   plot_normalized_profiles_parallel         : diagnostic plot of normalized
#                                               temperature profiles
#                                               (z / PBL_h) for
#                                               PLOT_TIME_LO_S <= time <=
#                                               PLOT_TIME_HI_S.
#   plot_normalized_profiles_parallel_by_season : seasonal version of above.



@everywhere function check_inversion(fixed_day, fixed_month, fixed_year, fixed_time, info_rh)
    rh_day    = info_rh[!, :day]
    rh_month  = info_rh[!, :month]
    rh_year   = info_rh[!, :year]
    rh_time   = info_rh[!, :time_s]
    rh_height = info_rh[!, :height_m]

    # potential temperature, not RH
    rh_val    = info_rh[!, :theta]

    mask_time = (rh_month .== fixed_month) .&
                (rh_year  .== fixed_year)  .&
                (rh_day   .== fixed_day)   .&
                (abs.(rh_time .- fixed_time) .<= TIME_TOL_S)

    rh_height = rh_height[mask_time]
    rh_val    = rh_val[mask_time]
    mask_valid = .!ismissing.(rh_val) .& .!isnan.(rh_val)
    rh_height = rh_height[mask_valid]
    rh_val    = rh_val[mask_valid]
    if isempty(rh_height)
        return missing
    end

    idx = sortperm(rh_height)
    rh_height = rh_height[idx]
    rh_val    = rh_val[idx]
    unique_h = sort(unique(rh_height))
    ave_h = Float64[]
    ave_t = Float64[]
    for h in unique_h
        vals = rh_val[rh_height .== h]
        push!(ave_h, h)
        push!(ave_t, mean(vals))
    end
    if length(ave_h) < 2
        return missing
    end
    grad_T = diff(ave_t) ./ diff(ave_h)
    inv_idx = findfirst(grad_T .> GRAD_THRESHOLD)
    inversion_height = isnothing(inv_idx) ? missing : ave_h[inv_idx]
    println(inversion_height)
    return inversion_height
end


# Alternate inversion detection: smooth the profile with a moving average and
# pick the height of the maximum gradient (rather than the first crossing).
# Less sensitive to noise spikes; more robust on finely-interpolated profiles.
@everywhere function check_inversion_smoothed(fixed_day, fixed_month, fixed_year, fixed_time, info_rh)
    rh_day    = info_rh[!, :day]
    rh_month  = info_rh[!, :month]
    rh_year   = info_rh[!, :year]
    rh_time   = info_rh[!, :time_s]
    rh_height = info_rh[!, :height_m]
    rh_val    = info_rh[!, :theta]

    mask_time = (rh_month .== fixed_month) .&
                (rh_year  .== fixed_year)  .&
                (rh_day   .== fixed_day)   .&
                (abs.(rh_time .- fixed_time) .<= TIME_TOL_S)

    rh_height = rh_height[mask_time]
    rh_val    = rh_val[mask_time]
    mask_valid = .!ismissing.(rh_val) .& .!isnan.(rh_val)
    rh_height = rh_height[mask_valid]
    rh_val    = rh_val[mask_valid]
    if isempty(rh_height)
        return missing
    end

    idx = sortperm(rh_height)
    rh_height = rh_height[idx]
    rh_val    = rh_val[idx]
    unique_h = sort(unique(rh_height))
    ave_h = Float64[]
    ave_t = Float64[]
    for h in unique_h
        vals = rh_val[rh_height .== h]
        push!(ave_h, h)
        push!(ave_t, mean(vals))
    end
    if length(ave_h) < SMOOTH_WINDOW + 1
        return missing
    end

    # Boxcar smoothing of the profile, then take the gradient and pick the maximum
    kernel = ones(SMOOTH_WINDOW) ./ SMOOTH_WINDOW
    ave_t_smooth = conv(ave_t, kernel)[(SMOOTH_WINDOW÷2 + 1):(end - SMOOTH_WINDOW÷2)]
    L = min(length(ave_t_smooth), length(ave_h))
    ave_t_smooth = ave_t_smooth[1:L]
    ave_h_use    = ave_h[1:L]

    grad_T = diff(ave_t_smooth) ./ diff(ave_h_use)
    if all(grad_T .<= GRAD_THRESHOLD)
        return missing
    end
    max_idx = argmax(grad_T)
    inversion_height = ave_h_use[max_idx]
    println(inversion_height, " (smoothed)")
    return inversion_height
end


function main()
    info_rh = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"), DataFrame)

    rh_day, rh_month, rh_year, rh_time = info_rh[!,:day], info_rh[!,:month], info_rh[!,:year], info_rh[!,:time_s]
    combos = unique([(rh_day[i], rh_month[i], rh_year[i], rh_time[i]) for i in eachindex(rh_day)])
    println("Total unique combinations: ", length(combos))

    @everywhere const INFO_RH_GLOBAL = $info_rh

    results = pmap(combos) do (day, month, year, time)
        pbl = check_inversion(day, month, year, time, INFO_RH_GLOBAL)
        [day, month, year, time, pbl]
    end

    df = DataFrame(
        day    = Int[r[1] for r in results],
        month  = Int[r[2] for r in results],
        year   = Int[r[3] for r in results],
        time_s = Float64[r[4] for r in results],
        pbl_h  = [r[5] for r in results],
    )
    output_file = joinpath(PATH_output_txt, "pbl_aligned_with_lidar_temp_inversion.csv")
    CSV.write(output_file, df)
    println(output_file)
end


@everywhere function process_profile(day, month, year, time, info_rh, info_pbl)
    if time < PLOT_TIME_LO_S || time > PLOT_TIME_HI_S
        return nothing
    end

    rh_day, rh_month, rh_year, rh_time = info_rh[!,:day], info_rh[!,:month], info_rh[!,:year], info_rh[!,:time_s]
    rh_height = info_rh[!,:height_m]
    rh_val    = info_rh[!,:theta]

    pbl_day, pbl_month, pbl_year, pbl_time =
        info_pbl[!,:day], info_pbl[!,:month], info_pbl[!,:year], info_pbl[!,:time_s]
    pbl_val = info_pbl[!,:pbl_h]

    mask_rh  = (rh_day  .== day) .& (rh_month  .== month) .& (rh_year  .== year) .& (rh_time  .== time)
    mask_pbl = (pbl_day .== day) .& (pbl_month .== month) .& (pbl_year .== year) .& (pbl_time .== time)

    if !any(mask_pbl)
        return nothing
    end

    pbl = pbl_val[mask_pbl][1]
    if ismissing(pbl) || pbl == 0
        return nothing
    end

    h = rh_height[mask_rh]
    t = rh_val[mask_rh]
    valid = .!ismissing.(t) .& .!isnan.(t)
    h = h[valid]
    t = t[valid]

    isempty(h) && return nothing

    z_norm = h ./ pbl
    return (t, z_norm)
end


function plot_normalized_profiles_parallel()
    info_rh  = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"), DataFrame)
    info_pbl = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_temp_inversion.csv"), DataFrame)

    combos = unique([(info_rh[i,:day], info_rh[i,:month], info_rh[i,:year], info_rh[i,:time_s]) for i in 1:nrow(info_rh)])

    println("Processing ", length(combos), " profiles in parallel...")

    results = pmap(c -> process_profile(c[1], c[2], c[3], c[4], info_rh, info_pbl), combos)
    results = filter(!isnothing, results)

    plt = plot(xlabel="Temperature (K)", ylabel="z / PBL height", legend=false,
               ylims=(0, 2), color=:blue, alpha=0.01, markersize=2)

    for (t, z_norm) in results
        scatter!(plt, t, z_norm, alpha=0.01)
    end

    savefig(plt, joinpath(PATH_output_png, "norm_temp_profiles_12to15_based_on_inv.png"))
    println(joinpath(PATH_output_png, "norm_temp_profiles_12to15_based_on_inv.png"))

    all_t = vcat([r[1] for r in results]...)
    all_z = vcat([r[2] for r in results]...)
    df_out = DataFrame(theta = all_t, z_norm = all_z)
    output_file = joinpath(PATH_output_txt, "norm_temp_profiles_12to15_based_on_inv.csv")
    CSV.write(output_file, df_out)
    println(output_file)
end


function plot_normalized_profiles_parallel_by_season()
    info_rh  = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"), DataFrame)
    info_pbl = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_temp_inversion.csv"), DataFrame)

    combos = unique([
        (info_rh[i,:day], info_rh[i,:month], info_rh[i,:year], info_rh[i,:time_s])
        for i in 1:nrow(info_rh)
    ])

    println("Processing ", length(combos), " profiles in parallel...")

    results = pmap(c -> begin
        out = process_profile(c[1], c[2], c[3], c[4], info_rh, info_pbl)
        isnothing(out) ? nothing : (month = c[2], result = out)
    end, combos)

    results = filter(!isnothing, results)

    function season_from_month(m)
        if m in (12, 1, 2)
            return "Winter"
        elseif m in (3, 4, 5)
            return "Spring"
        elseif m in (6, 7, 8)
            return "Summer"
        else
            return "Fall"
        end
    end

    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_title = Dict("Winter" => L"\mathrm{Winter}", "Spring" => L"\mathrm{Spring}", "Summer" => L"\mathrm{Summer}", "Fall" => L"\mathrm{Fall}")

    plots = []

    for season in season_order
        plt = plot(
            xlabel = L"$\theta/\theta_0\,(-)$",
            ylabel = L"$z/z_i\, (-)$",
            title = season_title[season],
            legend = false,
            guidefontsize = 20,
            tickfontsize = 16,
            titlefontsize = 19,
            grid = false,
            ylims = (0, 1.5), xlims = (0.96, 1.06),
        )

        season_results = [
            r.result for r in results
            if season_from_month(r.month) == season
        ]

        for (t, z_norm) in season_results
            scatter!(
                plt,
                t ./ t[1],
                z_norm,
                alpha = 0.01,
                markersize = 2,
                color = :blue
            )
        end

        push!(plots, plt)
    end

    panel = plot(plots..., layout = (1, 4), size = (1800, 450), left_margin = 10mm, bottom_margin = 15mm)

    savefig(panel, joinpath(PATH_output_png, "norm_temp_profiles_12to15_based_on_inv_by_season.png"))
    println(joinpath(PATH_output_png, "norm_temp_profiles_12to15_based_on_inv_by_season.png"))

    return panel
end


main()
plot_normalized_profiles_parallel()
plot_normalized_profiles_parallel_by_season()