using DataFrames
using CSV
using Plots
using Measures
using LaTeXStrings

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

const PLOT_TIME_LO_S  = 12 * 3600   # s
const PLOT_TIME_HI_S  = 15 * 3600   # s
const TIME_MATCH_S    = 0.5 * 3600  # s, ±0.5 h tolerance for sonde<->PBL match


function group_by_day(df::DataFrame)
    groups = Dict{Tuple{Int,Int,Int}, Vector{Int}}()
    for i in 1:nrow(df)
        key = (df[i,:day], df[i,:month], df[i,:year])
        push!(get!(groups, key, Int[]), i)
    end
    return groups
end


function process_profile(day, month, year, time, info_rh, info_pbl;
                         pbl_value_col::Symbol)
    if time < PLOT_TIME_LO_S || time > PLOT_TIME_HI_S
        return nothing
    end

    rh_day, rh_month, rh_year, rh_time = info_rh[!,:day], info_rh[!,:month], info_rh[!,:year], info_rh[!,:time_s]
    rh_height = info_rh[!,:height_m]
    rh_val    = info_rh[!,:theta]

    pbl_day, pbl_month, pbl_year = info_pbl[!,:day], info_pbl[!,:month], info_pbl[!,:year]
    pbl_time = info_pbl[!,:time_s]
    pbl_val  = info_pbl[!, pbl_value_col]

    mask_rh  = (rh_day  .== day) .& (rh_month  .== month) .& (rh_year  .== year) .& (abs.(rh_time  .- time) .<= TIME_MATCH_S)
    mask_pbl = (pbl_day .== day) .& (pbl_month .== month) .& (pbl_year .== year) .& (abs.(pbl_time .- time) .<= TIME_MATCH_S)

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


function plot_normalized_profiles_methods_by_season()
    info_rh   = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"),    DataFrame)
    info_thl  = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_temp_inversion.csv"),    DataFrame)
    info_filt = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"),          DataFrame)

    combos = unique([
        (info_rh[i,:day], info_rh[i,:month], info_rh[i,:year], info_rh[i,:time_s])
        for i in 1:nrow(info_rh)
    ])

    println("Processing ", length(combos), " profiles per method...")

    function run_method(info_pbl; pbl_value_col)
        out = map(c -> begin
            res = process_profile(c[1], c[2], c[3], c[4], info_rh, info_pbl;
                                  pbl_value_col = pbl_value_col)
            isnothing(res) ? nothing : (month = c[2], result = res)
        end, combos)
        return filter(!isnothing, out)
    end

    results_thl  = run_method(info_thl;  pbl_value_col = :pbl_h)
    results_filt = run_method(info_filt; pbl_value_col = :pbl_filtered)

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
    season_title = Dict(
        "Winter" => L"\mathrm{Winter}",
        "Spring" => L"\mathrm{Spring}",
        "Summer" => L"\mathrm{Summer}",
        "Fall"   => L"\mathrm{Fall}",
    )

    method_label = Dict(
        :thl  => L"z/PBL_{\theta}",
        :filt => L"z/PBL_{filtered}",
    )

    method_results = Dict(
        :thl  => results_thl,
        :filt => results_filt,
    )

    plots = []

    for method in (:thl, :filt)
        for season in season_order
            title_str = method == :thl ? season_title[season] : ""

            plt = plot(
                xlabel = L"$\theta/\theta_0\,(-)$",
                ylabel = method_label[method],
                title  = title_str,
                legend = false,
                guidefontsize = 20,
                tickfontsize  = 16,
                titlefontsize = 19,
                grid = false,
                ylims = (0, 1.5), xlims = (0.96, 1.06),
            )

            season_results = [
                r.result for r in method_results[method]
                if season_from_month(r.month) == season
            ]

            for (t, z_norm) in season_results
                scatter!(
                    plt,
                    t ./ t[1],
                    z_norm,
                    alpha = 0.01,
                    markersize = 2,
                    color = :blue,
                )
            end

            push!(plots, plt)
        end
    end

    panel = plot(
        plots...,
        layout = (2, 4),
        size = (1800, 900),
        left_margin = 10mm,
        bottom_margin = 15mm,
    )

    output_file = joinpath(PATH_output_png, "normalized_temp_profiles_methods_by_season.png")
    savefig(panel, output_file)
    println(output_file)

    return panel
end


plot_normalized_profiles_methods_by_season()