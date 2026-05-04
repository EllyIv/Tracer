using CSV
using Plots
using Colors
using Dates
using Statistics
using DataFrames
using Distributed
using Measures

addprocs(64)

@everywhere using DataFrames, CSV, Statistics, Dates

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")
@everywhere include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Filter parameters
@everywhere const RH_THRESHOLD       = 90.0    # %, "near-cloud" RH cutoff
@everywhere const CLOUD_MIN_NONCLEAN = 100.0   # m, lower cloud-base cutoff in nonclean mode
@everywhere const ZL_THRESHOLD       = 0.2     # |z/L| cutoff for stability classes


# Multi-stage lidar timestamp filter:
#   Stage 1: time-of-day window + month/year
#   Stage 2: stability (z/L)
#            "unstable": z/L < -ZL_THRESHOLD
#            "neutral":  |z/L| <= ZL_THRESHOLD
#            "stable":   z/L > ZL_THRESHOLD
#   Stage 3: clouds vs PBL relationship
#            "clean":    cloud_h >= PBL_h (cloud-free CBL or clouds above PBL)
#            "nonclean": CLOUD_MIN_NONCLEAN < cloud_h < PBL_h (clouds inside PBL)
#   Stage 4: no precipitation (precip_rate == 0)
#   Stage 5: RH-based moist-layer cutoff
#            "clean":    keep only if no RH >= 90% layer below PBL
#            "nonclean": always keep; replace PBL with bottom of RH >= 90% layer
#
# Output: filtered_time_lidar_<stability>_<condition>.csv with columns
#   day, month, year, time_s, pbl_h, cloud_h, z_L, rh_h


@everywhere function one_month(df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh,
                                time_in_h, time_out_h, month_flag, stability_flag, condition_flag)
    time_in  = time_in_h  * 3600
    time_out = time_out_h * 3600

    println("------", month_flag)

    l_day, l_month, l_year, l_time = df_time.day, df_time.month, df_time.year, df_time.time_start_s

    stab_day, stab_month, stab_year = df_stab.day, df_stab.month, df_stab.year
    stab_time, stab_val             = df_stab.time_s, df_stab.z_L

    pbl_day, pbl_month, pbl_year = df_pbl.day, df_pbl.month, df_pbl.year
    pbl_time, pbl_val            = df_pbl.time_s, df_pbl.pbl_filtered

    cld_day, cld_month, cld_year = df_clouds.day, df_clouds.month, df_clouds.year
    cld_time, cld_val            = df_clouds.time_s, df_clouds.cbh

    prc_day, prc_month, prc_year = df_precip.day, df_precip.month, df_precip.year
    prc_time, prc_val            = df_precip.time_s, df_precip.precip

    rh_day, rh_month, rh_year = df_rh.day, df_rh.month, df_rh.year
    rh_time   = df_rh.time_s
    rh_height = df_rh.height_m
    rh_val    = df_rh.rh

    println("total amount of profiles: ", length(l_time))

    # === Stage 1: time-of-day + month/year window ===
    if month_flag[1] == true
        mask_time = (l_time .>= time_in) .& (l_time .<= time_out) .&
                    (l_month .== month_flag[2]) .& (l_year .== month_flag[3])
    else
        mask_time = (l_time .>= time_in) .& (l_time .<= time_out)
    end

    l_day   = l_day[mask_time]
    l_month = l_month[mask_time]
    l_year  = l_year[mask_time]
    l_time  = l_time[mask_time]
    println("for ", time_in/3600, "-", time_out/3600, " time range: ", length(l_time))
    N_total = length(l_time)

    # === Stage 2: stability (z/L) ===
    ind_after_zl = []
    set_stblt    = []
    for t in 1:length(l_time)
        mask = (l_day[t] .== stab_day) .& (l_month[t] .== stab_month) .&
               (l_year[t] .== stab_year) .& (l_time[t] .== stab_time)
        stab_val_new = stab_val[mask]
        if !isempty(stab_val_new) && !ismissing(stab_val_new[1])
            v = stab_val_new[1]
            if stability_flag == "neutral"
                if abs(v) <= ZL_THRESHOLD
                    push!(ind_after_zl, t)
                    push!(set_stblt, v)
                end
            elseif stability_flag == "unstable"
                if v < -ZL_THRESHOLD
                    push!(ind_after_zl, t)
                    push!(set_stblt, v)
                end
            elseif stability_flag == "stable"
                if v > ZL_THRESHOLD
                    push!(ind_after_zl, t)
                    push!(set_stblt, v)
                end
            end
        end
    end

    l_day   = l_day[ind_after_zl]
    l_month = l_month[ind_after_zl]
    l_year  = l_year[ind_after_zl]
    l_time  = l_time[ind_after_zl]
    println("after stability-filter: ", length(l_time))
    N_zl = length(l_time)

    # === PBL height lookup ===
    pbl_for_lidar = []
    for t in 1:length(l_time)
        mask = (l_day[t] .== pbl_day) .& (l_month[t] .== pbl_month) .&
               (l_year[t] .== pbl_year) .& (l_time[t] .== pbl_time)
        pbl_val_new = pbl_val[mask]
        if !isempty(pbl_val_new) && !ismissing(pbl_val_new[1])
            push!(pbl_for_lidar, pbl_val_new[1])
        else
            push!(pbl_for_lidar, missing)
        end
    end

    # === Stage 3: clouds vs PBL ===
    ind_after_cld = []
    set_cld_h     = []
    set_pbl_h     = []
    for t in 1:length(l_time)
        mask = (l_day[t] .== cld_day) .& (l_month[t] .== cld_month) .&
               (l_year[t] .== cld_year) .& (l_time[t] .== cld_time)
        cld_val_new = cld_val[mask]
        pbl_t = pbl_for_lidar[t]

        if !isempty(cld_val_new) && !ismissing(cld_val_new[1]) && !ismissing(pbl_t)
            if condition_flag == "clean"
                if cld_val_new[1] >= pbl_t
                    push!(ind_after_cld, t)
                    push!(set_cld_h, cld_val_new[1])
                    push!(set_pbl_h, pbl_t)
                end
            elseif condition_flag == "nonclean"
                if cld_val_new[1] > CLOUD_MIN_NONCLEAN && cld_val_new[1] < pbl_t
                    push!(ind_after_cld, t)
                    push!(set_cld_h, cld_val_new[1])
                    push!(set_pbl_h, pbl_t)
                end
            end
        end
    end

    l_day     = l_day[ind_after_cld]
    l_month   = l_month[ind_after_cld]
    l_year    = l_year[ind_after_cld]
    l_time    = l_time[ind_after_cld]
    set_stblt = set_stblt[ind_after_cld]
    println("after cloud-filter: ", length(l_time))
    N_clouds = length(l_day)

    # === Stage 4: no precipitation ===
    ind_after_prc = []
    for t in 1:length(l_time)
        mask = (l_day[t] .== prc_day) .& (l_month[t] .== prc_month) .&
               (l_year[t] .== prc_year) .& (l_time[t] .== prc_time)
        prc_val_new = prc_val[mask]
        if !isempty(prc_val_new) && !ismissing(prc_val_new[1]) && prc_val_new[1] == 0
            push!(ind_after_prc, t)
        end
    end

    l_day     = l_day[ind_after_prc]
    l_month   = l_month[ind_after_prc]
    l_year    = l_year[ind_after_prc]
    l_time    = l_time[ind_after_prc]
    set_stblt = set_stblt[ind_after_prc]
    set_cld_h = set_cld_h[ind_after_prc]
    set_pbl_h = set_pbl_h[ind_after_prc]
    println("after precipitation-filter: ", length(l_time))
    N_prcpt = length(l_day)

    # If PBL is missing, fall back to cloud height
    set_pbl_h = ifelse.(ismissing.(set_pbl_h), set_cld_h, set_pbl_h)

    # === Stage 5: RH-based moist-layer cutoff ===
    ind_after_rh = []
    set_rh_h     = []
    for t in 1:length(l_time)
        mask = (l_day[t] .== rh_day) .& (l_month[t] .== rh_month) .&
               (l_year[t] .== rh_year) .& (l_time[t] .== rh_time)
        rh_height_new = rh_height[mask]
        rh_val_new    = rh_val[mask]

        pbl_t = ismissing(set_pbl_h[t]) ? NaN : set_pbl_h[t]

        # Only consider heights below PBL
        mask_h = pbl_t .>= rh_height_new
        rh_height_new = rh_height_new[mask_h]
        rh_val_new    = rh_val_new[mask_h]

        # Find lowest height where RH >= threshold
        inds = findall(x -> !ismissing(x) && x >= RH_THRESHOLD, rh_val_new)
        min_h = isempty(inds) ? pbl_t : minimum(rh_height_new[inds])

        if condition_flag == "clean"
            if min_h >= pbl_t
                push!(ind_after_rh, t)
                push!(set_rh_h, min_h)
            end
        elseif condition_flag == "nonclean"
            push!(ind_after_rh, t)
            push!(set_rh_h, min_h)
        end
    end

    l_day     = l_day[ind_after_rh]
    l_month   = l_month[ind_after_rh]
    l_year    = l_year[ind_after_rh]
    l_time    = l_time[ind_after_rh]
    set_stblt = set_stblt[ind_after_rh]
    set_cld_h = set_cld_h[ind_after_rh]
    set_pbl_h = set_pbl_h[ind_after_rh]
    println("after rh-filter: ", length(l_time))
    N_rh = length(l_day)

    return N_total, N_zl, N_clouds, N_prcpt, N_rh,
           [l_day, l_month, l_year, l_time, set_pbl_h, set_cld_h, set_stblt, set_rh_h]
end


@everywhere function compute_one_month(d, df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh,
                                        stability_flag, condition_flag)
    month_flag = [true, month(d), year(d)]
    true_time = Float64[]
    true_N1, true_N2, true_N3, true_N4, true_N5 =
        Float64[], Float64[], Float64[], Float64[], Float64[]
    filtered_time_lidar = (Int[], Int[], Int[], Float64[],
                           Union{Missing, Float64}[], Union{Missing, Float64}[],
                           Union{Missing, Float64}[], Union{Missing, Float64}[])

    for t in 0:1.0:23.0
        N1, N2, N3, N4, N5, time_info = one_month(df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh,
                                                   t, t + 1.0, month_flag, stability_flag, condition_flag)
        push!(true_time, t)
        push!(true_N1, N1)
        push!(true_N2, N2)
        push!(true_N3, N3)
        push!(true_N4, N4)
        push!(true_N5, N5)
        for j in 1:length(time_info)
            append!(filtered_time_lidar[j], time_info[j])
        end
    end
    return (d, true_time, true_N1, true_N2, true_N3, true_N4, true_N5, filtered_time_lidar)
end


function main(stability_flag, condition_flag)
    df_time   = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"),                            DataFrame)
    df_stab   = CSV.read(joinpath(PATH_output_txt, "z_L_stability_cond_aligned_with_lidar.csv"), DataFrame)
    df_pbl    = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"),       DataFrame)
    df_clouds = CSV.read(joinpath(PATH_output_txt, "clouds_aligned_with_lidar_dl.csv"),          DataFrame)
    df_precip = CSV.read(joinpath(PATH_output_txt, "prc_aligned_with_lidar.csv"),                DataFrame)
    df_rh     = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"), DataFrame)

    start_date = Date(2021, 10, 1)
    end_date   = Date(2022, 10, 1)
    months_all = collect(start_date:Month(1):end_date)

    ncols = 4
    nrows = cld(length(months_all), ncols)
    plt = plot(layout=(nrows, ncols), size=(1200, 300 * nrows), left_margin=10mm)

    filtered_time_lidar = (Int[], Int[], Int[], Float64[],
                           Union{Missing, Float64}[], Union{Missing, Float64}[],
                           Union{Missing, Float64}[], Union{Missing, Float64}[])

    total_N1, total_N2, total_N3, total_N4, total_N5 = 0.0, 0.0, 0.0, 0.0, 0.0

    results = pmap(d -> compute_one_month(d, df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh,
                                           stability_flag, condition_flag), months_all)

    vir = cgrad(:viridis)
    c1 = vir[0.2]   # Total
    c2 = vir[0.4]   # after ZL
    c3 = vir[0.6]   # after Clouds
    c4 = vir[0.8]   # after Precip+RH

    for (i, (d, true_time, true_N1, true_N2, true_N3, true_N4, true_N5, time_info)) in enumerate(results)
        total_N1 += sum(true_N1)
        total_N2 += sum(true_N2)
        total_N3 += sum(true_N3)
        total_N4 += sum(true_N4)
        total_N5 += sum(true_N5)

        bar!(plt, true_time, true_N1, bar_width=0.9, alpha=0.2, label="Total",            color=c1, subplot=i, ylim=(0, 150), xlim=[-1, 25])
        bar!(plt, true_time, true_N2, bar_width=0.6, alpha=0.5, label="ZL",               color=c2, subplot=i, ylim=(0, 150), xlim=[-1, 25])
        bar!(plt, true_time, true_N3, bar_width=0.3, alpha=0.8, label="Clouds",           color=c3, subplot=i, ylim=(0, 150), xlim=[-1, 25])
        bar!(plt, true_time, true_N5, bar_width=0.3, alpha=1.0, label="Precipitation+RH", color=c4, subplot=i, ylim=(0, 150), xlim=[-1, 25])

        title!(plt[i], "$(month(d))/$(year(d))")
        xlabel!(plt[i], "Time (h)")
        ylabel!(plt[i], "N")

        for j in 1:length(filtered_time_lidar)
            append!(filtered_time_lidar[j], time_info[j])
        end
    end

    println("total: ", total_N1, " after zl: ", total_N2, " after clouds: ", total_N3, " after precip: ", total_N4, " after rh: ", total_N5)
    println(round(total_N1/total_N1*100, digits=1), " ",
            round(total_N2/total_N1*100, digits=1), " ",
            round(total_N3/total_N1*100, digits=1), " ",
            round(total_N4/total_N1*100, digits=1), " ",
            round(total_N5/total_N1*100, digits=1))

    df_out = DataFrame(
        day     = filtered_time_lidar[1],
        month   = filtered_time_lidar[2],
        year    = filtered_time_lidar[3],
        time_s  = filtered_time_lidar[4],
        pbl_h   = filtered_time_lidar[5],
        cloud_h = filtered_time_lidar[6],
        z_L     = filtered_time_lidar[7],
        rh_h    = filtered_time_lidar[8],
    )
    output_csv = joinpath(PATH_output_txt, "filtered_time_lidar_$(stability_flag)_$(condition_flag).csv")
    CSV.write(output_csv, df_out)
    println(output_csv)

    output_png = joinpath(PATH_output_png, "monthly_time_vs_N_panel_$(stability_flag)_$(condition_flag).png")
    savefig(output_png)
    println(output_png)
end


# main("unstable", "clean")

cloud_options     = ["clean", "nonclean"]
stability_options = ["unstable", "neutral", "stable"]
for cf in cloud_options
    for sf in stability_options
        main(sf, cf)
    end
end