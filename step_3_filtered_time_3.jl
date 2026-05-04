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


# Unified multi-stage lidar timestamp filter (single output, all categories
# carried as columns). Stages applied in order:
#
#   Stage 1: time-of-day window + month/year
#   Stage 2: clouds vs PBL relationship -> condition flag
#              cloud_h >= PBL_h                     -> condition =  1 (clean / "complete")
#              CLOUD_MIN_NONCLEAN < cloud_h < PBL_h -> condition = -1 (nonclean / "partial")
#              other                                -> reject
#   Stage 3: no precipitation (precip_rate == 0)
#   Stage 4: RH-based moist-layer cutoff (per Petters et al. 2024):
#              clean (cond= 1):  if RH >= 90% layer below PBL exists ->
#                                 RELABEL as nonclean (cond=-1) and use min_h as ceiling
#              nonclean (cond=-1): keep; replace ceiling with bottom of RH >= 90% layer
#                                  if it is lower than the cloud-derived ceiling
#   Stage 5: stability (z/L) classification -> stability flag (label only, no filter)
#              z/L < -ZL_THRESHOLD : stability = -1 (unstable)
#              |z/L| <= ZL_THRESHOLD: stability =  0 (neutral)
#              z/L >  ZL_THRESHOLD : stability =  1 (stable)
#              missing             : stability =  missing
#            Filtering by stability is done downstream.
#
# Output: filtered_time_lidar.csv with columns
#   day, month, year, time_s, pbl_h, cloud_h, rh_h, z_L, stability, condition


@everywhere function classify_stability(zL)
    ismissing(zL)              && return missing
    abs(zL) <= ZL_THRESHOLD    && return 0
    zL < -ZL_THRESHOLD         && return -1
    return 1
end


@everywhere function one_month(df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh,
                                time_in_h, time_out_h, month_flag)
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
    N_total = length(l_time)

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

    # === Stage 2: clouds vs PBL  (yields condition flag) ===
    ind_after_cld = Int[]
    set_cld_h     = []
    set_pbl_h     = []
    set_cond      = Int[]
    for t in 1:length(l_time)
        mask = (l_day[t] .== cld_day) .& (l_month[t] .== cld_month) .&
               (l_year[t] .== cld_year) .& (l_time[t] .== cld_time)
        cld_val_new = cld_val[mask]
        pbl_t = pbl_for_lidar[t]

        if !isempty(cld_val_new) && !ismissing(cld_val_new[1]) && !ismissing(pbl_t)
            ch = cld_val_new[1]
            if ch >= pbl_t
                push!(ind_after_cld, t)
                push!(set_cld_h, ch)
                push!(set_pbl_h, pbl_t)
                push!(set_cond, 1)              # clean
            elseif ch > CLOUD_MIN_NONCLEAN && ch < pbl_t
                push!(ind_after_cld, t)
                push!(set_cld_h, ch)
                push!(set_pbl_h, pbl_t)
                push!(set_cond, -1)             # nonclean
            end
        end
    end

    l_day   = l_day[ind_after_cld]
    l_month = l_month[ind_after_cld]
    l_year  = l_year[ind_after_cld]
    l_time  = l_time[ind_after_cld]
    N_clouds = length(l_day)

    # === Stage 3: no precipitation ===
    ind_after_prc = Int[]
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
    set_cld_h = set_cld_h[ind_after_prc]
    set_pbl_h = set_pbl_h[ind_after_prc]
    set_cond  = set_cond[ind_after_prc]
    N_prcpt = length(l_day)

    # === Stage 4: RH-based moist-layer cutoff ===
    # Per Petters et al. (2024): RH >= 90% layer below PBL truncates the profile
    # and forces a "partial" classification (cond = -1).
    ind_after_rh = Int[]
    set_rh_h     = []
    set_cond_new = Int[]
    for t in 1:length(l_time)
        mask = (l_day[t] .== rh_day) .& (l_month[t] .== rh_month) .&
               (l_year[t] .== rh_year) .& (l_time[t] .== rh_time)
        rh_height_new = rh_height[mask]
        rh_val_new    = rh_val[mask]

        pbl_t = set_pbl_h[t]   # always non-missing here (Stage 2 guarantees it)

        # Only consider heights below PBL
        mask_h = pbl_t .>= rh_height_new
        rh_height_new = rh_height_new[mask_h]
        rh_val_new    = rh_val_new[mask_h]

        # Find lowest height where RH >= threshold
        inds = findall(x -> !ismissing(x) && x >= RH_THRESHOLD, rh_val_new)
        rh_layer_present = !isempty(inds)
        min_h = rh_layer_present ? minimum(rh_height_new[inds]) : pbl_t

        cond_in = set_cond[t]
        if cond_in == 1                      # clean (cloud >= PBL)
            if rh_layer_present
                # RH >= 90% below PBL --> relabel as partial (per paper)
                push!(ind_after_rh, t)
                push!(set_rh_h, min_h)
                push!(set_cond_new, -1)
            else
                # No moist layer below PBL --> stays clean
                push!(ind_after_rh, t)
                push!(set_rh_h, pbl_t)
                push!(set_cond_new, 1)
            end
        elseif cond_in == -1                 # nonclean (cloud inside PBL)
            # Use the lower of cloud-derived and RH-derived ceilings
            ceil_h = rh_layer_present ? min(min_h, set_cld_h[t]) : set_cld_h[t]
            push!(ind_after_rh, t)
            push!(set_rh_h, ceil_h)
            push!(set_cond_new, -1)
        end
    end

    l_day     = l_day[ind_after_rh]
    l_month   = l_month[ind_after_rh]
    l_year    = l_year[ind_after_rh]
    l_time    = l_time[ind_after_rh]
    set_cld_h = set_cld_h[ind_after_rh]
    set_pbl_h = set_pbl_h[ind_after_rh]
    set_cond  = set_cond_new
    N_rh = length(l_day)

    # === Stage 5: stability (z/L) classification — keep all, just label ===
    set_zL    = Vector{Union{Missing, Float64}}(undef, length(l_time))
    set_stab  = Vector{Union{Missing, Int}}(undef, length(l_time))
    for t in 1:length(l_time)
        mask = (l_day[t] .== stab_day) .& (l_month[t] .== stab_month) .&
               (l_year[t] .== stab_year) .& (l_time[t] .== stab_time)
        stab_val_new = stab_val[mask]
        if !isempty(stab_val_new) && !ismissing(stab_val_new[1])
            v = stab_val_new[1]
            set_zL[t]   = v
            set_stab[t] = classify_stability(v)
        else
            set_zL[t]   = missing
            set_stab[t] = missing
        end
    end

    return N_total, N_clouds, N_prcpt, N_rh,
           [l_day, l_month, l_year, l_time, set_pbl_h, set_cld_h, set_rh_h, set_zL, set_stab, set_cond]
end


@everywhere function compute_one_month(d, df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh)
    month_flag = [true, month(d), year(d)]
    true_time = Float64[]
    true_N1, true_N2, true_N3, true_N4 = Float64[], Float64[], Float64[], Float64[]
    filtered_time_lidar = (Int[], Int[], Int[], Float64[],
                           Union{Missing, Float64}[], Union{Missing, Float64}[], Union{Missing, Float64}[],
                           Union{Missing, Float64}[], Union{Missing, Int}[], Union{Missing, Int}[])

    for t in 0:1.0:23.0
        N1, N2, N3, N4, time_info = one_month(df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh,
                                               t, t + 1.0, month_flag)
        push!(true_time, t)
        push!(true_N1, N1)
        push!(true_N2, N2)
        push!(true_N3, N3)
        push!(true_N4, N4)
        for j in 1:length(time_info)
            append!(filtered_time_lidar[j], time_info[j])
        end
    end
    return (d, true_time, true_N1, true_N2, true_N3, true_N4, filtered_time_lidar)
end


function main()
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
                           Union{Missing, Float64}[], Union{Missing, Float64}[], Union{Missing, Float64}[],
                           Union{Missing, Float64}[], Union{Missing, Int}[], Union{Missing, Int}[])

    total_N1, total_N2, total_N3, total_N4 = 0.0, 0.0, 0.0, 0.0

    results = pmap(d -> compute_one_month(d, df_time, df_stab, df_pbl, df_clouds, df_precip, df_rh), months_all)

    vir = cgrad(:viridis)
    c1 = vir[0.2]   # Total
    c2 = vir[0.5]   # after Clouds
    c3 = vir[0.8]   # after Precip+RH

    for (i, (d, true_time, true_N1, true_N2, true_N3, true_N4, time_info)) in enumerate(results)
        total_N1 += sum(true_N1)
        total_N2 += sum(true_N2)
        total_N3 += sum(true_N3)
        total_N4 += sum(true_N4)

        bar!(plt, true_time, true_N1, bar_width=0.9, alpha=0.2, label="Total",            color=c1, subplot=i, ylim=(0, 200), xlim=[-1, 25])
        bar!(plt, true_time, true_N2, bar_width=0.6, alpha=0.6, label="Clouds",           color=c2, subplot=i, ylim=(0, 200), xlim=[-1, 25])
        bar!(plt, true_time, true_N4, bar_width=0.3, alpha=1.0, label="Precipitation+RH", color=c3, subplot=i, ylim=(0, 200), xlim=[-1, 25])

        title!(plt[i], "$(month(d))/$(year(d))")
        xlabel!(plt[i], "Time (h)")
        ylabel!(plt[i], "N")

        for j in 1:length(filtered_time_lidar)
            append!(filtered_time_lidar[j], time_info[j])
        end
    end

    println("total: ", total_N1, " after clouds: ", total_N2, " after precip: ", total_N3, " after rh: ", total_N4)
    println(round(total_N1/total_N1*100, digits=1), " ",
            round(total_N2/total_N1*100, digits=1), " ",
            round(total_N3/total_N1*100, digits=1), " ",
            round(total_N4/total_N1*100, digits=1))

    df_out = DataFrame(
        day        = filtered_time_lidar[1],
        month      = filtered_time_lidar[2],
        year       = filtered_time_lidar[3],
        time_s     = filtered_time_lidar[4],
        pbl_h      = filtered_time_lidar[5],
        cloud_h    = filtered_time_lidar[6],
        rh_h       = filtered_time_lidar[7],
        z_L        = filtered_time_lidar[8],
        stability  = filtered_time_lidar[9],
        condition  = filtered_time_lidar[10],
    )

    # Per-category breakdown
    println("\n=== Per-category breakdown ===")
    for cond_v in [1, -1]
        for stab_v in [-1, 0, 1]
            n = sum(skipmissing(df_out.condition .== cond_v .&& df_out.stability .== stab_v))
            println("  cond=$cond_v stab=$stab_v : $n")
        end
        n_missing = sum(df_out.condition .== cond_v .&& ismissing.(df_out.stability))
        println("  cond=$cond_v stab=missing : $n_missing")
    end

    output_csv = joinpath(PATH_output_txt, "filtered_time_lidar.csv")
    CSV.write(output_csv, df_out)
    println("\n", output_csv)

    output_png = joinpath(PATH_output_png, "monthly_time_vs_N_panel.png")
    savefig(output_png)
    println(output_png)
end


main()