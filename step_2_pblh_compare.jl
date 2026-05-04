using DelimitedFiles
using Statistics
using DataFrames
using CSV
using Plots
using Measures
using ColorSchemes
using LaTeXStrings

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Matching / agreement parameters
const TIME_GAP_H        = 3       # h, max time difference for nearest-neighbor PBL match
const PBL_AGREE_M       = 500     # m, max difference between two PBL methods to be considered agreeing
const DAYTIME_LO        = 12      # h, daytime window for histograms / scatter
const DAYTIME_HI        = 15      # h
const HIST_BINS         = 30
const PBLH_BIN_EDGES    = 0:250:3500   # m


# Compares the independent PBL-height retrievals (ceilometer, four sonde
# methods, temperature inversion) and produces two canonical PBL records.
# All input files now use time_s (local seconds-of-day).



function find_closest(day_arr, month_arr, year_arr, time_arr, val_arr, day, month, year, time_sec)
    time_h = round(time_sec / 3600, digits=1)
    mask = (day_arr .== day) .& (month_arr .== month) .& (year_arr .== year)
    if !any(mask)
        return missing
    end

    times = time_arr[mask]
    vals  = val_arr[mask]
    mask_valid = .!ismissing.(vals)
    if !any(mask_valid)
        return missing
    end

    times = times[mask_valid]
    vals  = vals[mask_valid]
    diffs = abs.(times .- time_h)
    idx = argmin(diffs)
    return diffs[idx] <= TIME_GAP_H ? vals[idx] : missing
end


function main()
    # Read ceil PBL data (time in seconds)
    df_ceil = CSV.read(joinpath(PATH_output_txt, "pbl_ceil.csv"), DataFrame)
    ceil_time_h = round.(df_ceil.time_s ./ 3600, digits=1)

    # Read inversion PBL data (time in seconds, already on lidar grid)
    df_inv = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_temp_inversion.csv"), DataFrame)
    inv_time_h = round.(df_inv.time_s ./ 3600, digits=1)

    # Read sonde PBL data (time in seconds)
    df_sonde = CSV.read(joinpath(PATH_output_txt, "pbl_sonde.csv"), DataFrame)
    sonde_time_h = round.(df_sonde.time_s ./ 3600, digits=1)

    # Read lidar time grid (time in seconds)
    df_lidar = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"), DataFrame)

    combos = unique([(df_lidar[i,:day], df_lidar[i,:month], df_lidar[i,:year], df_lidar[i,:time_start_s])
                     for i in 1:nrow(df_lidar)])
    println("Total lidar time points: ", length(combos))

    results = Any[]
    for (day, month, year, time_sec) in combos
        pbl_ceil   = find_closest(df_ceil.day,  df_ceil.month,  df_ceil.year,  ceil_time_h,  df_ceil.pbl_h,            day, month, year, time_sec)
        pbl_sonde1 = find_closest(df_sonde.day, df_sonde.month, df_sonde.year, sonde_time_h, df_sonde.pbl_heffter,     day, month, year, time_sec)
        pbl_sonde2 = find_closest(df_sonde.day, df_sonde.month, df_sonde.year, sonde_time_h, df_sonde.pbl_liu_liang,   day, month, year, time_sec)
        pbl_sonde3 = find_closest(df_sonde.day, df_sonde.month, df_sonde.year, sonde_time_h, df_sonde.pbl_bulkri_pt25, day, month, year, time_sec)
        pbl_sonde4 = find_closest(df_sonde.day, df_sonde.month, df_sonde.year, sonde_time_h, df_sonde.pbl_bulkri_pt5,  day, month, year, time_sec)
        pbl_inv    = find_closest(df_inv.day,   df_inv.month,   df_inv.year,   inv_time_h,   df_inv.pbl_h,             day, month, year, time_sec)

        push!(results, [day, month, year, time_sec, pbl_ceil, pbl_sonde1, pbl_sonde2, pbl_sonde3, pbl_sonde4, pbl_inv])
    end

    df_out = DataFrame(
        day        = Int[r[1] for r in results],
        month      = Int[r[2] for r in results],
        year       = Int[r[3] for r in results],
        time_s     = Float64[r[4] for r in results],
        pbl_ceil   = [r[5] for r in results],
        pbl_sonde1 = [r[6] for r in results],
        pbl_sonde2 = [r[7] for r in results],
        pbl_sonde3 = [r[8] for r in results],
        pbl_sonde4 = [r[9] for r in results],
        pbl_inv    = [r[10] for r in results],
    )
    output_file = joinpath(PATH_output_txt, "pbl_aligned_with_lidar_all.csv")
    CSV.write(output_file, df_out)
    println(output_file)
end


function plot_pbl_months()
    df = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_all.csv"), DataFrame)

    mask_daytime = (df.time_s ./ 3600 .>= DAYTIME_LO) .& (df.time_s ./ 3600 .<= DAYTIME_HI)
    df = df[mask_daytime, :]

    label_map = Dict("pbl_ceil"   => "Ceil",
                     "pbl_inv"    => "Temp Inv",
                     "pbl_sonde1" => "Heffter",
                     "pbl_sonde2" => "Liu-Liang",
                     "pbl_sonde3" => "Richardson1",
                     "pbl_sonde4" => "Richardson2")
    
    set_pbl_1 = df.pbl_sonde2
    set_pbl_2 = df.pbl_inv

    name_1 = "pbl_sonde2"
    name_2 = "pbl_inv"

    months = sort(unique(df.month))
    p = []
    for m in months
        mask_m = df.month .== m
        valid = .!(ismissing.(set_pbl_1[mask_m]) .| ismissing.(set_pbl_2[mask_m]))

        if any(valid)
            println(m, " ", length(set_pbl_1[mask_m][valid]))
            p_m = scatter(set_pbl_1[mask_m][valid], set_pbl_2[mask_m][valid],
                          xlabel=label_map[name_1],
                          ylabel=label_map[name_2],
                          title="Month = $m",
                          legend=false,
                          markersize=4,
                          alpha=0.1,
                          color=:blue,
                          xlims=[0, 3000], ylims=[0, 3000])
            plot!(p_m, [0, 3000], [0, 3000],
                  color=:red, linestyle=:dash, linewidth=1.5, label=false)
        else
            p_m = plot(title="Month = $m\n(no valid data)", grid=false, xlims=[0, 3000], ylims=[0, 3000])
        end

        push!(p, p_m)
    end

    layout_grid = (ceil(Int, length(months) / 3), 3)
    plt = plot(p..., layout=layout_grid, size=(500*4, 450*3), bottom_margin=10mm, left_margin=10mm)
    output_file = joinpath(PATH_output_png, "pbl_sonde2_vs_ceil_by_month.png")
    savefig(plt, output_file)
    println(output_file)
end


function pbl_filtered()
    df = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_all.csv"), DataFrame)

    label_map = Dict(
        "pbl_ceil"   => "Ceil",
        "pbl_inv"    => "Temp Inv",
        "pbl_sonde1" => "Heffter",
        "pbl_sonde2" => "Liu-Liang",
        "pbl_sonde3" => "Richardson1",
        "pbl_sonde4" => "Richardson2",
    )

    data_map = Dict(
        "pbl_ceil"   => df.pbl_ceil,
        "pbl_inv"    => df.pbl_inv,
        "pbl_sonde1" => df.pbl_sonde1,
        "pbl_sonde2" => df.pbl_sonde2,
        "pbl_sonde3" => df.pbl_sonde3,
        "pbl_sonde4" => df.pbl_sonde4,
    )

    pairs = [
        ("pbl_sonde2", "pbl_sonde1"),
        ("pbl_sonde2", "pbl_sonde3"),
        ("pbl_sonde2", "pbl_sonde4"),
        ("pbl_sonde2", "pbl_ceil"),
        ("pbl_sonde2", "pbl_inv"),
    ]

    p = []
    for (name_x, name_y) in pairs
        set_x = data_map[name_x]
        set_y = data_map[name_y]
        valid = .!(ismissing.(set_x) .| ismissing.(set_y))

        if any(valid)
            p_m = scatter(set_x[valid], set_y[valid],
                          xlabel=label_map[name_x],
                          ylabel=label_map[name_y],
                          legend=false,
                          markersize=4,
                          alpha=0.15,
                          color=:blue,
                          xlims=[0, 3000], ylims=[0, 3000])
            plot!(p_m, [0, 3000], [0, 3000],
                  color=:red, linestyle=:dash, linewidth=1.5, label=false)
            upper_line = [PBL_AGREE_M, 3000+PBL_AGREE_M]
            lower_line = [-PBL_AGREE_M, 3000-PBL_AGREE_M]
            plot!(p_m, [0, 3000], upper_line, fillrange=lower_line, fillalpha=0.2,
                  fillcolor=:blue, linecolor=:blue, linestyle=:dash, lw=1.5, label="±$(PBL_AGREE_M) m band")
            plot!(p_m, [0, 3000], upper_line, color=:blue, lw=1.5, linestyle=:dash, label="+$(PBL_AGREE_M) m")
            plot!(p_m, [0, 3000], lower_line, color=:blue, lw=1.5, linestyle=:dash, label="-$(PBL_AGREE_M) m")
        else
            p_m = plot(title="$(label_map[name_y]) vs $(label_map[name_x])\n(no valid data)",
                       grid=false, xlims=[0, 3000], ylims=[0, 3000])
        end
        push!(p, p_m)
    end

    layout_grid = (ceil(Int, length(pairs) / 3), 3)
    plt = plot(p..., layout=layout_grid, size=(500*length(pairs), 450*2),
               left_margin=15mm, bottom_margin=10mm)
    output_file = joinpath(PATH_output_png, "pbl_aligned_with_lidar_filtered.png")
    savefig(plt, output_file)
    println(output_file)

    # Filtering rule
    n = nrow(df)
    pbl_filt = Vector{Union{Missing, Float64}}(missing, n)

    for i in 1:n
        ll  = df.pbl_sonde2[i]
        inv = df.pbl_inv[i]

        if !ismissing(ll) && !ismissing(inv)
            if abs(ll - inv) <= PBL_AGREE_M
                pbl_filt[i] = Float64(ll)
            end
        elseif !ismissing(ll)
            pbl_filt[i] = Float64(ll)
        elseif !ismissing(inv)
            pbl_filt[i] = Float64(inv)
        end
    end

    df_filt = DataFrame(
        day          = df.day,
        month        = df.month,
        year         = df.year,
        time_s       = df.time_s,
        pbl_filtered = pbl_filt,
    )
    println("Total points  : $n")
    println("Kept points   : $(count(!ismissing, pbl_filt))")
    println("Missing points: $(count(ismissing, pbl_filt))")

    output_file = joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv")
    CSV.write(output_file, df_filt)
    println(output_file)
end


function pbl_combined()
    df_filt = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"), DataFrame)
    pbl_f = df_filt.pbl_filtered

    df = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_all.csv"), DataFrame)

    pbl_ceil   = df.pbl_ceil
    pbl_sonde2 = df.pbl_sonde2
    pbl_inv    = df.pbl_inv

    n = nrow(df)
    pbl_comb = Vector{Union{Missing, Float64}}(missing, n)
    chosen_pair  = Vector{Union{Missing, String}}(missing, n)

    for i in 1:n
        ll  = pbl_sonde2[i]
        cl  = pbl_ceil[i]
        inv = pbl_inv[i]

        vals       = [ll, cl, inv]
        labels_lcl = ["LL", "Ceil", "Inv"]
        valid      = .!ismissing.(vals)
        n_valid    = sum(valid)

        if n_valid == 0
            continue
        elseif n_valid == 1
            pbl_comb[i] = Float64(vals[findfirst(valid)])
            chosen_pair[i]  = labels_lcl[findfirst(valid)]
        elseif n_valid == 2
            idx = findall(valid)
            v1, v2 = Float64(vals[idx[1]]), Float64(vals[idx[2]])
            pbl_comb[i] = (v1 + v2) / 2
            chosen_pair[i]  = labels_lcl[idx[1]] * "+" * labels_lcl[idx[2]]
        else
            v1, v2, v3 = Float64(ll), Float64(cl), Float64(inv)
            d12 = abs(v1 - v2)
            d13 = abs(v1 - v3)
            d23 = abs(v2 - v3)
            d_min = min(d12, d13, d23)
            if d12 == d_min
                pbl_comb[i] = (v1 + v2) / 2
                chosen_pair[i]  = "LL+Ceil"
            elseif d13 == d_min
                pbl_comb[i] = (v1 + v3) / 2
                chosen_pair[i]  = "LL+Inv"
            else
                pbl_comb[i] = (v2 + v3) / 2
                chosen_pair[i]  = "Ceil+Inv"
            end
        end
    end

    println("\n=== Combined PBLH (closest pair) ===")
    println("Total points  : $n")
    println("Kept points   : $(count(!ismissing, pbl_comb))")
    println("Pair breakdown:")
    for s in ["LL+Ceil", "LL+Inv", "Ceil+Inv", "LL", "Ceil", "Inv"]
        c = count(==(s), skipmissing(chosen_pair))
        if c > 0
            println("  $s: $c")
        end
    end

    colors = ColorSchemes.viridis[range(0, 1, length=6)]
    p1 = histogram(pbl_sonde2, bins=HIST_BINS, label="Sonde (Liu-Liang)", xlabel="PBLH (m)", ylabel="Count", color=colors[2], alpha=1)
    histogram!(p1, pbl_ceil, bins=HIST_BINS, label="Ceil",     color=colors[3], alpha=0.7)
    histogram!(p1, pbl_inv,  bins=HIST_BINS, label="Temp Inv", color=colors[4], alpha=0.7)
    annotate!(p1, (700, 6900, text("a", :left, 18)))

    p2 = histogram(pbl_comb, bins=HIST_BINS, xlabel="PBLH (m)", ylabel="Count", color=colors[1], alpha=0.5, label="combined")
    histogram!(p2, pbl_f,    bins=HIST_BINS, label="filtered",  xlabel="PBLH (m)", ylabel="Count", color=colors[5], alpha=0.5)
    annotate!(p2, (700, 5600, text("b", :left, 18)))

    plt = plot(p1, p2, layout=(1, 2), size=(1500, 450), left_margin=10mm, bottom_margin=10mm)
    output_file = joinpath(PATH_output_png, "pbl_aligned_with_lidar_combine.png")
    savefig(plt, output_file)
    println(output_file)

    df_out = DataFrame(
        day      = df.day,
        month    = df.month,
        year     = df.year,
        time_s   = df.time_s,
        pbl_comb = pbl_comb,
        source   = chosen_pair,
    )
    output_file1 = joinpath(PATH_output_txt, "pbl_aligned_with_lidar_combine.csv")
    CSV.write(output_file1, df_out)
    println(output_file1)
end


function load_filtered_with_sigma()
    df_filt = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"), DataFrame)
    df_all  = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_all.csv"),      DataFrame)

    @assert nrow(df_filt) == nrow(df_all) "Filtered and full-method files have different row counts"

    n = nrow(df_filt)
    sigma = Vector{Union{Missing, Float64}}(missing, n)

    for i in 1:n
        pbl_kept = df_filt.pbl_filtered[i]
        ll       = df_all.pbl_sonde2[i]
        inv      = df_all.pbl_inv[i]

        if ismissing(pbl_kept)
            continue
        end

        if !ismissing(ll) && !ismissing(inv)
            sigma[i] = std([Float64(ll), Float64(inv)])
        else
            sigma[i] = 0.0
        end
    end

    df_out = DataFrame(
        day    = df_filt.day,
        month  = df_filt.month,
        year   = df_filt.year,
        time_s = df_filt.time_s,
        time_h = df_filt.time_s ./ 3600,
        pblh   = df_filt.pbl_filtered,
        sigma  = sigma,
    )
    return df_out
end


function season_of(month)
    if month in (12, 1, 2)
        return "DJF"
    elseif month in (3, 4, 5)
        return "MAM"
    elseif month in (6, 7, 8)
        return "JJA"
    else
        return "SON"
    end
end


function plot_pblh_diurnal_seasonal_filtered()
    df = load_filtered_with_sigma()
    df.season = season_of.(df.month)
    df.hour   = floor.(Int, df.time_h)

    seasons = ["DJF", "MAM", "JJA", "SON"]
    panels = []

    for s in seasons
        mask = (df.season .== s) .& .!ismissing.(df.pblh)
        df_s = df[mask, :]

        hours_present = sort(unique(df_s.hour))
        pblh_by_hour = [collect(skipmissing(df_s[df_s.hour .== h, :pblh])) for h in hours_present]

        valid_idx = findall(x -> length(x) >= 5, pblh_by_hour)
        hours_use = hours_present[valid_idx]
        pblh_use  = pblh_by_hour[valid_idx]
        n = length(hours_use)

        p10 = [quantile(x, 0.10) for x in pblh_use]
        q1  = [quantile(x, 0.25) for x in pblh_use]
        med = [median(x)         for x in pblh_use]
        q3  = [quantile(x, 0.75) for x in pblh_use]
        p90 = [quantile(x, 0.90) for x in pblh_use]

        p = plot(xlabel=L"$Time,\,h$",
                 ylabel=L"$PBLH,\,m$",
                 title=s,
                 legend=false,
                 ylim=(0, 3000),
                 xlim=(0, n+1),
                 xticks=(1:n, string.(hours_use)),
                 xguidefont=16, yguidefont=16,
                 xtickfont=font(10), ytickfont=font(10),
                 titlefont=font(18))

        bw = 0.35   # box half-width
        cw = 0.18   # whisker cap half-width
        for i in 1:n
            plot!(p, Shape([i-bw, i+bw, i+bw, i-bw],
                           [q1[i], q1[i], q3[i], q3[i]]),
                  fillcolor=:lightblue, fillalpha=0.6,
                  linecolor=:black, linewidth=1.0, label="")
            plot!(p, [i-bw, i+bw], [med[i], med[i]],
                  color=:black, linewidth=1.5, label="")
            plot!(p, [i, i], [p10[i], q1[i]],  color=:black, linewidth=1.0, label="")
            plot!(p, [i-cw, i+cw], [p10[i], p10[i]], color=:black, linewidth=1.0, label="")
            plot!(p, [i, i], [q3[i], p90[i]],  color=:black, linewidth=1.0, label="")
            plot!(p, [i-cw, i+cw], [p90[i], p90[i]], color=:black, linewidth=1.0, label="")
        end

        push!(panels, p)
    end

    plt = plot(panels..., layout=(2, 2), size=(1400, 900),
               left_margin=12mm, bottom_margin=12mm)
    output_file = joinpath(PATH_output_png, "pbl_diurnal_seasonal_filtered.png")
    savefig(plt, output_file)
    println(output_file)
end


function summarize_pblh_by_season()
    df_filt = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"), DataFrame)
    df_comb = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_combine.csv"),  DataFrame)

    for df in (df_filt, df_comb)
        df.hour   = floor.(Int, df.time_s ./ 3600)
        df.season = season_of.(df.month)
    end

    seasons = ["DJF", "MAM", "JJA", "SON"]
    rows = []

    function stats_block(df::DataFrame, valcol::Symbol, label::String)
        mask = (df.hour .>= DAYTIME_LO) .& (df.hour .<= DAYTIME_HI)
        for s in seasons
            mask_s = mask .& (df.season .== s)
            vals = collect(skipmissing(df[mask_s, valcol]))
            vals = Float64.(vals)
            vals = vals[isfinite.(vals)]
            if isempty(vals)
                push!(rows, (label, s, NaN, NaN, NaN, NaN, NaN, 0))
            else
                push!(rows, (label, s,
                             mean(vals),
                             median(vals),
                             std(vals),
                             quantile(vals, 0.25),
                             quantile(vals, 0.75),
                             length(vals)))
            end
        end
    end

    stats_block(df_filt, :pbl_filtered, "filtered")
    stats_block(df_comb, :pbl_comb,     "combined")

    df_summary = DataFrame(
        method = [r[1] for r in rows],
        season = [r[2] for r in rows],
        mean   = round.([r[3] for r in rows], digits=0),
        median = round.([r[4] for r in rows], digits=0),
        std    = round.([r[5] for r in rows], digits=0),
        q25    = round.([r[6] for r in rows], digits=0),
        q75    = round.([r[7] for r in rows], digits=0),
        n      = [r[8] for r in rows],
    )

    println("\n=== PBLH SEASONAL SUMMARY (12–15 LT) ===")
    println(df_summary)

    output_file = joinpath(PATH_output_txt, "pbl_seasonal_summary.csv")
    CSV.write(output_file, df_summary)
    println("\nSaved: ", output_file)
end


function plot_pblh_source_diurnal_seasonal_filtered()
    df_filt = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"), DataFrame)
    df_all  = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_all.csv"),      DataFrame)
    @assert nrow(df_filt) == nrow(df_all) "Filtered and full-method files have different row counts"

    n = nrow(df_filt)
    source = Vector{Union{Missing,Symbol}}(missing, n)
    for i in 1:n
        ismissing(df_filt.pbl_filtered[i]) && continue
        ll  = df_all.pbl_sonde2[i]
        inv = df_all.pbl_inv[i]
        if !ismissing(ll)
            source[i] = :ll
        elseif !ismissing(inv)
            source[i] = :inv
        end
    end

    df = DataFrame(
        month  = df_filt.month,
        hour   = floor.(Int, df_filt.time_s ./ 3600),
        source = source,
    )
    df.season = season_of.(df.month)
    df = df[.!ismissing.(df.source), :]

    seasons = ["DJF", "MAM", "JJA", "SON"]
    panels = []

    for s in seasons
        df_s = df[df.season .== s, :]
        hours_use = sort(unique(df_s.hour))
        nh = length(hours_use)

        n_ll  = [count(==(:ll ), df_s[df_s.hour .== h, :source]) for h in hours_use]
        n_inv = [count(==(:inv), df_s[df_s.hour .== h, :source]) for h in hours_use]

        ymax = maximum(n_ll .+ n_inv)

        p = plot(xlabel=L"$Time,\,h$",
                 ylabel=L"$Count$",
                 title=s,
                 legend= :topright,
                 xlim=(0, nh+1),
                 ylim=(0, ymax * 1.1),
                 xticks=(1:nh, string.(hours_use)),
                 xguidefont=16, yguidefont=16,
                 xtickfont=font(10), ytickfont=font(10),
                 titlefont=font(18))

        bw = 0.4
        seen = Dict{Symbol,Bool}()
        for i in 1:nh
            y0 = 0.0
            for (cnt, col, lbl, key) in ((n_ll[i],  :steelblue, "LL", :ll),
                                         (n_inv[i], :gray,      "TI", :ti))
                cnt == 0 && continue
                this_label = get(seen, key, false) ? "" : (seen[key] = true; lbl)
                plot!(p, Shape([i-bw, i+bw, i+bw, i-bw],
                            [y0, y0, y0+cnt, y0+cnt]),
                      fillcolor=col, fillalpha=0.85,
                      linecolor=:black, linewidth=0.5,
                      label = this_label)
                y0 += cnt
            end
        end
        push!(panels, p)
    end

    plt = plot(panels..., layout=(2, 2), size=(1400, 900),
               left_margin=12mm, bottom_margin=12mm)
    output_file = joinpath(PATH_output_png, "pbl_source_diurnal_seasonal_filtered.png")
    savefig(plt, output_file)
    println(output_file)
end



main()
pbl_filtered()
pbl_combined()

# plot_pbl_months()

plot_pblh_diurnal_seasonal_filtered()
summarize_pblh_by_season()
plot_pblh_source_diurnal_seasonal_filtered()