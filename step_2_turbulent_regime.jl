using NCDatasets
using Dates
using TimeZones
using Statistics
using DataFrames
using CSV
using DelimitedFiles
using Plots
using Measures
using LaTeXStrings

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Physical constants
const G  = 9.81    # gravity [m s-2]
const CP = 1005.5  # specific heat at const pressure [J kg-1 K-1]

# Reference levels for plot guidelines (no longer used for grouping)
const USTAR_LEVEL_1 = 0.2     # m/s
const USTAR_LEVEL_2 = 0.35    # m/s
const HEAT_LEVEL_1  = 50      # W/m^2
const HEAT_LEVEL_2  = 120
const WSTAR_LEVEL_1 = 0.8     # m/s
const WSTAR_LEVEL_2 = 1.5

# Alignment
const TIME_GAP_S = 1 * 3600   # max time difference (s) between lidar and ECOR sample

# Daytime window for plotting
const DAYTIME_LO = 12
const DAYTIME_HI = 15


# Combined turbulent-regime processing.
#   main          : reads each ECOR file, writes turbulent_regime.csv with
#                   day/month/year/time_s/ustar/h/mean_t/rho/mr.
#   compute_wstar : merges turbulent_regime.csv with the lidar-aligned PBL
#                   height (pbl_aligned_with_lidar_filtered.csv) to compute
#                   the Deardorff convective velocity scale w*. Writes
#                   turbulent_regime_full.csv with the columns above + wstar.
#   align_data_with_lidar : matches each lidar block timestamp to the nearest
#                   ECOR sample within TIME_GAP_S; writes
#                   regime_aligned_with_lidar.csv with ustar, h, wstar.
#
# w* formula: w*^3 = g * z_i * H / (T * (1 + 0.6 q) * rho * c_p)
# where (1+0.6q) sits in the denominator because T_v = T(1+0.61q) (virtual
# temperature) is the correct buoyancy reference. Same convention as z/L.



function from_datetime_to_sec(set_time, tz)
    zoned    = ZonedDateTime.(set_time, tz"UTC")
    dt_local = DateTime.(astimezone.(zoned, tz))

    true_year  = year.(dt_local)
    true_month = month.(dt_local)
    true_day   = day.(dt_local)

    midnight_local = DateTime.(true_year, true_month, true_day)
    true_time_sec  = Dates.value.(dt_local .- midnight_local) ./ 1000

    return true_time_sec, true_day, true_month, true_year
end


function read_ecor_data_for_one_file(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    all_info = []
    try
        time_in_datetime = dataset["time"][:]
        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        for ind in 1:length(time_in_sec)
            u_star_one = dataset["ustar"][ind]   # Friction velocity m/s
            heat_one   = dataset["h"][ind]       # Sensible heat flux W/m^2
            mean_t_one = dataset["mean_t"][ind]  # Sonic temperature K
            rho_one    = dataset["rho"][ind]     # Moist air density kg/m^3
            mr_one     = dataset["mr"][ind]      # Mixing ratio kg/kg
            time_one   = time_in_sec[ind]
            push!(all_info, [day[ind], month[ind], year[ind], time_one,
                             u_star_one, heat_one, mean_t_one, rho_one, mr_one])
        end
    finally
        close(dataset)
    end

    return all_info
end


function main()
    tz = tz"America/Chicago"

    info_all_files = []

    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_ecor))
    for file in set_files
        println(file)
        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_ecor, file)
        info_one_file = read_ecor_data_for_one_file(file_new, file_date, tz)
        append!(info_all_files, info_one_file)
    end

    df = DataFrame(
        day    = Int[r[1] for r in info_all_files],
        month  = Int[r[2] for r in info_all_files],
        year   = Int[r[3] for r in info_all_files],
        time_s = Float64[r[4] for r in info_all_files],
        ustar  = [r[5] for r in info_all_files],
        h      = [r[6] for r in info_all_files],
        mean_t = [r[7] for r in info_all_files],
        rho    = [r[8] for r in info_all_files],
        mr     = [r[9] for r in info_all_files],
    )
    output_file = joinpath(PATH_output_txt, "turbulent_regime.csv")
    println(output_file)
    CSV.write(output_file, df)
end


function compute_wstar()
    df_ecor = CSV.read(joinpath(PATH_output_txt, "turbulent_regime.csv"),                DataFrame)
    df_pbl  = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"), DataFrame)

    # Pre-group PBL entries by (day, month, year) for fast nearest-time lookup
    pbl_by_day = Dict{Tuple{Int,Int,Int}, Vector{Tuple{Float64, Union{Missing,Float64}}}}()
    for i in 1:nrow(df_pbl)
        key = (df_pbl.day[i], df_pbl.month[i], df_pbl.year[i])
        push!(get!(pbl_by_day, key, Tuple{Float64, Union{Missing,Float64}}[]),
              (Float64(df_pbl.time_s[i]), df_pbl.pbl_filtered[i]))
    end

    n = nrow(df_ecor)
    wstar = Vector{Union{Missing, Float64}}(undef, n)

    for i in 1:n
        d, m, y, t = df_ecor[i,:day], df_ecor[i,:month], df_ecor[i,:year], df_ecor[i,:time_s]
        H, T, rho, r = df_ecor[i,:h], df_ecor[i,:mean_t], df_ecor[i,:rho], df_ecor[i,:mr]

        best_zi = missing
        best_dt = Inf
        candidates = get(pbl_by_day, (d, m, y), nothing)
        if candidates !== nothing
            for (ts, zi) in candidates
                dt = abs(ts - Float64(t))
                if dt < best_dt && dt <= TIME_GAP_S
                    best_dt = dt
                    best_zi = zi
                end
            end
        end

        if ismissing(best_zi) || ismissing(H) || ismissing(T) || ismissing(rho) || ismissing(r) || H <= 0
            wstar[i] = missing
        else
            # w*^3 = g * z_i * H / (T_v * rho * c_p), where T_v = T * (1 + 0.6 q)
            arg = G * best_zi * H / (T * (1 + 0.6 * r) * rho * CP)
            wstar[i] = cbrt(arg)
        end
    end

    df_out = DataFrame(
        day    = df_ecor.day,
        month  = df_ecor.month,
        year   = df_ecor.year,
        time_s = df_ecor.time_s,
        ustar  = df_ecor.ustar,
        h      = df_ecor.h,
        mean_t = df_ecor.mean_t,
        rho    = df_ecor.rho,
        mr     = df_ecor.mr,
        wstar  = wstar,
    )
    output_file = joinpath(PATH_output_txt, "turbulent_regime_full.csv")
    println(output_file)
    CSV.write(output_file, df_out)
end


function align_data_with_lidar()
    input_data_time = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"),             DataFrame)
    input_data_stb  = CSV.read(joinpath(PATH_output_txt, "turbulent_regime_full.csv"),  DataFrame)
    println(size(input_data_time))

    time_day_keys = [(input_data_time[i,:day], input_data_time[i,:month], input_data_time[i,:year]) for i in 1:nrow(input_data_time)]
    unique_days = unique(time_day_keys)
    interpolated_stb = []

    count, count_2 = 0, 0
    count_days, count_days_2 = [], []

    for day in unique_days
        day_lidar, m_lidar, y_lidar = day

        idxs_time = findall(x -> x[1]==day_lidar && x[2]==m_lidar && x[3]==y_lidar, time_day_keys)
        lidar_times = input_data_time[idxs_time, :time_start_s]

        mask_day = (input_data_stb[!,:day]   .== day_lidar) .&
                   (input_data_stb[!,:month] .== m_lidar)   .&
                   (input_data_stb[!,:year]  .== y_lidar)

        stb_times    = input_data_stb[mask_day, :time_s]
        values_ustar = input_data_stb[mask_day, :ustar]
        values_heat  = input_data_stb[mask_day, :h]
        values_wstar = input_data_stb[mask_day, :wstar]

        if isempty(stb_times)
            for t_lidar in lidar_times
                push!(interpolated_stb,
                      [day_lidar, m_lidar, y_lidar, t_lidar, missing, missing, missing])
                count += 1
                push!(count_days, [day_lidar, m_lidar, y_lidar])
            end
            continue
        end

        for t_lidar in lidar_times
            time_diffs = abs.(stb_times .- t_lidar)
            if minimum(time_diffs) <= TIME_GAP_S
                idx_closest = argmin(time_diffs)
                ustar_val = values_ustar[idx_closest]
                heat_val  = values_heat[idx_closest]
                wstar_val = values_wstar[idx_closest]
                push!(interpolated_stb,
                      [day_lidar, m_lidar, y_lidar, t_lidar,
                       ustar_val, heat_val, wstar_val])
            else
                push!(interpolated_stb,
                      [day_lidar, m_lidar, y_lidar, t_lidar, missing, missing, missing])
                count_2 += 1
                push!(count_days_2, [day_lidar, m_lidar, y_lidar])
            end
        end
    end

    unique_days   = unique(count_days)
    unique_days_2 = unique(count_days_2)

    println("missing days ", count, ", which is ", round(count/length(interpolated_stb)*100),"% from all data set,  or ", length(unique_days), " days")
    println("missing time gap ", count_2, ", which is ", round(count_2/length(interpolated_stb)*100),"% from all data set,  or ", length(unique_days_2), " days")
    println(size(interpolated_stb))

    df = DataFrame(
        day    = Int[r[1] for r in interpolated_stb],
        month  = Int[r[2] for r in interpolated_stb],
        year   = Int[r[3] for r in interpolated_stb],
        time_s = Float64[r[4] for r in interpolated_stb],
        ustar  = [r[5] for r in interpolated_stb],
        h      = [r[6] for r in interpolated_stb],
        wstar  = [r[7] for r in interpolated_stb],
    )
    output_path = joinpath(PATH_output_txt, "regime_aligned_with_lidar.csv")
    CSV.write(output_path, df)
    println(output_path)
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


function plot_regime_diagram_h()
    df = CSV.read(joinpath(PATH_output_txt, "regime_aligned_with_lidar.csv"), DataFrame)
    time_h = df.time_s ./ 3600
    df = df[(time_h .>= DAYTIME_LO) .& (time_h .<= DAYTIME_HI), :]

    plot(size=(750, 550), left_margin=10mm, bottom_margin=10mm, right_margin=10mm,
         xguidefont=18, yguidefont=18, ytickfont=font(12), xtickfont=font(12), legendfont=font(12))
    scatter!(df.ustar, df.h, color=:yellow, label=false, marker=:circle,
             xlabel=L"u^* (m/s)", ylabel=L"H (W/m^2)",
             alpha=0.1, legend=:topright, markersize=3,
             ylim=(0, 350), xlims=(0, 0.8))
    hline!([0, HEAT_LEVEL_1, HEAT_LEVEL_2], linestyle=:dash, color=:gray, label="")
    vline!([USTAR_LEVEL_1, USTAR_LEVEL_2],  linestyle=:dash, color=:gray, label="")

    savefig(joinpath(PATH_output_png, "turbulent_regime_h_12to15LT.png"))
end


function plot_regime_diagram_w()
    df = CSV.read(joinpath(PATH_output_txt, "regime_aligned_with_lidar.csv"), DataFrame)
    time_h = df.time_s ./ 3600
    df = df[(time_h .>= DAYTIME_LO) .& (time_h .<= DAYTIME_HI), :]

    plot(size=(750, 550), left_margin=10mm, bottom_margin=10mm, right_margin=10mm,
         xguidefont=18, yguidefont=18, ytickfont=font(12), xtickfont=font(12), legendfont=font(12))
    scatter!(df.ustar, df.wstar, color=:yellow, label=false, marker=:circle,
             xlabel=L"u^* (m/s)", ylabel=L"w^* (m/s)",
             alpha=0.1, legend=:topright, markersize=3,
             ylim=(0, 3.0), xlims=(0, 0.8))
    hline!([0, WSTAR_LEVEL_1, WSTAR_LEVEL_2], linestyle=:dash, color=:gray, label="")
    vline!([USTAR_LEVEL_1, USTAR_LEVEL_2],    linestyle=:dash, color=:gray, label="")

    savefig(joinpath(PATH_output_png, "turbulent_regime_w_12to15LT.png"))
end


function plot_regime_diagram_by_season(; var::Symbol, ylab, ylim,
                                          level_1, level_2, output_filename)
    df = CSV.read(joinpath(PATH_output_txt, "regime_aligned_with_lidar.csv"), DataFrame)
    df.time_h = df.time_s ./ 3600
    df = df[(df.time_h .>= DAYTIME_LO) .& (df.time_h .<= DAYTIME_HI), :]
    df.season = season_of.(df.month)

    seasons = ["DJF", "MAM", "JJA", "SON"]
    panels = []

    for s in seasons
        df_s = df[df.season .== s, :]

        p = plot(left_margin=10mm, bottom_margin=10mm, right_margin=10mm,
                 xguidefont=14, yguidefont=14, ytickfont=font(11), xtickfont=font(11),
                 legendfont=font(11), title=s, titlefont=font(16))
        scatter!(p, df_s.ustar, df_s[!, var],
                 color=:yellow, label=false, marker=:circle,
                 xlabel=L"u^* (m/s)", ylabel=ylab,
                 alpha=0.2, legend=false, markersize=3,
                 ylim=ylim, xlims=(0, 0.8))

        hline!(p, [0, level_1, level_2], linestyle=:dash, color=:gray, label="")
        vline!(p, [USTAR_LEVEL_1, USTAR_LEVEL_2], linestyle=:dash, color=:gray, label="")

        push!(panels, p)
    end

    plt = plot(panels..., layout=(2, 2), size=(1400, 1100),
               left_margin=12mm, bottom_margin=12mm)
    savefig(plt, joinpath(PATH_output_png, output_filename))
    println(joinpath(PATH_output_png, output_filename))
end


function plot_regime_diagram_h_by_season()
    plot_regime_diagram_by_season(
        var=:h, ylab=L"H (W/m^2)", ylim=(0, 350),
        level_1=HEAT_LEVEL_1, level_2=HEAT_LEVEL_2,
        output_filename="turbulent_regime_h_by_season_12to15LT.png",
    )
end


function plot_regime_diagram_w_by_season()
    plot_regime_diagram_by_season(
        var=:wstar, ylab=L"w^* (m/s)", ylim=(0, 3.0),
        level_1=WSTAR_LEVEL_1, level_2=WSTAR_LEVEL_2,
        output_filename="turbulent_regime_w_by_season_12to15LT.png",
    )
end



main()
compute_wstar()
align_data_with_lidar()
# plot_regime_diagram_h()
# plot_regime_diagram_w()
# plot_regime_diagram_h_by_season()
# plot_regime_diagram_w_by_season()