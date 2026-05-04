using NCDatasets
using Dates
using TimeZones
using Statistics
using DataFrames
using CSV
using DelimitedFiles
using Plots
using Colors
using Measures
using LaTeXStrings

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Binning parameters
const N_BINS_DAY = 24 * 2   # 30-min bins for the diurnal cycle


# Surface meteorology from houmetM1: wind speed and direction at the ground.
# Per ARM MET handbook (DOE/SC-ARM-TR-086):
#   - 1-minute averaged data
#   - wind sensor (RM Young 05103/05106) at 10 m standard height
#   - wspd_arith_mean: arithmetic mean wind speed
#   - wdir_vec_mean:   vector-mean wind direction (treats wind as 2D vector)
#
# main          : reads each MET file, extracts wspd_arith_mean and
#                 wdir_vec_mean per timestamp, writes wind_surface_MET.csv.
# ave_wind_over_season : groups by season (DJF/MAM/JJA/SON), bins by
#                 hour-of-day, plots mean +/- std diurnal cycle. Wind speed
#                 uses arithmetic mean; wind direction uses circular mean
#                 (sin/cos averaging) because direction is a circular
#                 variable. Output direction band is wrapped to [0, 360].
#                 Writes wind_<var>_<season>.csv per panel and a single
#                 PNG figure with the four seasons.

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


function read_met_data_for_one_file(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    all_info = []

    try
        time_in_datetime = dataset["time"][:]
        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        for ind in 1:length(time_in_sec)
            wind_spd = dataset["wspd_arith_mean"][ind]
            wind_dir = dataset["wdir_vec_mean"][ind]
            time_one = time_in_sec[ind]
            push!(all_info, [day[ind], month[ind], year[ind], time_one, wind_spd, wind_dir])
        end
    finally
        close(dataset)
    end
    return all_info
end


function main()
    tz = tz"America/Chicago"  # Houston

    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_met))
    info_all_files = []
    for file in set_files
        println("processing file: $file")
        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])
        println("day ", day, " month ", month, " year ", year)

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_met, file)
        info_one_file = read_met_data_for_one_file(file_new, file_date, tz)
        append!(info_all_files, info_one_file)
        println(size(info_all_files))
    end

    df = DataFrame(
        day        = Int[r[1] for r in info_all_files],
        month      = Int[r[2] for r in info_all_files],
        year       = Int[r[3] for r in info_all_files],
        time_s     = Float64[r[4] for r in info_all_files],
        wind_spd   = [r[5] for r in info_all_files],
        wind_dir   = [r[6] for r in info_all_files],
    )
    output_file = joinpath(PATH_output_txt, "wind_surface_MET.csv")
    CSV.write(output_file, df)
end


# Circular mean / std for wind direction (in degrees, [0, 360))
function circular_mean_std(angles_deg)
    valid = collect(skipmissing(angles_deg))
    valid = filter(x -> !isnan(x), valid)
    if isempty(valid)
        return NaN, NaN
    end
    rad = valid .* (pi / 180)
    s = mean(sin.(rad))
    c = mean(cos.(rad))
    mean_rad = atan(s, c)
    mean_deg = mod(mean_rad * 180 / pi, 360)
    R = sqrt(s^2 + c^2)
    # Guard against R = 0 (perfectly bimodal directions) -> std_deg = NaN
    std_deg = R > 0 ? sqrt(-2 * log(R)) * 180 / pi : NaN
    return mean_deg, std_deg
end


function divide_by_bins(ind, time, val; circular::Bool=false)
    time = time[ind]
    val  = val[ind]

    interval_width = 24 / N_BINS_DAY
    bin_indices = floor.(Int, time / interval_width) .+ 1

    avg = Vector{Float64}(undef, N_BINS_DAY)
    sd  = Vector{Float64}(undef, N_BINS_DAY)
    for i in 1:N_BINS_DAY
        v = val[bin_indices .== i]
        if isempty(v)
            avg[i] = NaN
            sd[i]  = NaN
        elseif circular
            avg[i], sd[i] = circular_mean_std(v)
        else
            avg[i] = mean(skipmissing(v))
            sd[i]  = std(skipmissing(v))
        end
    end

    bin_edges = 0:interval_width:24
    time_bins = bin_edges[1:end-1] .+ interval_width / 2

    if circular
        # Wrap the +/- std band into [0, 360] so values are valid wind directions.
        # Note: when the mean is near 0/360, the wrapped band can look discontinuous;
        # this is a true visualisation artefact of plotting a circular variable on
        # a linear axis, not a numeric error.
        upper = mod.(avg .+ sd, 360)
        lower = mod.(avg .- sd, 360)
    else
        upper = avg .+ sd
        lower = avg .- sd
    end

    return [time_bins, avg, upper, lower]
end


function ave_wind_over_season(var::Symbol)
    info_w = CSV.read(joinpath(PATH_output_txt, "wind_surface_MET.csv"), DataFrame)
    println(size(info_w))

    months = info_w[!, :month]
    time_w = info_w[!, :time_s] ./ 3600   # hours, for binning
    val_w  = info_w[!, var]

    ind_winter = (months .== 12) .| (months .== 1) .| (months .== 2)
    ind_spring = (months .== 3)  .| (months .== 4) .| (months .== 5)
    ind_summer = (months .== 6)  .| (months .== 7) .| (months .== 8)
    ind_fall   = (months .== 9)  .| (months .== 10).| (months .== 11)

    is_dir = (var == :wind_dir)

    info_winter = divide_by_bins(ind_winter, time_w, val_w; circular = is_dir)
    info_spring = divide_by_bins(ind_spring, time_w, val_w; circular = is_dir)
    info_summer = divide_by_bins(ind_summer, time_w, val_w; circular = is_dir)
    info_fall   = divide_by_bins(ind_fall,   time_w, val_w; circular = is_dir)

    ylab   = is_dir ? L"Wind\ direction\ (deg)" : L"Wind\ (m/s)"
    ylimit = is_dir ? (0, 360) : (0, 7)
    suffix = is_dir ? "dir" : "spd"

    colors = cgrad([:purple, :green], 4, categorical = true)
    p = plot(layout=(1, 4), size=(1600, 400),
             left_margin=18mm, bottom_margin=10mm,
             xguidefont=18, yguidefont=18,
             ytickfont=font(12), xtickfont=font(12),
             legendfont=font(12), xlim=[-0.2, 24.2])

    panels = [(1, info_winter, "winter", "e"),
              (2, info_spring, "spring", "f"),
              (3, info_summer, "summer", "g"),
              (4, info_fall,   "fall",   "h")]

    for (k, info, title_str, label_str) in panels
        scatter!(p[k], info[1], info[2], label=false,
                 xlabel=L"Time\ (h)",
                 ylabel = (k == 1 ? ylab : ""),
                 lw=2, marker=:circle, color=colors[k],
                 legend=:topright, ylim=ylimit,
                 title=title_str, titlefont=font(18))
        plot!(p[k], info[1], info[3],
              fill_between=(info[4], info[3]),
              fillalpha=0.2, color=colors[k], label=false, lw=0)
        hline!(p[k], [0], color=:black, lw=1, linestyle=:dash, label=false)
        annotate!(p[k], (1, ylimit[2]*0.95, text(label_str, :left, 16)))
    end

    png_path = joinpath(PATH_output_png, "windD_over_season_$(suffix).png")
    println(png_path)
    savefig(png_path)

    for (info, season) in [(info_winter,"winter"), (info_spring,"spring"), (info_summer,"summer"), (info_fall,"fall")]
        df_out = DataFrame(time_h = info[1], avg = info[2], upper = info[3], lower = info[4])
        CSV.write(joinpath(PATH_output_txt, "wind_$(suffix)_$(season).csv"), df_out)
    end
end



# STEP 0
# run to collect data for whole campaign
main()

# plot diurnal pattern over the season
ave_wind_over_season(:wind_spd)
ave_wind_over_season(:wind_dir)