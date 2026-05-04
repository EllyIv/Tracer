using NCDatasets
using Dates
using TimeZones
using DataFrames
using CSV
using Statistics
using Plots


include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Precipitation alignment criteria
const TIME_GAP_S = 1.0 * 3600   # ±1 h, matches Petters et al. (2024) and paper text


# Builds a precipitation record aligned to the lidar time grid.
# main_part1: reads the interpolated-sonde product (houinterpolatedsondeM1),
#             extracts the precip field (mm, 1-min cadence), writes precipitation.csv.
# main_part2: reads the two ARM Best-Estimate (houarmbeatmM1) files that
#             cover the analysis period, extracts precip_rate_sfc (mm/hr,
#             hourly cadence), writes precipitation_2.csv.
# align_data_with_lidar: concatenates both precip sources and matches each
#             lidar block timestamp to the nearest precip value within
#             TIME_GAP_S; writes prc_aligned_with_lidar.csv.
#
# NOTE on precip = missing semantics:
#   For both INTERPSONDE and ARMBE, missing means "no measurement" (data gap).
#   A no-rain period has precip = 0, NOT missing. Downstream code that filters
#   on "non-zero precipitation" must treat missing separately from zero.
#
# NOTE on units:
#   INTERPSONDE precip is in mm (1-min accumulation).
#   ARMBE precip_rate_sfc is in mm/hr.
#   These are concatenated for the simple "is it raining?" (>0) check, which
#   works regardless of units. Do NOT use absolute precip values from the
#   merged file for quantitative work.


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


# UTC DateTime -> local seconds-since-midnight + local date components
function utc_datetime_to_local_sec(dt_utc, tz)
    zoned    = ZonedDateTime(dt_utc, tz"UTC")
    dt_local = DateTime(astimezone(zoned, tz))
    d = Date(dt_local)
    sec = Dates.value(dt_local - DateTime(d)) / 1000   # ms -> s
    return sec, day(d), month(d), year(d)
end


function read_sonde_data_for_one_file(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    all_info = []
    try
        time_in_datetime = dataset["time"][:]
        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        precip_data = dataset["precip"]
        for i in 1:length(time_in_sec)
            push!(all_info, [day[i], month[i], year[i], time_in_sec[i], precip_data[i]])
        end
    finally
        close(dataset)
    end
    return all_info
end


function align_data_with_lidar()
    input_data_time  = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"),       DataFrame)  # day / month / year / time_start_s / duration_s
    input_data_prc_1 = CSV.read(joinpath(PATH_output_txt, "precipitation.csv"),    DataFrame)
    input_data_prc_2 = CSV.read(joinpath(PATH_output_txt, "precipitation_2.csv"),  DataFrame)

    input_data_prc = vcat(input_data_prc_1, input_data_prc_2)

    time_day_keys = [(input_data_time[i,:day], input_data_time[i,:month], input_data_time[i,:year]) for i in 1:nrow(input_data_time)]
    unique_days = unique(time_day_keys)
    interpolated_prc = []

    count, count_2 = 0, 0
    count_days, count_days_2 = [], []

    for day in unique_days
        day_lidar, m_lidar, y_lidar = day

        idxs_time = findall(x -> x[1]==day_lidar && x[2]==m_lidar && x[3]==y_lidar, time_day_keys)
        lidar_times = input_data_time[idxs_time, :time_start_s]

        mask_day = (input_data_prc[!,:day]   .== day_lidar) .&
                   (input_data_prc[!,:month] .== m_lidar)   .&
                   (input_data_prc[!,:year]  .== y_lidar)

        prc_times  = input_data_prc[mask_day, :time_s]
        prc_values = input_data_prc[mask_day, :precip]

        # No precip measurements at all for this day -> missing for every lidar timestamp
        if isempty(prc_times)
            for t_lidar in lidar_times
                push!(interpolated_prc, [day_lidar, m_lidar, y_lidar, t_lidar, missing])
                count += 1
                push!(count_days, [day_lidar, m_lidar, y_lidar])
            end
            continue
        end

        for t_lidar in lidar_times
            time_diffs = abs.(prc_times .- t_lidar)
            if minimum(time_diffs) <= TIME_GAP_S
                idx_closest = argmin(time_diffs)
                prc_val = prc_values[idx_closest]
                # prc_val may be missing -> data gap (NOT "no rain"!)
                push!(interpolated_prc, [day_lidar, m_lidar, y_lidar, t_lidar, prc_val])
            else
                # No precip measurement within tolerance -> data gap
                push!(interpolated_prc, [day_lidar, m_lidar, y_lidar, t_lidar, missing])
                count_2 += 1
                push!(count_days_2, [day_lidar, m_lidar, y_lidar])
            end
        end
    end
    unique_days   = unique(count_days)
    unique_days_2 = unique(count_days_2)

    println("missing days ", count, ", which is ", round(count/length(interpolated_prc)*100),"% from all data set,  or ", length(unique_days), " days")
    println("missing time gap ", count_2, ", which is ", round(count_2/length(interpolated_prc)*100),"% from all data set,  or ", length(unique_days_2), " days")
    println(size(interpolated_prc))

    df = DataFrame(
        day    = Int[r[1] for r in interpolated_prc],
        month  = Int[r[2] for r in interpolated_prc],
        year   = Int[r[3] for r in interpolated_prc],
        time_s = Float64[r[4] for r in interpolated_prc],
        precip = [r[5] for r in interpolated_prc],
    )
    output_path = joinpath(PATH_output_txt, "prc_aligned_with_lidar.csv")
    CSV.write(output_path, df)
    println(output_path)
end


function main_part1()
    tz = tz"America/Chicago"

    info_precip_all_days = []

    set_files = filter(x -> endswith(x, ".nc"), readdir(PATH_folder_sonde))
    for file in set_files
        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])
        println("day ", day, " month ", month, " year ", year)

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_sonde, file)
        info_one_file = read_sonde_data_for_one_file(file_new, file_date, tz)
        append!(info_precip_all_days, info_one_file)
    end

    df = DataFrame(
        day    = Int[r[1] for r in info_precip_all_days],
        month  = Int[r[2] for r in info_precip_all_days],
        year   = Int[r[3] for r in info_precip_all_days],
        time_s = Float64[r[4] for r in info_precip_all_days],
        precip = [r[5] for r in info_precip_all_days],
    )
    output_file = joinpath(PATH_output_txt, "precipitation.csv")
    println(output_file)
    CSV.write(output_file, df)
end


function main_part2()
    # Only two files exist for the analysis period — read them explicitly.
    tz = tz"America/Chicago"

    file_1 = joinpath(PATH_folder_beatm, "houarmbeatmM1.c1.20211001.003000.nc")
    dataset_1 = NCDataset(file_1, "r")

    time_set_1   = dataset_1["time"][:]
    precip_set_1 = dataset_1["precip_rate_sfc"][:]
    new_precip = []
    for i in 1:length(time_set_1)
        sec, d, m, y = utc_datetime_to_local_sec(time_set_1[i], tz)
        push!(new_precip, [d, m, y, sec, precip_set_1[i]])
    end
    close(dataset_1)

    file_2 = joinpath(PATH_folder_beatm, "houarmbeatmM1.c1.20220101.003000.nc")
    dataset_2 = NCDataset(file_2, "r")
    time_set_2   = dataset_2["time"][:]
    precip_set_2 = dataset_2["precip_rate_sfc"][:]
    for i in 1:length(time_set_2)
        sec, d, m, y = utc_datetime_to_local_sec(time_set_2[i], tz)
        push!(new_precip, [d, m, y, sec, precip_set_2[i]])
    end
    close(dataset_2)

    df = DataFrame(
        day    = Int[r[1] for r in new_precip],
        month  = Int[r[2] for r in new_precip],
        year   = Int[r[3] for r in new_precip],
        time_s = Float64[r[4] for r in new_precip],
        precip = [r[5] for r in new_precip],
    )
    output_file = joinpath(PATH_output_txt, "precipitation_2.csv")
    println(output_file)
    CSV.write(output_file, df)
end


function check_precipitation_time_range()
    df1 = CSV.read(joinpath(PATH_output_txt, "precipitation.csv"), DataFrame)
    df2 = CSV.read(joinpath(PATH_output_txt, "precipitation_2.csv"), DataFrame)

    for (name, df) in [("precipitation.csv", df1), ("precipitation_2.csv", df2)]
        dt = DateTime.(
            df.year,
            df.month,
            df.day
        ) .+ Second.(round.(Int, df.time_s))

        println("========== $name ==========")
        println("Rows: ", nrow(df))
        println("Start: ", minimum(dt))
        println("End:   ", maximum(dt))
        println()
    end
end


function compare_precipitation_daily_mean_by_month()
    df1 = CSV.read(joinpath(PATH_output_txt, "precipitation.csv"), DataFrame)
    df2 = CSV.read(joinpath(PATH_output_txt, "precipitation_2.csv"), DataFrame)

    df1.file .= "precipitation.csv"
    df2.file .= "precipitation_2.csv"

    df = vcat(df1, df2)

    daily_mean = combine(
        groupby(df, [:file, :year, :month, :day]),
        :precip => (x -> mean(skipmissing(x))) => :daily_mean_precip
    )

    # Step 2:
    # if daily average > 0 → 1
    # if daily average = 0 → 0
    daily_mean.daily_binary =
        ifelse.(daily_mean.daily_mean_precip .> 0, 1, 0)

    plots = []

    for m in 1:12
        d1 = daily_mean[
            (daily_mean.month .== m) .&
            (daily_mean.file .== "precipitation.csv"), :
        ]

        d2 = daily_mean[
            (daily_mean.month .== m) .&
            (daily_mean.file .== "precipitation_2.csv"), :
        ]

        p = scatter(
            d1.day,
            d1.daily_binary,
            marker = :circle,
            label = "1",
            xlabel = "Day",
            ylabel = "Daily precip (0/1)",
            title = "Month $m",
            xticks = 1:31,
            ylims = (-0.1, 1.1)
        )

        scatter!(
            p,
            d2.day,
            d2.daily_binary,
            marker = :circle,
            label = "2",
            alpha = 0.5
        )

        push!(plots, p)
    end

    panel = plot(
        plots...,
        layout = (3, 4),
        size = (2000, 1200)
    )

    png(panel, joinpath(PATH_output_png, "precipitation_daily_mean_by_month.png"))
end


function compare_precipitation_match_heatmap_year()
    df1 = CSV.read(joinpath(PATH_output_txt, "precipitation.csv"), DataFrame)
    df2 = CSV.read(joinpath(PATH_output_txt, "precipitation_2.csv"), DataFrame)

    function hourly_binary(df)
        df.hour = floor.(Int, df.time_s ./ 3600)

        hourly = combine(
            groupby(df, [:year, :month, :day, :hour]),
            :precip => (x -> mean(skipmissing(x))) => :mean_precip
        )

        hourly.binary = ifelse.(hourly.mean_precip .> 0, 1, 0)

        return hourly[:, [:year, :month, :day, :hour, :binary]]
    end

    h1 = hourly_binary(df1)
    h2 = hourly_binary(df2)

    rename!(h1, :binary => :binary_1)
    rename!(h2, :binary => :binary_2)

    joined = innerjoin(
        h1, h2,
        on = [:year, :month, :day, :hour]
    )

    # match = 1, mismatch = 2
    joined.status = ifelse.(joined.binary_1 .== joined.binary_2, 1, 2)

    plots = []

    for m in 1:12
        df_m = joined[joined.month .== m, :]

        mat = fill(NaN, 24, 31)

        for r in eachrow(df_m)
            if 0 <= r.hour <= 23 && 1 <= r.day <= 31
                mat[r.hour + 1, r.day] = r.status
            end
        end

        p = heatmap(
            1:31,
            0:23,
            mat,
            xlabel = "Day",
            ylabel = "Hour",
            title = "Month $m",
            color = cgrad([:green, :red], 2, categorical=true),
            clims = (1, 2),
            colorbar = false
        )

        push!(plots, p)
    end

    panel = plot(
        plots...,
        layout = (3, 4),
        size = (1800, 1200),
    )

    png(panel, joinpath(PATH_output_png, "precipitation_heatmap_year.png"))
end


function compare_precipitation_source_availability_heatmap_year()
    df1 = CSV.read(joinpath(PATH_output_txt, "precipitation.csv"), DataFrame)
    df2 = CSV.read(joinpath(PATH_output_txt, "precipitation_2.csv"), DataFrame)

    function hourly_keys(df)
        df.hour = floor.(Int, df.time_s ./ 3600)

        hourly = combine(
            groupby(df, [:year, :month, :day, :hour]),
            nrow => :n_points
        )

        return hourly[:, [:year, :month, :day, :hour]]
    end

    h1 = hourly_keys(df1)
    h2 = hourly_keys(df2)

    h1.source1 .= true
    h2.source2 .= true

    joined = outerjoin(
        h1, h2,
        on = [:year, :month, :day, :hour]
    )

    joined.source1 = coalesce.(joined.source1, false)
    joined.source2 = coalesce.(joined.source2, false)

    # status:
    # 1 = only first source
    # 2 = only second source
    # 3 = both sources
    joined.status = ifelse.(
        joined.source1 .& .!joined.source2, 1,
        ifelse.(
            .!joined.source1 .& joined.source2, 2,
            3
        )
    )

    plots = []

    for m in 1:12
        df_m = joined[joined.month .== m, :]

        mat = fill(NaN, 24, 31)

        for r in eachrow(df_m)
            if 0 <= r.hour <= 23 && 1 <= r.day <= 31
                mat[r.hour + 1, r.day] = r.status
            end
        end

        p = heatmap(
            1:31,
            0:23,
            mat,
            xlabel = "Day",
            ylabel = "Hour",
            title = "Month $m",
            color = cgrad([:blue, :purple, :orange], 3, categorical=true),
            clims = (1, 3),
            colorbar = false
        )

        push!(plots, p)
    end

    panel = plot(
        plots...,
        layout = (3, 4),
        size = (1800, 1200)
    )

    png(panel, joinpath(PATH_output_png, "precipitation_source_heatmap_year.png"))
end


main_part1()
main_part2()
align_data_with_lidar()

compare_precipitation_daily_mean_by_month()
compare_precipitation_match_heatmap_year()
compare_precipitation_source_availability_heatmap_year()
# check_precipitation_time_range()