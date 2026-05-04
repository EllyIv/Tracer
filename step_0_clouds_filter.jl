using NCDatasets
using Dates
using TimeZones
using DataFrames
using CSV
using Plots
using Distributed

addprocs(64)


@everywhere using NCDatasets, Dates, TimeZones, DataFrames, CSV
@everywhere include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Cloud-base alignment criteria
@everywhere const TIME_GAP_H = 1       # max time difference (h) between lidar and cloud measurement
@everywhere const CBH_FACTOR = 0.9f0   # cloud base height correction factor


# Reads cloud-base height (CBH) from the Doppler-lidar profile-stats files,
# both DL and ceilometer estimates, scales them by CBH_FACTOR, and writes
# clouds.csv. Then aligns each lidar block timestamp (from time_lidar.csv) to
# the nearest cloud measurement within TIME_GAP_H hours and writes
# clouds_aligned_with_lidar_dl.csv and clouds_aligned_with_lidar_ceil.csv.
# Finally compares the two instruments via histogram.
#
# NOTE on cbh = missing semantics (per ARM handbook DOE/SC-ARM-TR-149):
#   missing CBH means "no cloud was detected during the averaging period"
#   (i.e., clear sky), NOT "instrument was off" or "no measurement available".
#   Downstream code that uses CBH as a height ceiling should interpret
#   missing as "no cloud cap to apply" and fall back to PBL/RH-based ceilings.


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


function align_data_with_lidar(col_cld::Symbol, title)
    input_data_time = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"), DataFrame)  # day / month / year / time_start_s / duration_s
    input_data_cld  = CSV.read(joinpath(PATH_output_txt, "clouds.csv"), DataFrame)      # day / month / year / time_s / cbh_dl / cbh_ceil
    println(size(input_data_time))

    time_day_keys = [(input_data_time[i,:day], input_data_time[i,:month], input_data_time[i,:year]) for i in 1:nrow(input_data_time)]
    unique_days = unique(time_day_keys)
    interpolated_cld = []

    count, count_2 = 0, 0
    count_days, count_days_2 = [], []

    for day in unique_days
        day_lidar, m_lidar, y_lidar = day

        idxs_time = findall(x -> x[1]==day_lidar && x[2]==m_lidar && x[3]==y_lidar, time_day_keys)
        lidar_times = input_data_time[idxs_time, :time_start_s]

        mask_day = (input_data_cld[!,:day]   .== day_lidar) .&
                   (input_data_cld[!,:month] .== m_lidar)   .&
                   (input_data_cld[!,:year]  .== y_lidar)

        cld_times  = input_data_cld[mask_day, :time_s]
        cld_values = input_data_cld[mask_day, col_cld]

        # No cloud measurements at all for this day -> missing for every lidar timestamp
        if isempty(cld_times)
            for t_lidar in lidar_times
                push!(interpolated_cld, [day_lidar, m_lidar, y_lidar, t_lidar, missing])
                count += 1
                push!(count_days, [day_lidar, m_lidar, y_lidar])
            end
            continue
        end

        for t_lidar in lidar_times
            time_diffs = abs.(cld_times .- t_lidar)
            if minimum(time_diffs) <= TIME_GAP_H * 3600
                # Closest cloud measurement within tolerance
                idx_closest = argmin(time_diffs)
                cld_val = cld_values[idx_closest]
                # cld_val may be missing -> clear sky at that time (NOT a data gap)
                push!(interpolated_cld, [day_lidar, m_lidar, y_lidar, t_lidar, cld_val])
                if ismissing(cld_val)
                    count_2 += 1
                    push!(count_days_2, [day_lidar, m_lidar, y_lidar])
                end
            else
                # No cloud measurement within tolerance -> true data gap
                push!(interpolated_cld, [day_lidar, m_lidar, y_lidar, t_lidar, missing])
                count_2 += 1
                push!(count_days_2, [day_lidar, m_lidar, y_lidar])
            end
        end
    end
    unique_days   = unique(count_days)
    unique_days_2 = unique(count_days_2)
    println(unique_days)

    println("missing days ", count, ", which is ", round(count/length(interpolated_cld)*100, digits = 2),"% from all data set,  or ", length(unique_days), " days")
    println("missing time gap ", count_2, ", which is ", round(count_2/length(interpolated_cld)*100),"% from all data set,  or ", length(unique_days_2), " days")
    println(size(interpolated_cld))

    df = DataFrame(
        day      = Int[r[1] for r in interpolated_cld],
        month    = Int[r[2] for r in interpolated_cld],
        year     = Int[r[3] for r in interpolated_cld],
        time_s   = Float64[r[4] for r in interpolated_cld],
        cbh      = [r[5] for r in interpolated_cld],
    )
    output_path = joinpath(PATH_output_txt, "clouds_aligned_with_lidar" * title * ".csv")
    CSV.write(output_path, df)
    println(output_path)
end


@everywhere function read_ceil_data(PATH_file, file_date::Date, tz)
    println(PATH_file)
    dataset = NCDataset(PATH_file, "r")
    info_one_file = []

    time_in_datetime = dataset["time"]
    time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)

    cld1 = dataset["dl_cbh"]    # Doppler lidar CBH (missing = clear sky)
    cld2 = dataset["ceil_cbh"]  # Ceilometer CBH    (missing = clear sky)

    for i in eachindex(time_in_sec)
        cldh_val_1 = if !ismissing(cld1[i])
            CBH_FACTOR * Float32(cld1[i])
        else
            missing
        end

        cldh_val_2 = if !ismissing(cld2[i])
            CBH_FACTOR * Float32(cld2[i])
        else
            missing
        end

        push!(info_one_file, [day[i], month[i], year[i], time_in_sec[i], cldh_val_1, cldh_val_2])
    end

    close(dataset)

    return info_one_file
end


@everywhere function process_file_ceil(file, tz)
    file_parts = split(basename(file), ".")
    year  = parse(Int, file_parts[3][1:4])
    month = parse(Int, file_parts[3][5:6])
    day   = parse(Int, file_parts[3][7:8])

    file_date = Date(year, month, day)
    file_path = joinpath(PATH_folder_clouds, file)
    return read_ceil_data(file_path, file_date, tz)
end


function cloud_all()
    tz = tz"America/Chicago"

    set_files = filter(x -> endswith(x, ".nc"), readdir(PATH_folder_clouds))
    println("there are ", length(set_files), " files in your folder")

    clouds_from_files = pmap(file -> process_file_ceil(file, tz), set_files)
    all_clouds = reduce(vcat, clouds_from_files)

    df = DataFrame(
        day      = Int[r[1] for r in all_clouds],
        month    = Int[r[2] for r in all_clouds],
        year     = Int[r[3] for r in all_clouds],
        time_s   = Float64[r[4] for r in all_clouds],
        cbh_dl   = [r[5] for r in all_clouds],
        cbh_ceil = [r[6] for r in all_clouds],
    )
    output_file = joinpath(PATH_output_txt, "clouds.csv")
    CSV.write(output_file, df)
    println(output_file)
end


function compare_ceil_vs_dl()
    df_dl   = CSV.read(joinpath(PATH_output_txt, "clouds_aligned_with_lidar_dl.csv"),   DataFrame)
    df_ceil = CSV.read(joinpath(PATH_output_txt, "clouds_aligned_with_lidar_ceil.csv"), DataFrame)

    cldh_1 = sort(skipmissing(df_dl.cbh)   |> collect)
    cldh_2 = sort(skipmissing(df_ceil.cbh) |> collect)

    p = histogram(cldh_1, bins=20, alpha=0.5, label="clouds DL")
    histogram!(cldh_2, bins=20, alpha=0.5, label="clouds CEIL")
    png(p, joinpath(PATH_output_png, "cloud_base.png"))
end


function compare_ceil_vs_dl_month()
    df_dl   = CSV.read(joinpath(PATH_output_txt, "clouds_aligned_with_lidar_dl.csv"),   DataFrame)
    df_ceil = CSV.read(joinpath(PATH_output_txt, "clouds_aligned_with_lidar_ceil.csv"), DataFrame)

    plots = []

    for m in 1:12
        cldh_dl = collect(skipmissing(df_dl[df_dl.month .== m, :cbh]))
        cldh_ceil = collect(skipmissing(df_ceil[df_ceil.month .== m, :cbh]))

        p = histogram(
            cldh_dl,
            bins = 20,
            alpha = 0.5,
            label = "DL",
            xlabel = "CBH (m)",
            ylabel = "Count",
            legend = :topright
        )

        histogram!(
            p,
            cldh_ceil,
            bins = 20,
            alpha = 0.5,
            label = "CEIL"
        )

        push!(plots, p)
    end

    panel = plot(
        plots...,
        layout = (3, 4),
        size = (1200, 900),
        )
    png(panel, joinpath(PATH_output_png, "cloud_base_month.png"))
end


cloud_all()
align_data_with_lidar(:cbh_dl,   "_dl")
align_data_with_lidar(:cbh_ceil, "_ceil")

compare_ceil_vs_dl()
compare_ceil_vs_dl_month()


# (32179, 5)
# Any[]
# missing days 0, which is 0.0% from all data set,  or 0 days
# missing time gap 76, which is 0.0% from all data set,  or 3 days
# (32179,)