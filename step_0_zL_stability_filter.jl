using NCDatasets
using Dates
using TimeZones
using Statistics
using DataFrames
using CSV
using DelimitedFiles

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Stability calculation parameters
const KARMAN  = 0.40    # von Karman constant
const G       = 9.81    # gravity [m s-2]
const Z_ECOR  = 3.0     # ECOR measurement height [m]
const CP      = 1005.5  # specific heat capacity [J kg-1 K-1]

const TIME_GAP_S = 1 * 3600   # max time difference (s) between lidar and ECOR sample


# Computes the dimensionless stability parameter z/L (height over Obukhov
# length) from the eddy-covariance product hou30ecorM1 and aligns it to the
# lidar grid.
#   main                  : reads each ECOR file, computes z/L per sample,
#                           writes z_L_stability_conditions.csv.
#   align_data_with_lidar : matches each lidar block timestamp to the nearest
#                           z/L value within TIME_GAP_S, writes
#                           z_L_stability_cond_aligned_with_lidar.csv.
#
# Per ARM ECOR handbook (DOE/SC-ARM-TR-052):
#   - 30-min cadence
#   - measurement height = 3 m AGL
#   - ustar, h, mean_t, rho, mr are b1-level outputs
#
# Stability formulation: z/L using virtual temperature T_v = T_0(1+0.61q),
# placed in the denominator (since T_v sits in the denominator of 1/L).



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
            time_one   = time_in_sec[ind]
            mean_t_one = dataset["mean_t"][ind]  # Mean t temperature (sonic anemometer) K
            rho_one    = dataset["rho"][ind]     # Moist air density kg/m^3
            mr_one     = dataset["mr"][ind]      # Mixing ratio kg/kg

            # Skip if any input is missing or ustar is non-positive
            if ismissing(u_star_one) || ismissing(heat_one) || ismissing(mean_t_one) ||
               ismissing(rho_one)    || ismissing(mr_one)   || u_star_one <= 0
                continue
            end

            # z/L = -z g k H / (u*^3 * T_v * rho * c_p)
            # where T_v = T_0 * (1 + 0.61 q) is virtual temperature
            inv_L_one = - G * KARMAN * heat_one /
                          (u_star_one^3 * mean_t_one * (1 + 0.6 * mr_one) * rho_one * CP)
            z_L_one   = Z_ECOR * inv_L_one

            push!(all_info, [day[ind], month[ind], year[ind], time_one, z_L_one])
        end
    finally
        close(dataset)
    end

    return all_info
end


function main()
    tz = tz"America/Chicago"  # Houston

    info_all_files = []

    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_ecor))
    for file in set_files
        println(file)

        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])
        println("day ", day, " month ", month, " year ", year)

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_ecor, file)
        info_one_file = read_ecor_data_for_one_file(file_new, file_date, tz)
        append!(info_all_files, info_one_file)
        println(size(info_all_files))
    end

    df = DataFrame(
        day    = Int[r[1] for r in info_all_files],
        month  = Int[r[2] for r in info_all_files],
        year   = Int[r[3] for r in info_all_files],
        time_s = Float64[r[4] for r in info_all_files],
        z_L    = Float64[r[5] for r in info_all_files],
    )
    output_file = joinpath(PATH_output_txt, "z_L_stability_conditions.csv")
    println(output_file)
    CSV.write(output_file, df)
end


function align_data_with_lidar()
    input_data_time = CSV.read(joinpath(PATH_output_txt, "time_lidar.csv"),                DataFrame)  # day / month / year / time_start_s / duration_s
    input_data_stb  = CSV.read(joinpath(PATH_output_txt, "z_L_stability_conditions.csv"),  DataFrame)  # day / month / year / time_s / z_L
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

        stb_times  = input_data_stb[mask_day, :time_s]
        stb_values = input_data_stb[mask_day, :z_L]

        # No ECOR samples at all for this day -> missing for every lidar timestamp
        if isempty(stb_times)
            for t_lidar in lidar_times
                push!(interpolated_stb, [day_lidar, m_lidar, y_lidar, t_lidar, missing])
                count += 1
                push!(count_days, [day_lidar, m_lidar, y_lidar])
            end
            continue
        end

        for t_lidar in lidar_times
            time_diffs = abs.(stb_times .- t_lidar)
            if minimum(time_diffs) <= TIME_GAP_S
                idx_closest = argmin(time_diffs)
                stb_set1 = stb_values[idx_closest]
                push!(interpolated_stb, [day_lidar, m_lidar, y_lidar, t_lidar, stb_set1])
            else
                push!(interpolated_stb, [day_lidar, m_lidar, y_lidar, t_lidar, missing])
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
        z_L    = [r[5] for r in interpolated_stb],
    )
    output_path = joinpath(PATH_output_txt, "z_L_stability_cond_aligned_with_lidar.csv")
    CSV.write(output_path, df)
    println(output_path)
end


main()
align_data_with_lidar()