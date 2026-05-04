using NCDatasets
using Dates
using TimeZones
using Statistics
using DataFrames
using CSV

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")


# Builds two independent PBL-height records on hourly bins (local time).
#   pbl_ceil  : reads houceilpblhtM1.a0, ceilometer-derived PBL height
#               (bl_height_1), writes pbl_ceil.csv.
#   pbl_sonde : reads houpblhtsonde1mcfarlM1, four sounding-based PBL
#               estimates (Heffter, Liu-Liang, bulk-Richardson 0.25, bulk-
#               Richardson 0.5). Each sonde value is broadcast across all
#               hours of its day. Writes pbl_sonde.csv.
# Lidar-grid alignment is performed in a separate script.




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




function read_ceil_data(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    # variable_names = keys(dataset)
    # println("Variables in the dataset: ", variable_names)
    # println(size(dataset["bl_height_1"][:]))
    # sleep(100)

    try 
        # pbl1 = dataset["pbl_height_heffter"][:] # ONE VALUE PER FILE!!!
        pbl1 = dataset["bl_height_1"]
        # pbl2 = dataset["pbl_height_liu_liang"][:]
        time_in_datetime = dataset["time"]

        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        time_in_hr = time_in_sec ./ 3600

        # --- bin averaging every 1 hour ---
        bins = floor.(time_in_hr)
        unique_bins = sort(unique(bins))

        ave_time  = Float64[]
        ave_pbl   = Float64[]
        ave_day   = Int[]
        ave_month = Int[]
        ave_year  = Int[]

        for b in unique_bins
            mask = bins .== b
            vals = pbl1[mask]
            push!(ave_time,  b * 3600)  # convert back to seconds
            push!(ave_pbl,   mean(skipmissing(vals)))
            push!(ave_day,   day[findfirst(mask)])
            push!(ave_month, month[findfirst(mask)])
            push!(ave_year,  year[findfirst(mask)])
        end

        return [ave_day, ave_month, ave_year, ave_time, ave_pbl]
             
    finally 
        close(dataset)
    end
end


function read_sonde_data(PATH_file, file_date::Date, tz)
    dataset = NCDataset(PATH_file, "r")

    # variable_names = keys(dataset)
    # println("Variables in the dataset: ", variable_names)
    # println(size(dataset["bl_height_1"][:]))
    # sleep(100)

    try 
        # All four sonde-based PBL estimates are single-valued per file
        # (one sounding per day); broadcast across hourly bins below.
        pbl1 = dataset["pbl_height_heffter"][1]
        pbl2 = dataset["pbl_height_liu_liang"][:]
        pbl3 = dataset["pbl_height_bulk_richardson_pt25"][:]
        pbl4 = dataset["pbl_height_bulk_richardson_pt5"][:]

        time_in_datetime = dataset["time"]

        time_in_sec, day, month, year = from_datetime_to_sec(time_in_datetime, tz)
        time_in_hr = time_in_sec ./ 3600

        # --- bin averaging every 1 hour ---
        bins = floor.(time_in_hr)
        unique_bins = sort(unique(bins))

        ave_time  = Float64[]
        ave_pbl1, ave_pbl2, ave_pbl3, ave_pbl4 = [], [], [], []
        ave_day   = Int[]
        ave_month = Int[]
        ave_year  = Int[]
        
        for b in unique_bins
            mask = bins .== b

            push!(ave_time,  b * 3600)
            push!(ave_pbl1,  pbl1)
            push!(ave_pbl2,  pbl2[1])
            push!(ave_pbl3,  pbl3[1])
            push!(ave_pbl4,  pbl4[1])

            push!(ave_day,   day[findfirst(mask)])
            push!(ave_month, month[findfirst(mask)])
            push!(ave_year,  year[findfirst(mask)])
        end

        return [ave_day, ave_month, ave_year, ave_time, ave_pbl1, ave_pbl2, ave_pbl3, ave_pbl4]
             
    finally 
        close(dataset)
    end
end


# function read_sonde_interp_data(PATH_file, file_date::Date, tz)
#     # case-study tool, kept for reference (interpolated-sonde version of read_sonde_data)
# end


function pbl_ceil()
    tz = tz"America/Chicago"

    set_files = filter(x -> endswith(x, ".nc"), readdir(PATH_folder_ceil_pbl))
    println("there are ", length(set_files), " in your folder")
    info_all_files = []
    for file in set_files
        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])
        println(file_parts, " ", day, " ", month, " ", year)

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_ceil_pbl, file)
        pbl_one_file = read_ceil_data(file_new, file_date, tz)
        for i in 1:length(pbl_one_file[1])
            info = [pbl_one_file[1][i], pbl_one_file[2][i], pbl_one_file[3][i], pbl_one_file[4][i], pbl_one_file[5][i]]
            push!(info_all_files, info)
        end
    end 

    df = DataFrame(
        day    = Int[r[1] for r in info_all_files],
        month  = Int[r[2] for r in info_all_files],
        year   = Int[r[3] for r in info_all_files],
        time_s = Float64[r[4] for r in info_all_files],
        pbl_h  = [r[5] for r in info_all_files],
    )
    output_file = joinpath(PATH_output_txt, "pbl_ceil.csv")
    CSV.write(output_file, df)
    println(output_file)
end


function pbl_sonde()
    tz = tz"America/Chicago"

    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_sonde_pbl))
    println("there are ", length(set_files), " in your folder")
    info_all_files = []
    for file in set_files
        file_parts = split(basename(file), ".")
        year  = parse(Int, file_parts[3][1:4])
        month = parse(Int, file_parts[3][5:6])
        day   = parse(Int, file_parts[3][7:8])
        println(file_parts, " ", day, " ", month, " ", year)

        file_date = Date(year, month, day)
        file_new = joinpath(PATH_folder_sonde_pbl, file)
        pbl_one_file = read_sonde_data(file_new, file_date, tz)
        for i in 1:length(pbl_one_file[1])
            info = [pbl_one_file[1][i], pbl_one_file[2][i], pbl_one_file[3][i], pbl_one_file[4][i],
                    pbl_one_file[5][i], pbl_one_file[6][i], pbl_one_file[7][i], pbl_one_file[8][i]]
            push!(info_all_files, info)
        end
    end 

    df = DataFrame(
        day               = Int[r[1] for r in info_all_files],
        month             = Int[r[2] for r in info_all_files],
        year              = Int[r[3] for r in info_all_files],
        time_s            = Float64[r[4] for r in info_all_files],
        pbl_heffter       = [r[5] for r in info_all_files],
        pbl_liu_liang     = [r[6] for r in info_all_files],
        pbl_bulkri_pt25   = [r[7] for r in info_all_files],
        pbl_bulkri_pt5    = [r[8] for r in info_all_files],
    )
    output_file = joinpath(PATH_output_txt, "pbl_sonde.csv")
    CSV.write(output_file, df)
    println(output_file)
end


# function pbl_sonde_interp()
#     # case-study tool, kept for reference (interpolated-sonde version of pbl_sonde)
# end


pbl_ceil()
pbl_sonde()

