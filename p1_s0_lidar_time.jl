using Glob 
using CSV   
using Plots 
using Colors
using NCDatasets 
using Dates 
using Statistics 
using Dierckx    
using Polynomials
using StatsBase 
using DSP  
using DelimitedFiles
using Measures
using LaTeXStrings
using Interpolations
using LsqFit 
using Distributed

addprocs(64)

const PATH_folder_lidar = "/home/ellai/data/archive/DOE-SC0021074 (Tracer)/level 1/houdlfptM1/"
const PATH_output_png = "/home/ellai/test_arm_data/flux_calc/VER6/output_png/"
const PATH_output_txt = "/home/ellai/test_arm_data/flux_calc/VER6/output_txt/"

@everywhere using NCDatasets, Statistics, Interpolations, LsqFit, Dates, Polynomials
@everywhere using Glob, DelimitedFiles

@everywhere function from_datetime_to_sec(set_time, day_in, month_in, year_in, time_zone)
    time_sec = hour.(set_time) .* 3600 .+ minute.(set_time) .* 60 .+ second.(set_time) 
    
    base_datetime = DateTime(year_in, month_in, day_in)
    total_time_dt = base_datetime .+ Second.(time_sec)
    dt_local = total_time_dt .+ Hour(time_zone)

    true_year = year.(dt_local)
    true_month = month.(dt_local)
    true_day = day.(dt_local)

    true_time_sec = Dates.value.(dt_local .- DateTime.(true_year, true_month, true_day)) ./ 1000

    return true_time_sec, true_day, true_month, true_year
end


@everywhere function block_division(set_time)
    pause_ind = vcat(findall(abs.(diff(set_time)) .> 2), length(set_time))  # find indices where step > 2 and add last index

    if pause_ind[1] != 1
        index_in = [1; pause_ind[1:end-1] .+ 1]
        index_out = pause_ind 
    else 
        index_in = pause_ind[1:end-1] .+ 1
        index_out = pause_ind[2:end]
    end

    blocks = []
    for i in eachindex(index_in)
        block_size = index_out[i] - index_in[i]
        if block_size > 600 
            push!(blocks, (index_in[i], index_out[i], block_size)) 
        end
    end
    return blocks
end



@everywhere function read_lidar_data_for_one_file(PATH_file, init_day, time_zone, init_month , init_year)
    dataset = NCDataset(PATH_file, "r")

    all_info = []
    time_in_datetime = dataset["time"][:] # sec in considered day (without taking into account day, month or year. and w/o timezome )
    time_in_sec, day, month, year =  from_datetime_to_sec(time_in_datetime, init_day, init_month, init_year, time_zone)
    set_blocks_ind = block_division(time_in_sec)

    if set_blocks_ind != false 
        for ind_t in set_blocks_ind
            time_start = time_in_sec[ind_t[1]]
            time_range = ind_t[3]
            all_info = push!(all_info, [day[ind_t[1]], month[ind_t[1]] , year[ind_t[1]], time_start, time_range]) 
        end 
    end
    
    close(dataset) 
    return all_info
end

@everywhere function process_file(PATH_folder_lidar, file, time_zone)
    file_parts = split(basename(file), ".")
    year  = parse(Int, file_parts[3][1:4])
    month = parse(Int, file_parts[3][5:6])
    day   = parse(Int, file_parts[3][7:8])
   
    file_date = Date(year,  month, day) 
    start_date = Date(2021, 10, 1)
    end_date   = Date(2022, 10, 1)   
    if file_date >= start_date && file_date <= end_date
        PATH_file = joinpath(PATH_folder_lidar, file)
        info_one_file = read_lidar_data_for_one_file(PATH_file, day, time_zone, month, year)
        return info_one_file
    end
end


function main()
    time_zone = -5 # houston 
    
    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_lidar))
    time_from_one_file = pmap(file -> process_file(PATH_folder_lidar, file, time_zone), set_files)
    time_from_one_file = filter(x -> x !== nothing, time_from_one_file)

    all_time = reduce(vcat, time_from_one_file)
    info_matrix = hcat(all_time...)'  

    output_file_name = "time_lidar.txt"
    writedlm(PATH_output_txt * output_file_name, info_matrix)
end


main()



# in the end i have blocks of time [day, month, year, time_start, time_range]
# block defines as set of time that is more than 600 sec, and between each time is less than 2 sec break. 