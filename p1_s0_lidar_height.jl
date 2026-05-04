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

# addprocs(64)
@everywhere using NCDatasets


const PATH_folder_lidar = "/home/ellai/data/archive/DOE-SC0021074 (Tracer)/level 1/houdlfptM1/"
const PATH_output_png = "/home/ellai/test_arm_data/flux_calc/VER6/output_png/"
const PATH_output_txt = "/home/ellai/test_arm_data/flux_calc/VER6/output_txt/"


@everywhere function process_file(PATH_folder_lidar, file)
    file_parts = split(basename(file), ".")
    day = file_parts[3][7:8]
    month = file_parts[3][5:6]
    year = file_parts[3][1:4]
    println("day ", day, " month ", month ," year ", year)

    dataset = NCDataset(PATH_folder_lidar * file, "r")

    # variable_names = keys(dataset)
    # println("Variables in the dataset: ", variable_names)
    # sleep(10)

    height = dataset["range"][:]
    mask = (height .>= 100) .& (height .<= 4000)
    return height[mask]
end


function main()    
    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_lidar))

    heights_from_one_file = pmap(file -> process_file(PATH_folder_lidar, file), set_files)
    
    all_heights = reduce(vcat, heights_from_one_file)
    unique_h = sort(unique(all_heights))
    
    println(unique_h)
    output_file_name = "height_lidar.txt"
    writedlm(PATH_output_txt * output_file_name, unique_h)
end

# height from lidar measurment in m 
main()
