using NCDatasets
using DelimitedFiles
using Distributed

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

@everywhere using NCDatasets

@everywhere const H_MIN = 100
@everywhere const H_MAX = 4000


# Walks every lidar .cdf file, reads the range coordinate, keeps heights
# between H_MIN and H_MAX, and writes the sorted unique values across all
# files to height_lidar.txt — the master vertical grid for downstream code.


@everywhere function process_file(PATH_folder_lidar, file)
    PATH_file = joinpath(PATH_folder_lidar, file)
    dataset = NCDataset(PATH_file, "r")

    height = dataset["range"][:]
    mask = (height .>= H_MIN) .& (height .<= H_MAX)
    h = height[mask]

    close(dataset)
    return h
end


function main()    
    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_lidar))

    heights_from_one_file = pmap(file -> process_file(PATH_folder_lidar, file), set_files)
    
    all_heights = reduce(vcat, heights_from_one_file)
    unique_h = sort(unique(all_heights))
    
    output_file_name = "height_lidar.txt"
    writedlm(joinpath(PATH_output_txt, output_file_name), unique_h)
end

# height from lidar measurment in m 
main()