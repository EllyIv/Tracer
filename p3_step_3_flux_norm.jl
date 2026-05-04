using DataFrames
using CSV
using Statistics

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")


# Normalizes the final flux dataset to dimensionless variables for
# mixed-layer similarity analysis.
#   z_norm     = height / PBL_h         (dimensionless height)
#   w2_norm    = <w'^2> / w*^2          (Deardorff-scaled vertical velocity variance)
#   flux_norm  = <w'beta'> / (w* * beta_scale)   [optional, only if requested]
#
# Reads pbl_aligned_with_lidar_filtered.csv only to attach a fresh PBL height
# (consistent with the canonical PBL used elsewhere). Output: fluxH_norm.csv



function build_pbl_lookup(df_pbl)
    lookup = Dict{Tuple{Int,Int,Int,Float64}, Union{Missing, Float64}}()
    for i in 1:nrow(df_pbl)
        key = (Int(df_pbl.day[i]), Int(df_pbl.month[i]), Int(df_pbl.year[i]), Float64(df_pbl.time_s[i]))
        lookup[key] = df_pbl.pbl_filtered[i]
    end
    return lookup
end


function lookup_pbl(pbl_lookup, day, month, year, time_s)
    key = (Int(day), Int(month), Int(year), Float64(time_s))
    return get(pbl_lookup, key, missing)
end


function main()
    println("loading inputs...")
    df_flux = CSV.read(joinpath(PATH_output_txt, "fluxH_final.csv"),                       DataFrame)
    df_pbl  = CSV.read(joinpath(PATH_output_txt, "pbl_aligned_with_lidar_filtered.csv"),   DataFrame)

    println("rows in fluxH_final: ", nrow(df_flux))

    pbl_lookup = build_pbl_lookup(df_pbl)

    n = nrow(df_flux)
    pbl_h_arr = Vector{Union{Missing, Float64}}(missing, n)
    z_norm    = Vector{Union{Missing, Float64}}(missing, n)
    w2_norm   = Vector{Union{Missing, Float64}}(missing, n)

    println("computing normalizations...")
    for i in 1:n
        pbl = lookup_pbl(pbl_lookup, df_flux.day[i], df_flux.month[i], df_flux.year[i], df_flux.time_s[i])
        pbl_h_arr[i] = pbl

        # z / PBL
        if !ismissing(pbl) && pbl > 0
            z_norm[i] = Float64(df_flux.height[i]) / Float64(pbl)
        end

        # <w'^2> / w*^2
        wstar = df_flux.wstar[i]
        w2    = df_flux.w2_mean[i]
        if !ismissing(wstar) && wstar > 0 && !ismissing(w2)
            w2_norm[i] = Float64(w2) / Float64(wstar)^2
        end
    end

    df_out = DataFrame(
        day          = df_flux.day,
        month        = df_flux.month,
        year         = df_flux.year,
        time_s       = df_flux.time_s,
        time_range   = df_flux.time_range,
        height       = df_flux.height,
        pbl_h        = pbl_h_arr,
        z_norm       = z_norm,
        mean_b       = df_flux.mean_b,
        std_b        = df_flux.std_b,
        mean_w       = df_flux.mean_w,
        std_w        = df_flux.std_w,
        w2_mean      = df_flux.w2_mean,
        w2_std       = df_flux.w2_std,
        w2_norm      = w2_norm,
        wstar        = df_flux.wstar,
        flux_raw     = df_flux.flux_raw,
        flux_std     = df_flux.flux_std,
        flux_calib   = df_flux.flux_calib,
        flux_corr    = df_flux.flux_corr,
        rh           = df_flux.rh,
        lod_mean     = df_flux.lod_mean,
        err_noise    = df_flux.err_noise,
        err_sample   = df_flux.err_sample,
        err_ensemble = df_flux.err_ensemble,
        stat_ratio   = df_flux.stat_ratio,
        z_L          = df_flux.z_L,
        stability    = df_flux.stability,
        condition    = df_flux.condition,
        wind         = df_flux.wind,
        corr_factor  = df_flux.corr_factor,
        ustar        = df_flux.ustar,
        h_flux       = df_flux.h_flux,
    )
    # df_out = DataFrame(
    #     day          = df_flux.day,
    #     month        = df_flux.month,
    #     year         = df_flux.year,
    #     time_s       = df_flux.time_s,
    #     time_range   = df_flux.time_range,
    #     height       = df_flux.height,
    #     pbl_h        = pbl_h_arr,
    #     z_norm       = z_norm,
    #     w2_mean      = df_flux.w2_mean,
    #     w2_norm      = w2_norm,
    #     wstar        = df_flux.wstar,
    #     flux_corr    = df_flux.flux_corr,
    #     flux_raw     = df_flux.flux_raw,
    #     flux_calib   = df_flux.flux_calib,
    #     rh           = df_flux.rh,
    #     z_L          = df_flux.z_L,
    #     stability    = df_flux.stability,
    #     condition    = df_flux.condition,
    #     ustar        = df_flux.ustar,
    #     h_flux       = df_flux.h_flux,
    #     mean_b       = df_flux.mean_b, 
    # )

    println("\n=== Coverage breakdown ===")
    println("  rows with z_norm  : $(count(!ismissing, df_out.z_norm))")
    println("  rows with w2_norm : $(count(!ismissing, df_out.w2_norm))")

    output_csv = joinpath(PATH_output_txt, "fluxH_norm.csv")
    CSV.write(output_csv, df_out)
    println("\n", output_csv)
    println("rows written: ", nrow(df_out))
end


main()