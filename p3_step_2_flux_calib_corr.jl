using DataFrames
using CSV
using Statistics
using Interpolations

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# === Filter thresholds ===
const HEIGHT_MIN_M     = 105      # m, lowest valid lidar measurement height
const STAT_RATIO_LIMIT = 0.30     # |stat_ratio| > 0.30 -> non-stationary (Foken-Wichura)
const SNR_MIN          = 1.0      # |flux_raw / err| must be >= this for all 3 uncertainty types

# === Calibration scaling ===
const CALIB_MM_FACTOR  = 100.0    # multiplicative factor: flux_calib = flux_raw * CALIB_MM_FACTOR / slope

# === Lookup tolerances ===
const TIME_MATCH_S     = 1.0      # s, tolerance for matching flux time_s to calibration time_s


# Calibrates and filters the raw flux output (fluxH_unc.csv).
#   1. For each flux row, looks up slope at the matching (day, month, year, time_s)
#      from calibration_slope_intercept.csv. Match is by exact (day, month, year)
#      + nearest time_s within TIME_MATCH_S; height matches by closest height_m.
#      flux_calibrated = flux_raw * CALIB_MM_FACTOR / slope
#   2. Applies the high-frequency response correction (Massman 2000):
#      flux_corrected = flux_calibrated * corr_factor
#      (rows with missing corr_factor keep flux_corrected = missing; they're
#       retained so downstream code can decide whether to use them)
#   3. Applies all three uncertainty filters on RAW flux:
#      |flux_raw / err_*| >= SNR_MIN
#      (SNR is a property of the raw lidar measurement, not the post-correction
#       value; matches the old pipeline's behavior)
#   4. Applies the stationarity filter: |stat_ratio| <= STAT_RATIO_LIMIT.
#   5. Applies height floor: height >= HEIGHT_MIN_M.
# Output: fluxH_final.csv



function build_calib_index(df_calib)
    idx = Dict{Tuple{Int,Int,Int},
               Vector{Tuple{Float64, Float64, Float64, Float64, Float64}}}()
    for i in 1:nrow(df_calib)
        key = (Int(df_calib.day[i]), Int(df_calib.month[i]), Int(df_calib.year[i]))
        push!(get!(idx, key, []),
              (Float64(df_calib.time_s[i]),
               Float64(df_calib.height_m[i]),
               Float64(df_calib.slope[i]),
               Float64(df_calib.intercept[i]),
               Float64(df_calib.rh[i])))
    end
    for key in keys(idx)
        sort!(idx[key], by = x -> x[1])
    end
    return idx
end


function lookup_calib(calib_index, day, month, year, time_s, height)
    key = (Int(day), Int(month), Int(year))
    haskey(calib_index, key) || return (NaN, NaN)
    rows = calib_index[key]
    isempty(rows) && return (NaN, NaN)

    matching_rows = filter(r -> abs(r[1] - Float64(time_s)) <= TIME_MATCH_S, rows)
    isempty(matching_rows) && return (NaN, NaN)

    h_target = Float64(height)
    closest = matching_rows[argmin(abs.([r[2] for r in matching_rows] .- h_target))]
    return (closest[3], closest[5])
end


function main()
    println("loading inputs...")
    df_unc   = CSV.read(joinpath(PATH_output_txt, "fluxH_unc.csv"),                          DataFrame)
    df_calib = CSV.read(joinpath(PATH_output_txt, "calibration_slope_intercept.csv"),        DataFrame)

    println("rows in fluxH_unc:    ", nrow(df_unc))
    println("rows in calibration:  ", nrow(df_calib))

    println("\nbuilding calibration lookup index...")
    calib_index = build_calib_index(df_calib)
    println("days in calibration index: ", length(calib_index))

    println("\nlooking up calibration per flux row...")
    n = nrow(df_unc)
    rh_at_row       = Vector{Union{Missing, Float64}}(missing, n)
    flux_calibrated = Vector{Union{Missing, Float64}}(missing, n)
    flux_corrected  = Vector{Union{Missing, Float64}}(missing, n)

    n_no_match = 0
    n_zero_slope = 0
    for i in 1:n
        slope, rh = lookup_calib(calib_index,
                                   df_unc.day[i], df_unc.month[i], df_unc.year[i],
                                   df_unc.time_s[i], df_unc.height[i])

        if isnan(slope)
            n_no_match += 1
            continue
        end
        if slope == 0
            n_zero_slope += 1
            continue
        end

        rh_at_row[i] = isnan(rh) ? missing : rh
        flux_calibrated[i] = Float64(df_unc.flux_mean[i]) * CALIB_MM_FACTOR / slope

        cf = df_unc.corr_factor[i]
        if !ismissing(cf)
            flux_corrected[i] = flux_calibrated[i] * Float64(cf)
        end
        # else: leave flux_corrected[i] as missing
    end

    println("rows with no calibration match:   $n_no_match")
    println("rows with zero slope:             $n_zero_slope")
    println("rows with calibrated flux:        $(count(!ismissing, flux_calibrated))")
    println("rows with corrected flux (Massman): $(count(!ismissing, flux_corrected))")

    # === Apply filters ===
    # Critical: SNR is checked on the RAW flux (flux_mean), not the corrected one.
    # This matches the old pipeline's behavior. SNR is a property of the raw
    # measurement, not the post-calibration product.
    println("\napplying filters...")
    flux_raw_arr = Float64.(coalesce.(df_unc.flux_mean,    NaN))
    err_n_arr    = Float64.(coalesce.(df_unc.err_noise,    NaN))
    err_s_arr    = Float64.(coalesce.(df_unc.err_sample,   NaN))
    err_e_arr    = Float64.(coalesce.(df_unc.err_ensemble, NaN))
    stat_arr     = Float64.(coalesce.(df_unc.stat_ratio,   NaN))
    height_arr   = Float64.(df_unc.height)

    mask_height = height_arr .>= HEIGHT_MIN_M
    mask_noise  = (.!isnan.(flux_raw_arr)) .& (abs.(flux_raw_arr ./ err_n_arr) .>= SNR_MIN) .& (flux_raw_arr .!= 0)
    mask_sample = (.!isnan.(flux_raw_arr)) .& (abs.(flux_raw_arr ./ err_s_arr) .>= SNR_MIN) .& (flux_raw_arr .!= 0)
    mask_ens    = (.!isnan.(flux_raw_arr)) .& (abs.(flux_raw_arr ./ err_e_arr) .>= SNR_MIN) .& (flux_raw_arr .!= 0)
    mask_stat   = (.!isnan.(stat_arr)) .& (abs.(stat_arr) .<= STAT_RATIO_LIMIT)

    final_mask = mask_height .& mask_noise .& mask_sample .& mask_ens .& mask_stat

    println("\n=== Filter breakdown ===")
    println("  total            : $n")
    println("  pass height      : $(sum(mask_height))")
    println("  pass noise SNR   : $(sum(mask_noise))")
    println("  pass sample SNR  : $(sum(mask_sample))")
    println("  pass ensemble SNR: $(sum(mask_ens))")
    println("  pass stationarity: $(sum(mask_stat))")
    println("  pass ALL         : $(sum(final_mask))")

    df_out = DataFrame(
        day          = df_unc.day[final_mask],
        month        = df_unc.month[final_mask],
        year         = df_unc.year[final_mask],
        time_s       = df_unc.time_s[final_mask],
        time_range   = df_unc.time_range[final_mask],
        height       = df_unc.height[final_mask],
        mean_b       = df_unc.mean_b[final_mask],
        std_b        = df_unc.std_b[final_mask],
        mean_w       = df_unc.mean_w[final_mask],
        std_w        = df_unc.std_w[final_mask],
        w2_mean      = df_unc.w2_mean[final_mask],
        w2_std       = df_unc.w2_std[final_mask],
        flux_raw     = df_unc.flux_mean[final_mask],
        flux_std     = df_unc.flux_std[final_mask],
        flux_calib   = flux_calibrated[final_mask],
        flux_corr    = flux_corrected[final_mask],
        rh           = rh_at_row[final_mask],
        lod_mean     = df_unc.lod_mean[final_mask],
        err_noise    = df_unc.err_noise[final_mask],
        err_sample   = df_unc.err_sample[final_mask],
        err_ensemble = df_unc.err_ensemble[final_mask],
        stat_ratio   = df_unc.stat_ratio[final_mask],
        z_L          = df_unc.z_L[final_mask],
        stability    = df_unc.stability[final_mask],
        condition    = df_unc.condition[final_mask],
        wind         = df_unc.wind[final_mask],
        corr_factor  = df_unc.corr_factor[final_mask],
        ustar        = df_unc.ustar[final_mask],
        h_flux       = df_unc.h_flux[final_mask],
        wstar        = df_unc.wstar[final_mask],
    )

    output_csv = joinpath(PATH_output_txt, "fluxH_final.csv")
    CSV.write(output_csv, df_out)
    println("\n", output_csv)
    println("rows written: ", nrow(df_out))
end


main()