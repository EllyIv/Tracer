using NCDatasets
using Dates
using TimeZones
using Statistics
using DataFrames
using CSV
using Interpolations
using LsqFit
using Polynomials
using Distributed

addprocs(64)

@everywhere using NCDatasets, Statistics, Interpolations, LsqFit, Polynomials, DataFrames, CSV, Dates, TimeZones

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")
@everywhere include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# === Lidar block / quality criteria ===
@everywhere const MIN_BLOCK_S        = 600     # s, minimum block duration
@everywhere const MAX_GAP_S          = 2       # s, max gap between consecutive samples within a block
@everywhere const TIME_MATCH_S       = 0.5     # s, tolerance when matching block start to filtered timestamp
@everywhere const SPIKE_AMP          = 1.5     # m/s, vertical-velocity spike threshold
@everywhere const MAX_SPIKE_RATIO    = 0.05    # max fraction of consecutive jumps > SPIKE_AMP
@everywhere const BETA_MEAN_MAX      = 30.0    # backscatter mean limit
@everywhere const BETA_STD_MAX       = 30.0    # backscatter std limit
@everywhere const BETA_OUTLIER_SIGMA = 2.0     # sigma cutoff for beta-outlier filter
@everywhere const BETA_KEEP_FRAC     = 0.95    # min fraction of samples that must survive beta-outlier filter
@everywhere const HEIGHT_MIN_M       = 105     # m, minimum height to compute flux
@everywhere const N_SUBWIN           = 5       # number of sub-windows for stationarity test

# === Lidar inversion / unit conversion ===
@everywhere const LIDAR_RATIO        = 50.0
@everywhere const BACKSCATTER_SCALE  = 1e6     # convert to Mm^-1 sr^-1

# === LOD / uncertainty ===
@everywhere const LAG_LOD            = 200:10:300   # s, lag range for LOD
@everywhere const MAX_LAG_AUTOCORR   = 200          # s, max lag for autocorrelation fit

# === Wind-correction (Massman 2000) ===
@everywhere const ALPHA_CORR = 7 / 8
@everywhere const TAU_CORR   = 10.0
@everywhere const NM_CORR    = 0.085

# === Lookup tolerances ===
@everywhere const WIND_LOOKUP_TOL_S   = 3600   # s, wind-profile match tolerance (= 1 h)
@everywhere const REGIME_LOOKUP_TOL_S = 1800   # s, ECOR-regime match tolerance


# Vertical aerosol-flux calculation from Doppler lidar.
#   - For each (block, height) that survives QA, computes mean flux <w'beta'>,
#     <w'^2>, std, lod, 3 uncertainty terms (Petters 2024) and the Foken-Wichura
#     stationarity ratio.
#   - Computes the Massman (2000) high-frequency correction factor (using z = height).
#   - Looks up u*, w*, H from turbulent_regime_full.csv.
#   - Carries through stability, condition, and z/L from filtered_time_lidar.csv.
#
#   Time is converted UTC -> America/Chicago BEFORE matching to the filter file
#   (which uses local time). Each lidar timestamp is independently converted to
#   local time so that profiles spanning UTC midnight are correctly placed in
#   their local calendar day.
#
# Wind lookup:
#   Keyed on (day, month, year). Within-day search picks the wind-profile entry
#   whose time_s is closest to the lidar timestamp, provided the difference is
#   within WIND_LOOKUP_TOL_S. If no within-day match is found, falls back to
#   the previous and next day to handle profiles near local midnight.
#
# LOD test:
#   abs(mean_f) <= lod_mean rejects (works for negative downward fluxes).
#
# Output: fluxH_unc.csv (no filtering on uncertainty/stationarity yet).



@everywhere function shiftsignal(signal::Vector{<:Real}, shift::Int)
    return vcat(fill(NaN, shift), signal[1:end-shift])
end


@everywhere function block_division(set_time)
    pause_ind = vcat(findall(abs.(diff(set_time)) .> MAX_GAP_S), length(set_time))

    if pause_ind[1] != 1
        index_in  = [1; pause_ind[1:end-1] .+ 1]
        index_out = pause_ind
    else
        index_in  = pause_ind[1:end-1] .+ 1
        index_out = pause_ind[2:end]
    end

    blocks = Tuple{Int,Int,Int}[]
    for i in eachindex(index_in)
        block_size = index_out[i] - index_in[i]
        if block_size > MIN_BLOCK_S
            push!(blocks, (index_in[i], index_out[i], block_size))
        end
    end
    return blocks
end


@everywhere function is_signal_good(signal)
    diff_signal = signal .- shiftsignal(signal, 1)
    spikes = (diff_signal .> SPIKE_AMP) .| (diff_signal .< -SPIKE_AMP)
    spike_ratio = sum(spikes) / length(diff_signal)
    return spike_ratio < MAX_SPIKE_RATIO
end


@everywhere function detrend_lin(data_in, time)
    data_in = collect(skipmissing(data_in))
    fit_data = Polynomials.fit(time, Float64.(data_in), 1)
    return Float64.(data_in) .- fit_data.(time)
end


@everywhere function calc_lod_lag_method_prebuilt(itp, t, beta_prime, lag_sec)
    t_shifted = t .+ lag_sec
    v_lagged  = itp.(t_shifted)
    valid     = .!isnan.(v_lagged)
    return mean(beta_prime[valid] .* v_lagged[valid])
end


@everywhere function calc_cov_irregular_time(t, x, lag_sec)
    itp = extrapolate(interpolate((t,), x, Gridded(Linear())), NaN)
    t_shifted = t .+ lag_sec
    x_lagged  = itp.(t_shifted)
    valid     = .!isnan.(x_lagged)
    return mean(x[valid] .* x_lagged[valid])
end


# Compute (delta_noise, I_noise) from autocorrelation fit.
# Returns (missing, missing) if curve_fit throws (e.g. on degenerate signal).
@everywhere function calc_uncert_irregular(t, signal_t)
    try
        lag_grid = 0:1:MAX_LAG_AUTOCORR
        cvr = [calc_cov_irregular_time(t, signal_t, lag) for lag in lag_grid]

        min_lag_fit = 1
        max_lag_fit = 50
        for c in 1:(length(cvr) - 1)
            if cvr[c+1] <= 0 && cvr[c] > 0
                max_lag_fit = c + 1
                break
            end
        end

        x_fit = lag_grid[min_lag_fit:max_lag_fit]
        y_fit = cvr[min_lag_fit:max_lag_fit]
        model(x, p) = p[1] .* exp.(-p[2] .* x)
        fit = curve_fit(model, x_fit, y_fit, [1.0, 0.01])
        nu, k = fit.param

        if nu / k < 0
            nu = -nu
        end
        delta_noise = cvr[1] - nu
        I_noise     = (2 / 5) * (nu / k)^(3 / 2)

        return delta_noise, I_noise
    catch e
        return missing, missing
    end
end


# Foken-Wichura stationarity ratio.
@everywhere function stationarity_ratio(time_int, vel_int, back_int, full_flux)
    n = length(time_int)
    n < N_SUBWIN && return missing

    sub_size = div(n, N_SUBWIN)
    sub_size < 5 && return missing

    sub_fluxes = Float64[]
    for k in 1:N_SUBWIN
        i1 = (k - 1) * sub_size + 1
        i2 = k == N_SUBWIN ? n : k * sub_size
        t_sub = time_int[i1:i2]
        v_sub = vel_int[i1:i2]
        b_sub = back_int[i1:i2]

        v_p = detrend_lin(v_sub, t_sub)
        b_p = detrend_lin(b_sub, t_sub)
        push!(sub_fluxes, mean(v_p .* b_p))
    end

    full_flux == 0 && return missing
    return (mean(sub_fluxes) - full_flux) / full_flux
end


@everywhere function calc_flux(time_int, vel_int, back_int, day, month, year, diff_t, hh,
                                zL, stability, condition)
    time_in = time_int[1]
    sample_period = time_int[end] - time_int[1]

    mean_w      = mean(vel_int)
    std_w       = std(vel_int)
    var_w       = detrend_lin(vel_int, time_int)
    mean_var_w2 = mean(var_w .* var_w)
    std_var_w2  = std(var_w .* var_w)

    mean_b      = mean(back_int)
    std_b       = std(back_int)
    var_b       = detrend_lin(back_int, time_int)
    mean_var_b  = mean(var_b)
    std_var_b   = std(var_b)

    flux_int    = var_w .* var_b
    mean_f      = mean(flux_int)
    std_f       = std(flux_int)

    # Stationarity test on full block (before beta filter)
    stat_ratio = stationarity_ratio(time_int, vel_int, back_int, mean_f)

    # Quality checks
    flag_good_signal = is_signal_good(var_w)

    lower = mean_var_b - BETA_OUTLIER_SIGMA * std_var_b
    upper = mean_var_b + BETA_OUTLIER_SIGMA * std_var_b
    good_mask = (var_b .>= lower) .& (var_b .<= upper)
    keep_frac = sum(good_mask) / length(var_b)

    if !(mean_b <= BETA_MEAN_MAX && std_b <= BETA_STD_MAX && flag_good_signal && keep_frac >= BETA_KEEP_FRAC)
        return Float64[]
    end

    # Apply beta-outlier filter
    var_w    = var_w[good_mask]
    var_b    = var_b[good_mask]
    time_int = time_int[good_mask]
    flux_int = flux_int[good_mask]

    # LOD
    itp = extrapolate(interpolate((time_int,), var_w, Gridded(Linear())), NaN)
    lod_fluxes = [calc_lod_lag_method_prebuilt(itp, time_int, var_b, lag) for lag in LAG_LOD]
    lod_mean   = mean(lod_fluxes)

    # Reject if |flux| is below noise floor (LOD test works for negative fluxes too)
    if abs(mean_f) <= lod_mean
        return Float64[]
    end

    # Uncertainty (Petters 2024 eq. 11)
    delta_w, _    = calc_uncert_irregular(time_int, var_w)
    delta_b, _    = calc_uncert_irregular(time_int, var_b)
    delta_f, II_f = calc_uncert_irregular(time_int, flux_int)
    N = length(time_int)

    # If any of the uncertainty fits failed, propagate missing through the row
    if ismissing(delta_w) || ismissing(delta_b) || ismissing(II_f)
        err_noise    = missing
        err_sample   = missing
        err_ensemble = missing
    else
        err_noise2 = mean(var_b .^ 2) * delta_b / N + mean(var_w .^ 2) * delta_w / N
        err_noise  = err_noise2 > 0 ? sqrt(err_noise2) : 0.0

        err_sample = 2 * II_f / sample_period *
                     (mean_f - (std_w^2 - delta_w^2) * (std_b^2 - delta_b^2))

        err_ensemble = 2 * II_f / sample_period * mean_f
    end

    return Any[day, month, year, time_in, diff_t, hh,
               mean_b, std_b, mean_w, std_w, mean_var_w2, std_var_w2,
               mean_f, std_f, lod_mean,
               err_noise, err_sample, err_ensemble,
               stat_ratio,
               zL, stability, condition]
end


# Convert UTC timestamps to LOCAL America/Chicago time, returning per-sample
# (year, month, day, time_in_sec).
# Each sample is converted independently so profiles spanning UTC midnight are
# correctly placed in the local calendar day.
@everywhere function from_datetime_to_local(set_time, tz)
    zoned    = ZonedDateTime.(set_time, tz"UTC")
    dt_local = DateTime.(astimezone.(zoned, tz))

    true_year  = year.(dt_local)
    true_month = month.(dt_local)
    true_day   = day.(dt_local)

    true_time_sec = Dates.value.(dt_local .- DateTime.(true_year, true_month, true_day)) ./ 1000

    return true_year, true_month, true_day, true_time_sec
end


@everywhere function read_lidar_data_for_one_file(PATH_file, tz,
                                                   day_keys, time_per_key,
                                                   pbl_per_key, cloud_per_key, rh_per_key,
                                                   zL_per_key, stab_per_key, cond_per_key)
    dataset = NCDataset(PATH_file, "r")
    all_info = []

    try
        height = dataset["range"][:]
        dh = unique(diff(height))
        if length(dh) > 1
            println("WARNING: non-uniform height grid in $PATH_file: $dh")
        end
        dh = dh[1]

        ind_h_start = findfirst(h -> h >= HEIGHT_MIN_M, height)
        ind_h_start === nothing && (close(dataset); return all_info)

        time_in_datetime = dataset["time"][:]
        println("Variables in $PATH_file:")
        for (name, var) in dataset
            println("  $name : $(size(var)) : ",
                    haskey(var.attrib, "long_name") ? var.attrib["long_name"] : "")
        end

        # Convert UTC -> local time per sample. block_division then operates on
        # local seconds-of-day (DST jumps will cause a > 2 s gap and split blocks
        # at the transition, which is correct).
        years_l, months_l, days_l, time_in_sec_local = from_datetime_to_local(time_in_datetime, tz)
        set_blocks_ind = block_division(time_in_sec_local)
        isempty(set_blocks_ind) && (close(dataset); return all_info)

        for ind_t in set_blocks_ind
            block_start_idx = ind_t[1]

            # Use LOCAL date and time of the block's first sample
            local_year  = years_l[block_start_idx]
            local_month = months_l[block_start_idx]
            local_day   = days_l[block_start_idx]
            t_block_start = time_in_sec_local[block_start_idx]

            key = (local_year, local_month, local_day)
            haskey(day_keys, key) || continue
            idxs_for_day = day_keys[key]

            # Match this block to a row in the filter file
            matched_idx = nothing
            for idx in idxs_for_day
                if abs(time_per_key[idx] - t_block_start) <= TIME_MATCH_S
                    matched_idx = idx
                    break
                end
            end
            matched_idx === nothing && continue

            # Per-row metadata
            pbl_h_   = pbl_per_key[matched_idx]
            cloud_h_ = cloud_per_key[matched_idx]
            rh_h_    = rh_per_key[matched_idx]
            zL_      = zL_per_key[matched_idx]
            stab_    = stab_per_key[matched_idx]
            cond_    = cond_per_key[matched_idx]

            # Height ceiling = min(cloud_h, pbl_h, rh_h)
            ceiling_candidates = filter(!ismissing, [pbl_h_, cloud_h_, rh_h_])
            isempty(ceiling_candidates) && continue
            height_ceiling = minimum(ceiling_candidates)
            ind_h_top = findlast(h -> h <= height_ceiling, height)
            ind_h_top === nothing && continue
            ind_h_top < ind_h_start && continue

            time_range_block = time_in_sec_local[ind_t[1]:ind_t[2]]

            for hh in ind_h_start:ind_h_top
                vel    = dataset["radial_velocity"][hh, ind_t[1]:ind_t[2]]
                atback = dataset["attenuated_backscatter"][hh, ind_t[1]:ind_t[2]]

                if any(ismissing, vel) || any(ismissing, atback)
                    continue
                end

                back = atback ./ (1 .- 2 .* LIDAR_RATIO .* atback .* dh) .* BACKSCATTER_SCALE

                row = calc_flux(time_range_block, vel, back,
                                local_day, local_month, local_year,
                                ind_t[3], height[hh],
                                zL_, stab_, cond_)
                if !isempty(row)
                    push!(all_info, row)
                end
            end
        end
    finally
        close(dataset)
    end

    return all_info
end


@everywhere function process_lidar_file(file::String, tz,
                                          day_keys, time_per_key,
                                          pbl_per_key, cloud_per_key, rh_per_key,
                                          zL_per_key, stab_per_key, cond_per_key)
    PATH_file = joinpath(PATH_folder_lidar, file)
    println("processing $file")
    return read_lidar_data_for_one_file(PATH_file, tz,
                                          day_keys, time_per_key,
                                          pbl_per_key, cloud_per_key, rh_per_key,
                                          zL_per_key, stab_per_key, cond_per_key)
end


# Wind lookup: (day, month, year) -> sorted Vector of (time_s, height_m, wind_speed)
# tuples. Within-day nearest-time match handles regular case;
# previous/next-day fallback handles profiles near local midnight.
function build_wind_lookup(df_wind)
    lookup = Dict{Tuple{Int,Int,Int}, Vector{Tuple{Float64,Float64,Float64}}}()
    for i in 1:nrow(df_wind)
        key = (Int(df_wind.day[i]), Int(df_wind.month[i]), Int(df_wind.year[i]))
        v   = (Float64(df_wind.time_s[i]), Float64(df_wind.height_m[i]), Float64(df_wind.wind[i]))
        push!(get!(lookup, key, Tuple{Float64,Float64,Float64}[]), v)
    end
    # Sort each day's entries by time_s for predictable downstream behavior
    for k in keys(lookup)
        sort!(lookup[k], by = x -> x[1])
    end
    return lookup
end


# Search within (day, month, year). Returns wind speed at the height closest to
# h_target, taken from the wind profile whose time_s is closest to t_target,
# subject to |delta_t| <= WIND_LOOKUP_TOL_S. Returns missing if no match.
function find_wind_in_day(wind_lookup, day, month, year, t_target, h_target)
    key = (Int(day), Int(month), Int(year))
    haskey(wind_lookup, key) || return missing
    entries = wind_lookup[key]
    isempty(entries) && return missing

    # Distinct sonde-launch times present this day
    times_unique = unique([e[1] for e in entries])
    diffs = abs.(times_unique .- t_target)
    m, idx_t = findmin(diffs)
    m <= WIND_LOOKUP_TOL_S || return missing
    t_match = times_unique[idx_t]

    # All (height, wind) at that matched time
    candidates = [(e[2], e[3]) for e in entries if e[1] == t_match]
    isempty(candidates) && return missing
    heights = [c[1] for c in candidates]
    closest_h_idx = argmin(abs.(heights .- h_target))
    return candidates[closest_h_idx][2]
end


# Wind speed lookup with day-rollover handling.
# For each lidar timestamp, tries:
#   1. same local day, nearest time within tolerance
#   2. if (1) returns missing AND lidar time is near start of day, try previous day
#      shifted to negative t_target (i.e. t_target - 86400)
#   3. if (1) returns missing AND lidar time is near end of day, try next day
#      shifted to t_target + 86400
function find_wind_speed(wind_lookup, day_arr, month_arr, year_arr, time_s_arr, h_arr)
    n = length(time_s_arr)
    out = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        d, m, y = Int(day_arr[i]), Int(month_arr[i]), Int(year_arr[i])
        t = Float64(time_s_arr[i])
        h_target = Float64(h_arr[i])

        # 1. Same-day match
        v = find_wind_in_day(wind_lookup, d, m, y, t, h_target)
        if !ismissing(v)
            out[i] = v
            continue
        end

        # 2. Previous-day fallback (if lidar time is within tolerance of midnight)
        if t < WIND_LOOKUP_TOL_S
            prev_date = Date(y, m, d) - Day(1)
            v = find_wind_in_day(wind_lookup,
                                 day(prev_date), month(prev_date), year(prev_date),
                                 t + 86400, h_target)
            if !ismissing(v)
                out[i] = v
                continue
            end
        end

        # 3. Next-day fallback (if lidar time is within tolerance of next midnight)
        if t > 86400 - WIND_LOOKUP_TOL_S
            next_date = Date(y, m, d) + Day(1)
            v = find_wind_in_day(wind_lookup,
                                 day(next_date), month(next_date), year(next_date),
                                 t - 86400, h_target)
            if !ismissing(v)
                out[i] = v
                continue
            end
        end
    end
    return out
end


function build_regime_lookup(df_reg)
    lookup = Dict{Tuple{Int,Int,Int}, Vector{Tuple{Float64,Union{Missing,Float64},Union{Missing,Float64},Union{Missing,Float64}}}}()
    for i in 1:nrow(df_reg)
        key = (df_reg.day[i], df_reg.month[i], df_reg.year[i])
        v   = (Float64(df_reg.time_s[i]), df_reg.ustar[i], df_reg.h[i], df_reg.wstar[i])
        push!(get!(lookup, key, []), v)
    end
    return lookup
end


function find_regime(regime_lookup, day_arr, month_arr, year_arr, time_s_arr)
    n = length(time_s_arr)
    ustar_o = Vector{Union{Missing, Float64}}(missing, n)
    h_o     = Vector{Union{Missing, Float64}}(missing, n)
    wstar_o = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        key = (Int(day_arr[i]), Int(month_arr[i]), Int(year_arr[i]))
        haskey(regime_lookup, key) || continue
        candidates = regime_lookup[key]
        ts = [c[1] for c in candidates]
        diffs = abs.(ts .- time_s_arr[i])
        m, idx = findmin(diffs)
        if m <= REGIME_LOOKUP_TOL_S
            ustar_o[i] = candidates[idx][2]
            h_o[i]     = candidates[idx][3]
            wstar_o[i] = candidates[idx][4]
        end
    end
    return ustar_o, h_o, wstar_o
end


function main()
    println("loading filtered timestamps + reference data...")
    df_filtered = CSV.read(joinpath(PATH_output_txt, "filtered_time_lidar.csv"), DataFrame)
    df_wind     = CSV.read(joinpath(PATH_output_txt, "wind_profile_interp.csv"), DataFrame)
    df_regime   = CSV.read(joinpath(PATH_output_txt, "turbulent_regime_full.csv"), DataFrame)

    # Per-day index (LOCAL date) for fast lookup inside workers
    day_keys = Dict{Tuple{Int,Int,Int}, Vector{Int}}()
    for i in 1:nrow(df_filtered)
        key = (df_filtered.year[i], df_filtered.month[i], df_filtered.day[i])
        push!(get!(day_keys, key, Int[]), i)
    end

    time_per_key  = Float64.(df_filtered.time_s)
    pbl_per_key   = df_filtered.pbl_h
    cloud_per_key = df_filtered.cloud_h
    rh_per_key    = df_filtered.rh_h
    zL_per_key    = df_filtered.z_L
    stab_per_key  = df_filtered.stability
    cond_per_key  = df_filtered.condition

    tz = tz"America/Chicago"

    set_files = filter(x -> endswith(x, ".cdf"), readdir(PATH_folder_lidar))
    println("processing ", length(set_files), " lidar files in parallel...")

    info_all_files_list = pmap(file -> process_lidar_file(file, tz,
                                                            day_keys, time_per_key,
                                                            pbl_per_key, cloud_per_key, rh_per_key,
                                                            zL_per_key, stab_per_key, cond_per_key),
                               set_files)
    info_all = reduce(vcat, info_all_files_list)
    println("total flux rows after quality+LOD filter: ", length(info_all))
    if isempty(info_all)
        println("no surviving rows; nothing to write.")
        return
    end

    wind_lookup   = build_wind_lookup(df_wind)
    regime_lookup = build_regime_lookup(df_regime)

    day_a    = [r[1]  for r in info_all]
    month_a  = [r[2]  for r in info_all]
    year_a   = [r[3]  for r in info_all]
    time_s_a = [r[4]  for r in info_all]
    height_a = [r[6]  for r in info_all]

    println("looking up wind + computing Massman correction...")
    wind_a = find_wind_speed(wind_lookup, day_a, month_a, year_a, time_s_a, height_a)
    corr_a = [ismissing(u) ? missing : 1 + (2π * NM_CORR * TAU_CORR * u / Float64(z))^ALPHA_CORR
              for (u, z) in zip(wind_a, height_a)]

    n_wind_ok = count(!ismissing, wind_a)
    println("wind matches: $n_wind_ok / $(length(wind_a)) ($(round(100*n_wind_ok/length(wind_a), digits=1))%)")

    println("looking up u*, H, w* from turbulent_regime_full.csv...")
    ustar_a, hflux_a, wstar_a = find_regime(regime_lookup, day_a, month_a, year_a, time_s_a)

    df_out = DataFrame(
        day          = Int.(day_a),
        month        = Int.(month_a),
        year         = Int.(year_a),
        time_s       = time_s_a,
        time_range   = [r[5]  for r in info_all],
        height       = height_a,
        mean_b       = [r[7]  for r in info_all],
        std_b        = [r[8]  for r in info_all],
        mean_w       = [r[9]  for r in info_all],
        std_w        = [r[10] for r in info_all],
        w2_mean      = [r[11] for r in info_all],
        w2_std       = [r[12] for r in info_all],
        flux_mean    = [r[13] for r in info_all],
        flux_std     = [r[14] for r in info_all],
        lod_mean     = [r[15] for r in info_all],
        err_noise    = [r[16] for r in info_all],
        err_sample   = [r[17] for r in info_all],
        err_ensemble = [r[18] for r in info_all],
        stat_ratio   = [r[19] for r in info_all],
        z_L          = [r[20] for r in info_all],
        stability    = [r[21] for r in info_all],
        condition    = [r[22] for r in info_all],
        wind         = wind_a,
        corr_factor  = corr_a,
        ustar        = ustar_a,
        h_flux       = hflux_a,
        wstar        = wstar_a,
    )

    output_csv = joinpath(PATH_output_txt, "fluxH_unc.csv")
    CSV.write(output_csv, df_out)
    println(output_csv)
    println("rows written: ", nrow(df_out))
end


main()