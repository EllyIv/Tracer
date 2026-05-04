using DelimitedFiles
using DataFrames
using CSV
using Interpolations
using Plots
using Measures
using LaTeXStrings

include("/home/ellai/test_arm_data/flux_calc/VER7/all_paths.jl")

# Calibration unit conversion
const MM_FACTOR = 1e6   # natural units -> Mm^-1 sr^-1 cm^3


# Applies the RH-dependent calibration table (calibration.txt) to every
# RH measurement aligned to the lidar grid (sonde_rh_and_theta_aligned_with_lidar.csv).
# For each lidar-grid timestamp/height, finds the nearest-RH calibration
# bin (per Petters et al. 2024, Section 2.4.6: "closest calibrated slopes
# and intercepts are used") and writes its slope/intercept to
# calibration_slope_intercept.csv.
#
# quick_check : diagnostic plot of the calibration table itself, kept
#               for reference.



function quick_check()
    data = readdlm(joinpath(PATH_output_txt, "../calibration.txt"), skipstart=1)

    RH = data[:, 1]
    m1 = data[:, 2]
    m2 = data[:, 4]
    m3 = data[:, 6]

    p = plot(layout = (1,1), size = (800, 600), left_margin = 15mm, bottom_margin = 12mm, top_margin = 10mm,
             xguidefont = 20, yguidefont = 20, xtickfont = font(18), ytickfont = font(18), legendfont = font(18),
             yscale = :log10, xlims = (10, 95))

    plot!(p, RH, m1 .* MM_FACTOR, label = L"D \geq 0.53\ \mu m", marker = :circle,  lw = 2, color = :blue)
    plot!(p, RH, m2 .* MM_FACTOR, label = L"D \geq 1.03\ \mu m", marker = :square,  lw = 2, color = :red)
    plot!(p, RH, m3 .* MM_FACTOR, label = L"D \geq 3.25\ \mu m", marker = :diamond, lw = 2, color = :black)

    xlabel!(p, L"RH\ (\%)")
    ylabel!(p, L"\frac{d\beta}{dN}\ (Mm^{-1}\ sr^{-1}\ cm^{3})")

    mkpath(PATH_output_png)
    savefig(p, joinpath(PATH_output_png, "slopes_vs_rh.png"))
end


# Find the index of the calibration bin whose RH is closest to the target.
# Returns missing/NaN handling at the call site.
function nearest_calib_idx(target_rh, calib_rh_sorted)
    diffs = abs.(calib_rh_sorted .- target_rh)
    return argmin(diffs)
end


function main()
    info_calib = readdlm(joinpath(PATH_output_txt, "../calibration.txt"), skipstart=1)
    info_calib_rh  = info_calib[:, 1]
    info_calib_slp = info_calib[:, 2] .* MM_FACTOR
    info_calib_int = info_calib[:, 3] .* MM_FACTOR

    sort_idx = sortperm(info_calib_rh)
    info_calib_rh  = info_calib_rh[sort_idx]
    info_calib_slp = info_calib_slp[sort_idx]
    info_calib_int = info_calib_int[sort_idx]

    df_rh = CSV.read(joinpath(PATH_output_txt, "sonde_rh_and_theta_aligned_with_lidar.csv"), DataFrame)
    day, month, year = df_rh.day, df_rh.month, df_rh.year
    time, h          = df_rh.time_s, df_rh.height_m
    rh               = df_rh.rh

    # Per Petters 2024: "closest calibrated slopes and intercepts are used"
    # i.e. nearest-neighbor lookup over RH (NOT linear interpolation).
    slp_vals = Vector{Float64}(undef, length(rh))
    int_vals = Vector{Float64}(undef, length(rh))
    for (k, r) in enumerate(rh)
        if ismissing(r) || isnan(r)
            slp_vals[k] = NaN
            int_vals[k] = NaN
        else
            idx = nearest_calib_idx(r, info_calib_rh)
            slp_vals[k] = info_calib_slp[idx]
            int_vals[k] = info_calib_int[idx]
        end
    end

    # Diagnostic plot
    p1 = scatter(rh, slp_vals, label="Matched Slope", marker=:diamond, color=:blue)
    plot!(p1, info_calib_rh, info_calib_slp, label="Original Slope", lw=2, marker=:circle, color=:red,
          xlabel="RH", ylabel="Slope (*1e6)")

    p2 = scatter(rh, int_vals, label="Matched Intercept", marker=:diamond, color=:blue)
    plot!(p2, info_calib_rh, info_calib_int, label="Original Intercept", lw=2, marker=:circle, color=:red,
          xlabel="RH", ylabel="Intercept (*1e6)")

    final_plot = plot(p1, p2, layout=(1,2), size=(1000, 400),
                      top_margin = 10mm, left_margin = 10mm, bottom_margin = 10mm)
    png(final_plot, joinpath(PATH_output_png, "calibration_slope_intercept_plot.png"))

    df_out = DataFrame(
        day       = day,
        month     = month,
        year      = year,
        time_s    = time,
        height_m  = h,
        rh        = rh,
        intercept = int_vals,
        slope     = slp_vals,
    )
    output_calib_file = joinpath(PATH_output_txt, "calibration_slope_intercept.csv")
    CSV.write(output_calib_file, df_out)
    println(output_calib_file)
end



main()
# quick_check()