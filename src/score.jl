using LinearAlgebra
using Statistics

export mae
export mse
export nmse
export psnr
export rmse
export snr
export ssim

"""
    mae(x, y)

Calculates the mean absolute error
Reference: R.C. Gonzalez and R.E. Woods, "Digital Image Processing," Prentice Hall 2008
"""
function mae(x, y)
    return (1/length(x)) * sum(abs.(x - y))
end

"""
    mse(x, y)

Calculates the mean squared error
"""
function mse(x, y)
    return sum((x - y).^2) / length(x)
end

"""
    nmse(x, y)

Calculates the normalized mean squared error
"""
function nmse(x, y)
    return norm((x - y), 2) / norm(y, 2)
end

"""
    psnr(x, ref_x)

Calculates the peak signal to noise ratio in decibels given reference image
Reference: R.C. Gonzalez and R.E. Woods, "Digital Image Processing," Prentice Hall 2008
"""
function psnr(x, ref_x)
    return 10 * log10(maximum(ref_x)^2 / ((1/length(x)) * sum((ref_x - x).^2)))
end

"""
    rmse(x, y)

Calculates the root mean squared error
Reference: R.C. Gonzalez and R.E. Woods, "Digital Image Processing," Prentice Hall 2008
"""
function rmse(x, y)
    return sqrt(mse(x, y))
end

"""
    snr(x, ref_x)

Calculates the signal to noise ratio in decibels given reference image
Reference: R.C. Gonzalez and R.E. Woods, "Digital Image Processing," Prentice Hall 2008
"""
function snr(x, ref_x)
    return 10 * log10(sum(ref_x.^2) / sum((ref_x - x).^2))
end

"""
    ssim(x, y, k_1=0.01, k_2=0.03)

Calculates the structural similarity index measure (SSIM)
Note: 7*7 sized windows used in fastMRI: An Open Dataset and Benchmarks for Accelerated MRI (2019)
"""
function ssim(x, y, k_1=0.01, k_2=0.03)
    mu_x = mean(x)
    mu_y = mean(y)
    var_x = var(x)
    var_y = var(y)
    cov_x_y = cov(vec(x), vec(y))
    c_1 = (k_1*maximum(y))
    c_2 = (k_2*maximum(y))
    return ((2*mu_x*mu_y + c_1)*(2*cov_x_y + c_2)) / ((mu_x^2 + mu_y^2 + c_1)*(var_x + var_y + c_2))
end