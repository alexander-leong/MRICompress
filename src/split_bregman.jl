include("util.jl")

using Deconvolution
using FFTW
using ImageFiltering
using LinearAlgebra
using Noise

function shrink(x, gamma)
  return (x/norm(x, 1)) * max(norm(x, 1) - gamma, 0)
end

export split_bregman_anisotropic_tv_denoising

"""
    split_bregman_anisotropic_tv_denoising(img, lambda, mu, tol, max_iterations)

Implementation based on The Split Bregman Method For L1 Regularized Problems Goldstein, Osher
Returns a denoised image.
Note: tol currently unused.

# Examples
```julia-repl
julia> img = kspace_to_img();
julia> split_bregman_anisotropic_tv_denoising(img, 1, 1, 0, 1);
```
"""
function split_bregman_anisotropic_tv_denoising(img, lambda, mu, tol, max_iterations)
  f = zeros((size(img)[1]+2, size(img)[2]+2))
  f[2:end-1, 2:end-1] = img
  u_next = f
  u_prev = f
  d_x = zeros((size(img)[1]+2, size(img)[2]+2))
  d_y = zeros((size(img)[1]+2, size(img)[2]+2))
  b_x = zeros((size(img)[1]+2, size(img)[2]+2))
  b_y = zeros((size(img)[1]+2, size(img)[2]+2))
  i = 0
  while i < max_iterations
    u_prev = copy(u_next)
    # solve l2 problem by Gauss-Seidel method
    G = ((lambda / (mu + 4*lambda)) * (u_next[3:end,2:end-1] + u_next[1:end-2,2:end-1] + u_next[2:end-1,3:end] + u_next[2:end-1,1:end-2] + d_x[2:end-1,2:end-1] - d_x[3:end,2:end-1] + d_y[2:end-1,2:end-1] - d_y[2:end-1,3:end] - b_x[2:end-1,2:end-1] + b_x[3:end,2:end-1] - b_y[2:end-1,2:end-1] + b_y[2:end-1,3:end])) + ((mu / mu + 4*lambda) * img)
    u_next[2:end-1, 2:end-1] = G
    # calculate d_x, d_y via shrink
    d_x[2:end-1,2:end-1] = shrink(u_next[2:end-1,2:end-1] - u_next[3:end,2:end-1] + b_x[2:end-1, 2:end-1], 1/lambda)
    d_y[2:end-1,2:end-1] = shrink(u_next[2:end-1,2:end-1] - u_next[2:end-1,3:end] + b_y[2:end-1, 2:end-1], 1/lambda)
    # calculate b_x, b_y
    b_x[2:end-1,2:end-1] = b_x[2:end-1,2:end-1] + (u_next[2:end-1,2:end-1] - u_next[3:end,2:end-1] - d_x[2:end-1,2:end-1])
    b_y[2:end-1,2:end-1] = b_y[2:end-1,2:end-1] + (u_next[2:end-1,2:end-1] - u_next[2:end-1,3:end] - d_y[2:end-1,2:end-1])
    i += 1
  end
  return u_next
end