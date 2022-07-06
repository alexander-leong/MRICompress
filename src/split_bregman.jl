include("util.jl")

using Deconvolution
using FFTW
using ImageFiltering
using LinearAlgebra
using Noise
using Printf
using SparseArrays
using ToeplitzMatrices

function shrink(x, gamma)
  return (x/norm(x, 1)) * max(norm(x, 1) - gamma, 0)
end

export split_bregman_anisotropic_tv_denoising

"""
    split_bregman_anisotropic_tv_denoising(img, lambda, mu, tol, max_iterations)

Implementation based on The Split Bregman Method For L1 Regularized Problems Goldstein, Osher
Uses Dirichlet Boundary Conditions, zero at the boundary.
Returns a denoised image.

# Examples
```julia-repl
julia> img = kspace_to_img();
julia> split_bregman_anisotropic_tv_denoising(img, 1, 1, 0, 1);
```
"""
function split_bregman_anisotropic_tv_denoising(img, lambda=1, mu=1, tol=1e-3, max_iterations=1)
  f = zeros((size(img)[1]+2, size(img)[2]+2))
  f[2:end-1, 2:end-1] = img
  u_next = f
  u_prev = f
  d_x = zeros((size(img)[1]+2, size(img)[2]+2))
  d_y = zeros((size(img)[1]+2, size(img)[2]+2))
  b_x = zeros((size(img)[1]+2, size(img)[2]+2))
  b_y = zeros((size(img)[1]+2, size(img)[2]+2))
  i = 0
  # while i < max_iterations && norm(u_next - u_prev, 2) > tol
  while i < max_iterations
    @printf "Norm of difference: %.6f\n" norm(u_next - u_prev, 2)
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
  return u_next[2:end-1,2:end-1]
end

"""
    split_bregman_isotropic_tv_denoising(img, lambda, mu, tol, max_iterations)

Implementation based on The Split Bregman Method For L1 Regularized Problems Goldstein, Osher
Uses Dirichlet Boundary Conditions, zero at the boundary.
Returns a denoised image.

# Examples
```julia-repl
julia> img = kspace_to_img();
julia> split_bregman_isotropic_tv_denoising(img, 1, 1, 0, 1);
```
"""
function split_bregman_isotropic_tv_denoising(img, lambda=1, mu=1, tol=1e-3, max_iterations=1)
  f = zeros((size(img)[1]+2, size(img)[2]+2))
  f[2:end-1, 2:end-1] = img
  u_next = f
  u_prev = f
  d_x = zeros((size(img)[1]+2, size(img)[2]+2))
  d_y = zeros((size(img)[1]+2, size(img)[2]+2))
  b_x = zeros((size(img)[1]+2, size(img)[2]+2))
  b_y = zeros((size(img)[1]+2, size(img)[2]+2))
  i = 0
  # while i < max_iterations && norm(u_next - u_prev, 2) > tol
  while i < max_iterations
    u_prev = copy(u_next)
    # solve l2 problem by Gauss-Seidel method
    G = ((lambda / (mu + 4*lambda)) * (u_next[3:end,2:end-1] + u_next[1:end-2,2:end-1] + u_next[2:end-1,3:end] + u_next[2:end-1,1:end-2] + d_x[2:end-1,2:end-1] - d_x[3:end,2:end-1] + d_y[2:end-1,2:end-1] - d_y[2:end-1,3:end] - b_x[2:end-1,2:end-1] + b_x[3:end,2:end-1] - b_y[2:end-1,2:end-1] + b_y[2:end-1,3:end])) + ((mu / mu + 4*lambda) * img)
    u_next[2:end-1, 2:end-1] = G
    # calculate s_k
    s_k = sqrt.((u_next[2:end-1,2:end-1] .- u_next[3:end,2:end-1] .+ b_x[2:end-1, 2:end-1]) .^ 2 .+ (u_next[2:end-1,2:end-1] .- u_next[2:end-1,3:end] .+ b_y[2:end-1, 2:end-1]) .^ 2)
    # calculate d_x, d_y
    d_x[2:end-1,2:end-1] = max.((s_k .- 1) ./ lambda, 0) .* (u_next[2:end-1,2:end-1] .- u_next[3:end,2:end-1] .+ b_x[2:end-1, 2:end-1]) ./ s_k
    d_y[2:end-1,2:end-1] = max.((s_k .- 1) ./ lambda, 0) .* (u_next[2:end-1,2:end-1] .- u_next[2:end-1,3:end] .+ b_y[2:end-1, 2:end-1]) ./ s_k
    # calculate b_x, b_y
    b_x[2:end-1,2:end-1] = b_x[2:end-1,2:end-1] + (u_next[2:end-1,2:end-1] - u_next[3:end,2:end-1] - d_x[2:end-1,2:end-1])
    b_y[2:end-1,2:end-1] = b_y[2:end-1,2:end-1] + (u_next[2:end-1,2:end-1] - u_next[2:end-1,3:end] - d_y[2:end-1,2:end-1])
    i += 1
  end
  return u_next[2:end-1,2:end-1]
end

sdiff1(M) = sparse([ [1.0 zeros(1,M-1)]; diagm(1 => ones(M-1)) - I(M)])

function laplacian(Nx, Ny, Lx, Ly)
  dx = Lx / (Nx)
  dy = Ly / (Ny)
  Dx = sdiff1(Nx) / dx
  Dy = sdiff1(Ny) / dy
  Ax = Dx' * Dx
  Ay = Dy' * Dy
  return kron(sparse(I, Ny, Ny), Ax) + kron(Ay, sparse(I, Nx, Nx))
end

function split_bregman_anisotropic_isotropic_tv_denoising(img, lambda=1, mu=1, c=1, alpha=0.5, tol=1e-3, max_iterations=1, max_iterations_dca=1)
  img = img[1:Int(end/4), 1:Int(end/4)]
  u_next = zeros((size(img)[1]*size(img)[2]))
  q_x = zeros((size(img)[1]*size(img)[2]))
  q_y = zeros((size(img)[1]*size(img)[2]))
  d_x = zeros((size(img)[1]*size(img)[2]+1))
  d_y = zeros((size(img)[1]*size(img)[2]+1))
  D_xu, D_yu = zeros(size(img)), zeros(size(img))
  f = reshape(img, (size(img)[1]*size(img)[2], 1))
  I_d = Matrix(I, size(img)[1]*size(img)[2], size(img)[1]*size(img)[2])
  diff2 = laplacian(size(img)[1], size(img)[2], 1, 1)
  # inv_toeplitz = trench(SymmetricToeplitz(mu*I_d - lambda*diff2 + 2*c*I_d))
  # F = eigen(mu*I_d - lambda*diff2 + 2*c*I_d)
  # F_inv = F.vectors' * inv(Diagonal(F.values)) * F.vectors
  for i = 1:max_iterations_dca
    b_x = zeros((size(img)[1]*size(img)[2]+1))
    b_y = zeros((size(img)[1]*size(img)[2]+1))
    for j = 1:max_iterations
      # u_next = F_inv * (mu*alpha*I_d*f + lambda*((d_x-b_x)[1:end-1] - ((d_x-b_x)[2:end])) + lambda*((d_y-b_y)[1:end-1] - ((d_y-b_y)[2:end])) + 2*c*u_next)
      c = (2 / (size(img)[1]*size(img)[2]+1)) * sparse(I, size(img)[1]*size(img)[2])
      u_next = c * FFTW.r2r(lambda*((d_x-b_x)[1:end-1] - ((d_x-b_x)[2:end])) + lambda*((d_y-b_y)[1:end-1] - ((d_y-b_y)[2:end])) + 2*c*u_next, FFTW.RODFT00)
      N = size(img)[1]*size(img)[2]
      u_next = sparse(Diagonal(reshape([(2 - 2*cos(i*pi/(N+1) + 2 - 2*cos(j*pi/(N+1)))) for i in 1:size(img)[1], j in 1:size(img)[2]], (size(img)[1]*size(img)[2],)))) * u_next
      u_next = FFTW.r2r(u_next, FFTW.RODFT00)
      u_next = reshape(u_next, (size(img)[1], size(img)[2]))
      u_next = [zeros(size(u_next)[2]+2)'; zeros(size(u_next)[1]) u_next zeros(size(u_next)[1]); zeros(size(u_next)[2]+2)']
      D_xu = u_next[2:end-1,2:end-1] - u_next[3:end,2:end-1]
      D_yu = u_next[2:end-1,2:end-1] - u_next[2:end-1,3:end]
      u_next = u_next[2:end-1,2:end-1]
      u_next = reshape(u_next, (size(img)[1]*size(img)[2]))
      D_xu = reshape(D_xu, (size(img)[1]*size(img)[2], 1))
      D_yu = reshape(D_yu, (size(img)[1]*size(img)[2], 1))
      d_x[1:end-1] = shrink(D_xu + b_x[1:end-1] + alpha*q_x/lambda, 1/lambda)
      d_y[1:end-1] = shrink(D_yu + b_y[1:end-1] + alpha*q_y/lambda, 1/lambda)
      b_x[1:end-1] = b_x[1:end-1] + D_xu - d_x[1:end-1]
      b_y[1:end-1] = b_y[1:end-1] + D_yu - d_y[1:end-1]
    end
    q_x, q_y = (D_xu, D_yu) ./ sqrt(sum(D_xu .^ 2 .+ D_yu .^ 2))
  end
end

img = kspace_to_img("/home/alexander/Documents/alexander_leong/fastmri/MRICompress/src/file1002389.h5");
split_bregman_anisotropic_isotropic_tv_denoising(img, 1, 1, 1, 0.5, 1e-3, 1, 1);
