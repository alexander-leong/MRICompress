include("util.jl")

export block_svt
export svt

function admm_nnm(z, x_known_idx, num_iterations, lambda::Float64=1)
  S = zeros((minimum(size(z))))
  for _ = 1:num_iterations
    U, S, V = svd(z)
    for i in 1:size(Diagonal(S))[1]
      Diagonal(S)[i, i] = max(Diagonal(S)[i, i] - lambda, 0)
    end
    x = U * Diagonal(S) * transpose(V)
    z = x
  end
  return z
end

"""
    block_svt(z, c, patch_sz, num_iterations, lambda::Float64, overlap_y::Int64)

Reconstructs an image using weighted overlapped patches where an SVT is applied to each patch.
Patch size and overlap amount should be specified such that indexation errors are avoided.
Returns a compressed image using blocked singular value thresholding.
"""
function block_svt(z, c, patch_sz, num_iterations, lambda::Float64=1e-7, overlap_y::Int64=3)
  overlap_y = 2^overlap_y
	result = zeros((size(z)[1], size(z)[2]))
	for i in 1:Int(size(z)[1]/patch_sz[1]*2)-2
		for j in 1:Int(size(z)[2]/patch_sz[2]*overlap_y)-overlap_y
			idx_x = Int64(patch_sz[1]/2*i):Int64(patch_sz[1]/2*i)+patch_sz[1]
			idx_y = Int64(patch_sz[2]/overlap_y*j):Int64(patch_sz[2]/overlap_y*j)+patch_sz[2]
			result[idx_x, idx_y] += c * admm_nnm(z[idx_x, idx_y], [], num_iterations, lambda)
		end
	end
	return result
end

"""
    svt(img, lambda, num_iterations)

Returns a compressed image using singular value thresholding.
"""
function svt(img, lambda, num_iterations)
  result = admm_nnm(img, [], num_iterations, 1e-7*lambda)
  result = [vec(result[:,i]) for i in 1:size(result)[2]]
  return result
end