include("util.jl")
using Deconvolution
using Noise

export wiener_deconv

"""
    wiener_deconv(img, orig, sigma)

Returns a denoised image.
"""
function wiener_deconv(img, orig, sigma)
  noise = add_gauss(zeros(size(orig)), 1e-8*sigma)
  img = wiener(orig, img, noise)
  img = [vec(img[:,i]) for i in 1:size(img)[2]]
  return img
end