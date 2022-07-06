using HDF5

export kspace_to_img

"""
    kspace_to_img(img_path)

Converts an image in k-space to pixel values via the IFFT.
"""
function kspace_to_img(img_path="")
  fid = h5open(img_path, "r");
  kspace = fid["kspace"];
  img = broadcast(abs, ifft(kspace[:, :, end-1]));
  img_new = ifftshift(img);
  return img_new;
end