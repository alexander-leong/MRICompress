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
  img_new = copy(img);
  img_new[1:end, 1:Int64(end/2)] = img[1:end, Int64((end/2)+1):end];
  img_new[1:end, Int64((end/2)+1):end] = img[1:end, 1:Int64(end/2)];
  img_new_new = copy(img_new);
  img_new_new[1:Int64(end/2), 1:end] = img_new[Int64((end/2)+1):end, 1:end];
  img_new_new[Int64((end/2)+1):end, 1:end] = img_new[1:Int64(end/2), 1:end];
  return img_new_new;
end