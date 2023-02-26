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

function plot_mri(img)
  img_min, img_max = findmin(img)[1], findmax(img)[1]

  # normalize values between 0 and 1
  img = (img .- img_min) ./ (img_max - img_min)
  img = [vec(img[:,i]) for i in 1:size(img)[2]]
  return plot(heatmap(z=img, colorscale="Greys"), Layout(font_color="white", paper_bgcolor="#222222"))
end

function plot_raw_mri(img)
  img = plot_mri(img)
  return img
end