# include("autoencoder.jl")
include("score.jl")
include("split_bregman.jl")
include("svt.jl")
include("util.jl")
include("wiener.jl")

using PrettyTables

export compress
export evaluate_model
export load_model
export train_model

"""
    compress(method, files, args...)

Compresses a batch of images using a given method. 
method takes one of the following values: 'anisotropic\\_tv', 'blocked\\_svt', 'svt', 'wiener', 
files: an array of .h5 files, 
args: arguments supplied to the method
"""
function compress(method, files, args...)
    results = []
    for file in files
        img = kspace_to_img(file)
        if is_valid(file, img) == false
            continue
        end
        if method == "anisotropic_tv"
            x = split_bregman_anisotropic_tv_denoising(img, args...)
        elseif method == "isotropic_tv"
            x = split_bregman_isotropic_tv_denoising(img, args...)
        elseif method == "blocked_svt"
            x = block_svt(img, args...)
        elseif method == "svt"
            x = svt(img, args...)
        elseif method == "wiener"
            x = wiener_deconv(img, args...)
        end
        push!(results, [get_result(img, x)...]')
    end
    println("Compression results for method: $method")
    print_summary(results)
    println("Done!")
    println("")
    return results
end

function is_valid(file, img)
    if size(img) != (368, 640)
        println("Skipping: failed dimensionality check")
        return false
    end
    println("Processing: $file")
    return true
end

"""
    evaluate_model(model, file)

Evaluate the input using a trained model
"""
function evaluate_model(model, file)
  x = kspace_to_img(file)
  if is_valid(file, x) == false
      return
  end
    return model(x)
end

"""
    load_model(model_name)

Load the model from disk
"""
function load_model(model_name)
    load_autoencoder(model_name)
end

"""
    train_model(model_name, files)

Trains the model on the given files
"""
function train_model(model_name, files)
    return train_autoencoder(model_name, files)
end

function get_result(x, y)
    mae_result = mae(x, y)
    mse_result = mse(x, y)
    nmse_result = nmse(x, y)
    psnr_result = psnr(x, y)
    rmse_result = rmse(x, y)
    snr_result = snr(x, y)
    ssim_result = ssim(x, y)
    return mae_result, mse_result, nmse_result, psnr_result, rmse_result, snr_result, ssim_result
end

function print_summary(results)
    conf = set_pt_conf(tf = tf_markdown);
    header = ["MAE", "MSE", "NMSE", "PSNR", "RMSE", "SNR", "SSIM"]
    if length(results) > 0
        println("Average across dataset for each metric")
        pretty_table_with_conf(conf, mean(results), header=header)
        println("")
        println("Variance across dataset for each metric")
        pretty_table_with_conf(conf, var(results), header=header)
    end
end

compress("anisotropic_tv", ["file1002389.h5", "file1002389_copy.h5"])