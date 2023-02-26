# Introduction

Please note that MRICompress is in a pre-release stage which means that APIs are unstable or subject to change. Open to take suggestions or proposals on bugs and features. See Contributing for more information.

MRICompress is a Julia package that enables scientists and engineers to experiment and develop compressed sensing and denoising techniques for MRI reconstruction.

The goal is to develop a solution that allows practitioners the ability to ingest MRI data using the .h5 k-space format, perform compression and denoising operations, train and test algorithms, score methods and visualize results through a web application dashboard.

```@contents
Pages = ["reference.md", "review.md", "release_notes.md"]
Depth = 1
```

## Coming soon

* More compressed sensing and denoising methods
* Scoring methods
* Web dashboard
* Docker support

## Getting started

1 Sign up and download the MRI dataset from NYU Langone.

2 For single coil leg mri data (for example) you'll find a list of .h5 files.
You can ingest k-space data using the kspace to img function prior to calling any compression or denoising methods.

3 There are several compression and denoising methods to choose from: block svt, split bregman anisotropic tv denoising, svt and wiener deconv

4 Plot the reconstructed image.

5 Save the result.

## License

MRICompress is licensed under the MIT license. Consult the license for more information. In addition, MRICompress requires multiple dependencies to use which have their own licenses. Consult their package repositories for the specific licenses that apply.

## Citing MRICompress

If you are using MRICompress in your work, we kindly request that you cite the usage of this package as follows:

``` sourceCode
Title: MRI Compressed Sensing and Denoising in Julia
Year: 2023
Author: Alexander Leong
```

## Sponsorship

Please contact me at toshiba_alexander@live.com if you would like to sponsor my work.