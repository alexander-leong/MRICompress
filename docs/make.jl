push!(LOAD_PATH,"/home/alexander/Documents/alexander_leong/fastmri/MRICompress/src/")

using Documenter, MRICompress

makedocs(sitename="MRICompress")

deploydocs(
    repo = "github.com/alexander-leong/MRICompress.jl.git"
)