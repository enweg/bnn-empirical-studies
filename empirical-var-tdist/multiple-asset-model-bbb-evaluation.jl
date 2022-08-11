using BFlux
using Flux
using LinearAlgebra
using Random, Distributions
using CSV
using DataFrames
using JLD
using PDMats

data = load("data/multiple-asset.jld")
include("./multiple_network_structures.jl")
nus = collect(3f0:1f0:10f0)
nus = vcat(nus, [20f0, 30f0])

bbb_configs = vec(collect(Iterators.product(network_structures, nus)))

function get_bnn(net, x_train, y_train, nu)
    nc = destruct(net)
    prior = GaussianPrior(nc, 0.5f0)
    like = SeqToOneTDist(nc, Gamma(2.0, 0.5), nu)
    init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
    bnn = BNN(x_train, y_train, like, prior, init)
    return bnn
end

files = readdir("./bbb-objects-multiple/")
files = filter(x -> x != ".DS_Store", files)
files = joinpath.("bbb-objects-multiple", files)

df_all = DataFrame()

for f in files
    vi = load(f)
    netid = vi["netid"]
    @info("Working in netowork $netid")
    Random.seed!(6150533)
    ch = rand(vi["bbb"][1], 20_000)

    net = network_structures[netid]
    config = match(r"net([0-9]*)\.jld", f)[1]
    nu = bbb_configs[parse(Int, config)][2]

    bnn = get_bnn(net, data["x-test"], data["y-test"], nu)
    ypp = sample_posterior_predict(bnn, ch)

    VaR_levels = [0.001, 0.01, 0.05, 0.1]
    get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]

    yhat = vec(mean(ypp; dims = 2))
    mse = mean(abs2, data["y-test"] .- yhat)
    rmse = sqrt(mse)

    VaRs = Dict(alpha => get_VaR(ypp, alpha) for alpha in VaR_levels)

    df = DataFrame(
        "network" => vi["netid"], 
        "rmse" => rmse, 
        "nu" => nu,
    )
    for (key, v) in zip(keys(VaRs), values(VaRs))
        df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(data["y-test"] .< v)
    end
    global df_all
    df_all = append!(df_all, df)
end

df_all
CSV.write("./evaluations/bbb-multiple-asset-all.csv", df_all)
