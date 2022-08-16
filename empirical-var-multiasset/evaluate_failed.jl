# Some evaluations in estimation.jl failed because the workers could 
# not talk to the master. They still saved their intermediate results though
# and thus we can evaluate them separately. 
using Distributed
addprocs(50)
using Pkg
pmap(i -> Pkg.activate("."), workers())
pmap(i -> Pkg.instantiate(), workers())

@everywhere begin 
    using BFlux, Flux
    using Random, Distributions
    using CSV, DataFrames
    using DataFramesMeta
    using Dates
    using Serialization
    using Suppressor
    using JLD
    using SharedArrays
    using StatsBase

    logme(msg) = @info("$(Dates.now()) ==> $msg")
end


logme("Loading Data")

scaling_factor = 100f0
data = load("./data/multiple-asset.jld")
x_test = data["x-test"]
y_test = data["y-test"]

n_in = size(data["x-train"], 2)
n_out = size(data["y-train"], 1)

network_structures = [
    Flux.Chain(RNN(n_in, n_in), Dense(n_in, n_out)),
    Flux.Chain(RNN(n_in, n_out), Dense(n_out, n_out)),
    Flux.Chain(LSTM(n_in, n_in), Dense(n_in, n_out)),
    Flux.Chain(LSTM(n_in, n_out), Dense(n_out, n_out)),
]

target_accepts = [0.5f0]
betas = [0.5f0]
steps = [3]
m0s = collect(20f0:20f0:200f0)

ggmc_configs = vec(collect(Iterators.product(network_structures, target_accepts, betas, steps, m0s)))

@everywhere begin
    include("./multi-asset-likelihood-gaussian.jl")
    function get_bnn(net, x, y, m0 = 20)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        # like = MultiAssetGaussian(nc, Gamma(2.0, 0.5), size(y_train, 1), 1f0)
        # taking an empirical Bayes approach and use first 1/5 of observation 
        # to calculate scale matrix
        to_idx = floor(Int, size(y, 2)/5)
        scale_matrix = (m0-size(y, 1)-1)*StatsBase.cov(y[:, 1:to_idx]')
        like = MultiAssetGaussian(nc, InverseWishart(m0, scale_matrix))
        init = InitialiseAllSame(Normal(0, 0.5), like, prior)

        bnn = BNN(x, y, like, prior, init)
        return bnn
    end
end

logme("Determining work to do")

# getting all results files
files_chains = readdir("./mcmc-chains/")
files_cahins = filter(x -> occursin("chain-", x), files_chains)
configs_estimated = [parse(Int, match(r"chain-([0-9]*)", f)[1]) for f in files_chains]

files_evaluated = readdir("./evaluations/")
files_evaluated = filter(x -> occursin("mcmc-single-chain-config", x), files_evaluated)
configs_evaluated = [parse(Int, match(r"mcmc-single-chain-config([0-9]*)", f)[1]) for f in files_evaluated] 

configs_todo = filter(x -> !(x in configs_evaluated), configs_estimated)

logme("Starting Evaluations")

@sync @distributed for i=1:lastindex(configs_todo)
    config = configs_todo[i]
    logme("Starting config $config")
    try
        chain = load(joinpath("mcmc-chains", "chain-$config.jld"))
        net, _, _, _, m0 = ggmc_configs[config]
        netid = findfirst(x -> x==net, network_structures)
        ch = chain["chain"]
        bnn = get_bnn(net, data["x-train"], data["y-train"], m0)

        ypp = sample_posterior_predict(bnn, ch; x = x_test)
        pfolio_return_hat = vec(mean(ypp; dims = 2))
        pfolio_return = vec(mean(y_test; dims = 1))

        rmse = sqrt(mean(abs2, (pfolio_return .- pfolio_return_hat)/scaling_factor))

        VaR_levels = [0.001, 0.01, 0.05, 0.1]
        get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
        VaRs = Dict(alpha => get_VaR(ypp, alpha) for alpha in VaR_levels)

        df = DataFrame(
            "config" => config,
            "netid" => netid,
            "rmse" => rmse,
            "m0" => m0
        )
        for (key, v) in zip(keys(VaRs), values(VaRs))
            df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(pfolio_return .< v)
        end

        CSV.write("evaluations/mcmc-single-chain-config$config.csv", df)
        logme("Done config $config")
    catch e
        logme("Error in config $config ==> $e")
    end
end