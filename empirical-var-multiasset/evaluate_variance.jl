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

    logme(msg) = @info("$(Dates.now()) ==> $msg")
end


logme("Loading Data")


@everywhere begin
    data = load("./data/multiple-asset.jld")
    x_test = data["x-test"]
    y_test = data["y-test"]

    n_in = size(data["x-train"], 2)
    n_out = size(data["y-train"], 1)
    network_structures = [
        Flux.Chain(RNN(n_in, n_in), Dense(n_in, n_out)),
        Flux.Chain(RNN(n_in, n_out), Dense(n_out, n_out)),
        Flux.Chain(RNN(n_in, 2*n_in), Dense(2*n_in, n_out)),
        Flux.Chain(LSTM(n_in, n_in), Dense(n_in, n_out)),
        Flux.Chain(LSTM(n_in, n_out), Dense(n_out, n_out)),
        Flux.Chain(LSTM(n_in, 2*n_in), Dense(2*n_in, n_out)),
    ]

    include("./multi-asset-likelihood-gaussian.jl")
    function get_bnn(net, x, y)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        like = MultiAssetGaussian(nc, Gamma(2.0, 0.5), size(y, 1), 1f0)
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
configs_todo = configs_estimated

logme("Starting Evaluations")

@everywhere function get_pfolio_sd(config)
    logme("Starting config $config")
    try
        chain = load(joinpath("mcmc-chains", "chain-$config.jld"))
        netid = chain["netid"]
        net = network_structures[netid]
        ch = chain["chain"]
        bnn = get_bnn(net, x_test, y_test)

        pfolio_sd = Array{Float32}(undef, size(ch, 2))
        for c=1:size(ch, 2)
            θnet, θhyper, θlike = split_params(bnn, ch[:, c]) 
            _, cov = get_yhat_covariance(bnn.like, Float32.(θnet), Float32.(θlike), x_test)
            pfolio_sd[c] = sqrt(fill(1/12, 12)'*cov*fill(1/12, 12))
        end

        return mean(pfolio_sd)
    catch e
        logme("Error in config $config ==> $e")
    end
    logme("Done config $config")
    return missing
end

pfolio_sds = pmap(get_pfolio_sd, configs_todo)
CSV.write("./evaluations/pfolio-sds.csv", DataFrame("sd" => pfolio_sds))
