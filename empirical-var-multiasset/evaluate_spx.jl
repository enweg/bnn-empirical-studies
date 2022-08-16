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

spx_index = 7

data = load("./data/multiple-asset.jld")
x_test = data["x-test"]
y_test = data["y-test"]
spx_test = y_test[spx_index, :]

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

@everywhere begin
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

files = readdir("./mcmc-chains/")
files = filter(x -> occursin("chain-", x), files)
configs = [parse(Int, match(r"chain-([0-9]*)", f)[1]) for f in files] 

function predict_spx(bnn, θ, x)
    nethat = bnn.like.nc(θ)
    ypp = [nethat(xx) for xx in eachslice(x; dims = 1)][end]
    return ypp[spx_index, :]
end


logme("Starting Evaluations")

@sync @distributed for i=1:lastindex(configs)
    config = configs[i]
    logme("Starting config $config")
    try
        chain = load(joinpath("mcmc-chains", "chain-$config.jld"))
        netid = chain["netid"]
        net = network_structures[netid]
        ch = chain["chain"]
        bnn = get_bnn(net, x_test, y_test)

        ypp_spx = reduce(hcat, [predict_spx(bnn, ch[:, i], x_test) for i=1:size(ch, 2)])
        spx_pred = vec(mean(ypp_spx; dims = 2))

        rmse = sqrt(mean(abs2, spx_test .- spx_pred))

        df = DataFrame(
            "config" => config,
            "netid" => netid,
            "rmse" => rmse
        )

        CSV.write("evaluations/spx-config$config.csv", df)
        logme("Done config $config")
    catch e
        logme("Error in config $config ==> $e")
    end
end