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


    include("./multiple_network_structures.jl")
    function get_bnn(net, x_train, y_train)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
        init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
        bnn = BNN(x_train, y_train, like, prior, init)
        return bnn
    end
end

logme("Determining work to do")

# getting all results files
files_chains = readdir("./mcmc-chains-multiple/")
files_cahins = filter(x -> occursin("chain-", x), files_chains)
configs_estimated = [parse(Int, match(r"chain-([0-9]*)", f)[1]) for f in files_chains]
configs_todo = configs_estimated

logme("Starting Evaluations")

@everywhere function get_pfolio_sd(config)
    logme("Starting config $config")
    try
        chain = load(joinpath("mcmc-chains-multiple", "chain-$config.jld"))
        ch = chain["chain"]
        pfolio_sd = mean(sqrt.(exp.(vec(ch[end, :]))))
        return pfolio_sd

    catch e
        logme("Error in config $config ==> $e")
    end
    logme("Done config $config")
    return missing
end

pfolio_sds = pmap(get_pfolio_sd, configs_todo)
CSV.write("./evaluations/pfolio-sds.csv", DataFrame("sd" => pfolio_sds))
