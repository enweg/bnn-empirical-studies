# Things to measure
# - Single chain performance 
#   - MSE/RMSE
#   - MAPE 
#   - Mixing in Sigma 
#   - VaR performance at 0.1%, 1%, 5%, 10%
# - All Chains using the same network with same performance measures
# - All Chains across all network architectures
using Distributed
addprocs(50)
using Pkg
pmap(i -> Pkg.activate("."), workers())
pmap(i -> Pkg.instantiate(), workers())

@everywhere begin
    using JLD
    using MCMCChains
    using BFlux, Flux
    using Random, Distributions
    using DataFrames, CSV
    using Dates
    using Logging
    using SharedArrays
    
    function get_bnn(net, x_train, y_train, nu)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        like = SeqToOneTDist(nc, Gamma(2.0, 0.5), nu)
        init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
        bnn = BNN(x_train, y_train, like, prior, init)
        return bnn
    end

    get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]

    logme(msg) = @info("$(Dates.now()) => $msg")
end

data = load("data/multiple-asset.jld")
x_test = data["x-test"]
y_test = data["y-test"]
logme("Data has been loaded.")

# This loads network_structures 

include("./multiple_network_structures.jl")
target_accepts = [0.5f0]
betas = [0.5f0]
steps = [3]
nus = collect(3f0:1f0:10f0)
nus = vcat(nus, [20f0, 30f0])

ggmc_configs = vec(collect(Iterators.product(network_structures, target_accepts, betas, steps, nus)))

mcmc_chains = readdir("./mcmc-chains-multiple/")
mcmc_chains = filter(x -> x[1:5]=="chain", mcmc_chains)
configs = [match(r"-([0-9]*)\.jld", ch_name)[1] for ch_name in mcmc_chains]
VaR_levels = SharedArray([0.001, 0.01, 0.05, 0.1])

################################################################################
#### Single Chain Performance

@sync @distributed for i in 1:lastindex(mcmc_chains)
    chain_config = configs[i]
    nu = ggmc_configs[parse(Int, chain_config)][5]
    try
        logme("Config $chain_config => Loading Data")
        chain_data = load(joinpath("mcmc-chains-multiple", mcmc_chains[i]))
        chain = chain_data["chain"]
        net_used = network_structures[chain_data["netid"]]

        logme("Config $chain_config => Calculating Mixing")
        mixing_sigma = summarystats(MCMCChains.Chains(chain'))[:, 7][end]

        logme("Config $chain_config => Calculating Performance")
        bnn = get_bnn(net_used, x_test, y_test, nu)
        posterior_predict_test = sample_posterior_predict(bnn, chain; x = x_test)
        yhat = vec(mean(posterior_predict_test; dims = 2))
        mse = mean(abs2, y_test .- yhat)
        rmse = sqrt(mse)
        mape = mean((y_test .- yhat)./y_test)

        VaRs = Dict(alpha => get_VaR(posterior_predict_test, alpha) for alpha in VaR_levels)

        df = DataFrame(
            "config" => chain_config, 
            "network" => chain_data["netid"], 
            "rmse" => rmse, 
            "mape" => mape, 
            "rhat_sigma" => mixing_sigma,
        )
        for (key, v) in zip(keys(VaRs), values(VaRs))
            df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(y_test .< v)
        end

        CSV.write("./evaluations/mcmc-single-chain-evaluation-multiple-config-$chain_config.csv", df)
        logme("Config $chain_config => Done")
    catch e
        logme("Config $chain_config => Error $e")
    end
end

files = readdir("./evaluations/")
files = files[occursin.("mcmc-single-chain-evaluation-multiple-config", files)]
files = ["./evaluations/$f" for f in files]
evaluation = DataFrame(CSV.File(files))
CSV.write("./evaluations/mcmc-single-chain-evaluation-multiple-all.csv", evaluation)


# Adding nu information
df = DataFrame(CSV.File("./evaluations/mcmc-single-chain-evaluation-multiple-all.csv"))
df[!, "nu"] = [ggmc_configs[config][5] for config in df[!, "config"]]
CSV.write("./evaluations/mcmc-single-chain-evaluation-multiple-all.csv", df)

###############################################################################
### Performance across all chains using same network

logme("### Performance of all chains using same network")
network_files = Dict()
for i in 1:length(mcmc_chains)
    chain_data = load(joinpath("mcmc-chains-multiple", mcmc_chains[i]))
    files = get!(network_files, chain_data["netid"], String[])
    push!(files, mcmc_chains[i])
end
network_files = [(k, v) for (k, v) in network_files]

todo = length(network_files)
@sync @distributed for i in 1:todo
    k, v = network_files[i]
    try
        logme("Network $k => Starting")
        ypp_all = missing
        for chain_name in v
            chain_data = load(joinpath("mcmc-chains-multiple", chain_name))
            chain = chain_data["chain"]
            net_used = network_structures[chain_data["netid"]]
            config = match(r"-([0-9]*)\.jld", chain_name)[1] 
            nu = ggmc_configs[parse(Int, config)][5]

            bnn = get_bnn(net_used, x_test, y_test, nu)
            posterior_predict_test = sample_posterior_predict(bnn, chain; x = x_test)
            ypp_all = ismissing(ypp_all) ? posterior_predict_test : hcat(ypp_all, posterior_predict_test)
        end
        logme("Network $k => Got all ypp")

        yhat = vec(mean(ypp_all; dims = 2))
        mse = mean(abs2, y_test .- yhat)
        rmse = sqrt(mse)
        mape = mean((y_test .- yhat)./y_test)

        VaRs = Dict(alpha => get_VaR(ypp_all, alpha) for alpha in VaR_levels)
        VaR_performance = Dict(key => mean(y_test .< v) for (key, v) in zip(keys(VaRs), values(VaRs)))

        df = DataFrame(
            "network" => k, 
            "rmse" => rmse, 
            "mape" => mape
        )
        for (kvar, vvar) in VaRs 
            df[!, Symbol("VaR_"*replace(string(kvar*100), "."=>"_"))] .= mean(y_test .< vvar)
        end
        CSV.write("./evaluations/mcmc-same-network-evaluation-multiple-chunk$k.csv", df)
        logme("Network $k => Done")
    catch e
        logme("Network $k => Error $e")
    end
end


files = readdir("./evaluations/")
files = files[occursin.("mcmc-same-network-evaluation-multiple-chunk", files)]
files = ["./evaluations/$f" for f in files]
evaluation = DataFrame(CSV.File(files))

CSV.write("./evaluations/mcmc-same-network-evaluation-multiple-all.csv", evaluation)

