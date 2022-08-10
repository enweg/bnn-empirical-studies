# Things to measure
# - Single chain performance 
#   - MSE/RMSE
#   - MAPE 
#   - Mixing in Sigma 
#   - VaR performance at 0.1%, 1%, 5%, 10%
# - All Chains using the same network with same performance measures
# - All Chains across all network architectures

using JLD
using MCMCChains
using BFlux, Flux
using Random, Distributions
using DataFrames, CSV
using Dates
using Logging

# logger(msg) = println("@$(Dates.now()) ==> $msg")
# logger("Welcome. Setting up evaluation.")


io = open("log-single-asset-evaluation.txt", "w+")
logger = SimpleLogger(io)
flush(io)
global_logger(logger)

function logme(msg)
    global io
    @info("$(Dates.now()) ==> $msg")
    flush(io)
end

data = load("./data/single-asset.jld")
x_test = data["x-test"]
y_test = data["y-test"]
logme("Data has been loaded.")

# This loads network_structures 
include("./network_structures.jl")

function get_bnn(net, x_test, y_test)
    nc = destruct(net)
    prior = GaussianPrior(nc, 0.5f0)
    like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
    init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
    bnn = BNN(x_test, y_test, like, prior, init)
    return bnn
end

mcmc_chains = readdir("./mcmc-chains/")
configs = [match(r"-([0-9]*)\.jld", ch_name)[1] for ch_name in mcmc_chains]
get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
VaR_levels = [0.001, 0.01, 0.05, 0.1] 
logme("Setup is ready.")

################################################################################
#### Single Chain Performance

logme("### Single Chain Evaluation")
df_single_chain = DataFrame()
df_lock = Threads.SpinLock()
io_lock = Threads.SpinLock()
todo = length(mcmc_chains)
done = 0
Threads.@threads for i in 1:length(mcmc_chains)
    chain_config = configs[i]
    try
        Threads.lock(io_lock) do 
            logme("Starting Evaluating Config $chain_config")
        end
        chain_data = load(joinpath("mcmc-chains", mcmc_chains[i]))
        chain = chain_data["chain"]
        net_used = network_structures[chain_data["netid"]]

        mixing_sigma = summarystats(MCMCChains.Chains(chain'))[:, 7][end]

        bnn = get_bnn(net_used, x_test, y_test)
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
        global df_single_chain
        Threads.lock(df_lock) do 
            df_single_chain = append!(df_single_chain, df)
        end
        global done
        Threads.lock(io_lock) do 
            done += 1
            logme("Done Evaluating Config $chain_config; ==> $done/$todo")
        end
    catch e
        Threads.lock(io_lock) do 
            logme("Errors in config $chain_config with message $e")
        end
    end
end
logme("Writing mcmc-single-chain-evaluation.csv")
CSV.write("./evaluations/mcmc-single-chain-evaluation.csv", df_single_chain)    

################################################################################
#### Performance across all chains using same network

# logme("### Performance of all chains using same network")
# network_files = Dict()
# for i in 1:length(mcmc_chains)
#     chain_data = load(joinpath("mcmc-chains", mcmc_chains[i]))
#     files = get!(network_files, chain_data["netid"], String[])
#     push!(files, mcmc_chains[i])
# end
# network_files = [(k, v) for (k, v) in network_files]

# df_same_network = DataFrame()
# df_lock = Threads.SpinLock()
# io_lock = Threads.SpinLock()
# todo = length(network_files)
# done = 0
# Threads.@threads for i in 1:todo
#     k, v = network_files[i]
#     try
#         Threads.lock(io_lock) do 
#         logme("Network $k Starting") 
#         end
#         ypp_all = missing
#         for chain_name in v
#             Threads.lock(io_lock) do
#                 global chain_data
#                 chain_data = load(joinpath("mcmc-chains", chain_name))
#             end
#             chain = chain_data["chain"]
#             net_used = network_structures[chain_data["netid"]]

#             bnn = get_bnn(net_used, x_test, y_test)
#             posterior_predict_test = sample_posterior_predict(bnn, chain; x = x_test)
#             ypp_all = ismissing(ypp_all) ? posterior_predict_test : hcat(ypp_all, posterior_predict_test)
#         end
#         Threads.lock(io_lock) do 
#         logme("Network $k collected all chains") 
#         end

#         yhat = vec(mean(ypp_all; dims = 2))
#         mse = mean(abs2, y_test .- yhat)
#         rmse = sqrt(mse)
#         mape = mean((y_test .- yhat)./y_test)

#         VaRs = Dict(alpha => get_VaR(ypp_all, alpha) for alpha in VaR_levels)
#         VaR_performance = Dict(key => mean(y_test .< v) for (key, v) in zip(keys(VaRs), values(VaRs)))

#         df = DataFrame(
#             "network" => k, 
#             "rmse" => rmse, 
#             "mape" => mape
#         )
#         for (kvar, vvar) in VaRs 
#             df[!, Symbol("VaR_"*replace(string(kvar*100), "."=>"_"))] .= mean(y_test .< vvar)
#         end

#         global df_same_network
#         Threads.lock(df_lock) do 
#         df_same_network = append!(df_same_network, df) 
#         end
#         global done 
#         Threads.lock(io_lock) do 
#             done += 1
#         logme("Network $k Done ==> $done/$todo") 
#         end
#     catch e
#         Threads.lock(io_lock) do 
#            logme("Error in network $k with message $e") 
#         end
#     end
# end
# logme("Writing mcmc-same-network-evaluation.csv")
# CSV.write("./evaluations/mcmc-same-network-evaluation.csv", df_same_network)

# close(io)
