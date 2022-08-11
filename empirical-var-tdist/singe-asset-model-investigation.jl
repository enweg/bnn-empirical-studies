# We observe that higher nu performs better for high VaR but worse for 
# lowe VaR. This is weird since low df should have fatter tails than high df and
# should thus always have VaR forecasts lower than high df. So why do we observe
# a reversal? 

using JLD
using MCMCChains
using BFlux, Flux
using Random, Distributions
using DataFrames, CSV
using Dates
using Logging
using SharedArrays
using StatsPlots

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

data = load("./data/single-asset.jld")
x_test = data["x-test"]
y_test = data["y-test"]
logme("Data has been loaded.")

# This loads network_structures 

include("./network_structures.jl")
target_accepts = [0.1f0, 0.5f0]
betas = [0.01f0, 0.5f0, 0.99f0]
steps = [1, 2, 3]
nus = collect(3f0:1f0:10f0)
nus = vcat(nus, [20f0, 30f0])

ggmc_configs = vec(collect(Iterators.product(network_structures, target_accepts, betas, steps, nus)))

mcmc_chains = ["chain-10.jld", "chain-694.jld"]
mcmc_chains = filter(x -> x[1:5]=="chain", mcmc_chains)
configs = [match(r"-([0-9]*)\.jld", ch_name)[1] for ch_name in mcmc_chains]
VaR_levels = SharedArray([0.001, 0.01, 0.05, 0.1])

################################################################################
#### Single Chain Performance

function get_yhat_VaR(i)
    chain_config = configs[i]
    nu = ggmc_configs[parse(Int, chain_config)][5]

    chain_data = load(joinpath("mcmc-chains", mcmc_chains[i]))
    chain = chain_data["chain"]
    net_used = network_structures[chain_data["netid"]]
    vol = vec(chain[end, :])

    logme("Config $chain_config => Calculating yhat")
    bnn = get_bnn(net_used, x_test, y_test, nu)
    posterior_predict_test = sample_posterior_predict(bnn, chain; x = x_test)
    yhat = vec(mean(posterior_predict_test; dims = 2))

    logme("Config $chain_config => Calculting VaR")
    VaRs = Dict(alpha => get_VaR(posterior_predict_test, alpha) for alpha in VaR_levels)
    
    return posterior_predict_test, yhat, VaRs, vol
end

ypp_10, yhat_10, VaR_10, vol10 = get_yhat_VaR(1)
ypp_649, yhat_694, VaR_694, vol649 = get_yhat_VaR(2)


p = plot(yhat_10, label = "nu=3", color = :black);
p = plot!(p, yhat_694, label = "nu=30", color = :red);
# p = plot!(p, y_test);
p
Plots.pdf(p, "./evaluations/results/conditional-mean.pdf")

q01 = quantile(y_test, 0.001)
q1 = quantile(y_test, 0.01)
select01 = y_test .<= q01
select1 = y_test .<= q1

findfirst(select01)
p = density(ypp_10[244, :]; color = :black, label = "nu=3");
p = density!(p, ypp_649[244, :]; color = :red, label = "nu=30");
p = vline!(p, [quantile(ypp_10[244, :], 0.1)], color = :black, linestyle = :dash, label = missing);
p = vline!(p, [quantile(ypp_649[244, :], 0.1)], color = :red, linestyle = :dash, label = missing);
p = vline!(p, [quantile(ypp_10[244, :], 0.001)], color = :black, linestyle = :dash, label = missing);
p = vline!(p, [quantile(ypp_649[244, :], 0.001)], color = :red, linestyle = :dash, label = missing);
p
Plots.pdf(p, "./evaluations/results/posterior-preditive-distribution-obs244.pdf")

mean(ypp_10[244, :])
mean(ypp_649[244, :])

mean(exp.(vol10))
mean(exp.(vol649))

violated10_q01 = [y < v ? v : missing for (y, v) in zip(y_test, VaR_10[0.001])]

plot(VaR_10[0.001])
plot!(VaR_694[0.001])
plot!(y_test)

plot(VaR_10[0.1])
plot!(VaR_694[0.1])
plot!(y_test)