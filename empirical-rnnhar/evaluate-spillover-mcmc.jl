using Distributed
addprocs(50)
@everywhere using Pkg
pmap(i -> Pkg.activate("."), workers())
pmap(i -> Pkg.instantiate(), workers())

@everywhere begin
    using BFlux, Flux
    using Random, Distributions
    using JLD
    using CSV, DataFrames
    using DataFramesMeta
    using Dates
    using Suppressor
    using StatsBase
    using MCMCChains
    include("utils.jl")
    logme(msg) = @info("$(Dates.now()) ==> $msg")
end

################################################################################
#### Setup

logme("Starting Setup")
data = load("./data/informationset-spillover.jld")
x_train = data["x-train"]

@everywhere begin
    include("rnn-har-likelihood.jl")
    function get_bnn(net, x, y, idx_daily, idx_weekly, idx_monthly)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        like = HARLikelihood(nc, Gamma(2.0, 0.5), idx_daily, idx_weekly, idx_monthly)
        init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
        bnn = BNN(x, y, like, prior, init)

        return bnn
    end
end

n_in = size(x_train, 2)
n_out = 4
network_structures = [
    Flux.Chain(RNN(n_in, n_in), Dense(n_in, n_out)), 
    Flux.Chain(RNN(n_in, 2*n_in), Dense(2*n_in, n_out)),
    Flux.Chain(LSTM(n_in, n_in), Dense(n_in, n_out)), 
    Flux.Chain(LSTM(n_in, 2*n_in), Dense(2*n_in, n_out)),
]

logme("Done setting up")
################################################################################
#### Evaluation

@everywhere function var_performance(alpha, ypp, y, mu)
    VaR = reduce(hcat, [[quantile(Normal(mu, sqrt(exp(rv))), alpha) for rv in ts] for ts in eachcol(ypp)])
    VaR_mean = vec(mean(VaR; dims = 2))
    return mean(y .< VaR_mean)
end

files = readdir("./chains-spillover/")
files = filter(x->occursin("chain-", x), files)
files = [joinpath("chains-spillover", f) for f in files]

logme("Starting evaluation")
@sync @distributed for i=1:lastindex(files)
    # getting chain data
    chain_data = load(files[i])
    netid = chain_data["netid"]
    rep = chain_data["rep"]
    net = network_structures[netid]
    chain = chain_data["chain"]
    logme("Starting with net $netid rep $rep")

    # mixing
    rhat_sigma = MCMCChains.summarystats(MCMCChains.Chains(chain'))[:, 7][end]
    df = DataFrame(
        "net" => netid, 
        "rep" => rep,
        "rhat_sigma" => rhat_sigma, 
    )

    # Analysing coefficients
    bnn = get_bnn(net, data["x-train"], data["y-train"], data["idx_daily"], data["idx_weekly"], data["idx_monthly"])
    coeffs = [calculate_yhat_and_coeffs(bnn.like, bnn.like.nc(chain[:, i]), data["x-test"])[2] for i=1:10]
    coeffs = Dict(variable => reduce(hcat, getfield.(coeffs, variable)) for variable in keys(first(coeffs)))

    coeffs_mean = reduce(hcat, [vec(mean(coeffs[variable]; dims = 2)) for variable in keys(coeffs)])
    coeffs_mean = DataFrame(coeffs_mean, [k for k in keys(coeffs)])

    CSV.write("./evaluations-spillover/coeffs-mean-net$netid-rep$rep.csv", coeffs_mean)

    # Posterior predictive analysis
    ypp = sample_posterior_predict(bnn, chain, x = data["x-test"])
    yhat = vec(mean(ypp; dims = 2))

    rmse_log = sqrt(mean(abs2, yhat .- data["y-test"]))
    rmse_level = sqrt(mean(abs2, exp.(yhat) .- exp.(data["y-test"])))

    df[!, :rmse_log] = [rmse_log]
    df[!, :rmse_level] = [rmse_level]

    # distribution match
    target_q = collect(0.01:0.01:1.0)
    qperformance = vec(get_observed_quantiles(data["y-test"], ypp, target_q))
    qperformance = DataFrame([qperformance, target_q], [:observed, :target])

    CSV.write("./evaluations-spillover/qperformance-net$netid-rep$rep.csv", qperformance)

    # VaR performance
    spx_train = data["spx-train"]
    spx_test = data["spx-test"]
    mu_spx = mean(spx_train)

    for (alpha, name) in zip([0.001, 0.01, 0.05, 0.1], ["VaR_0_1", "VaR_1_0", "VaR_5_0", "VaR_10_0"])
        df[!, Symbol(name)] = [var_performance(alpha, ypp, spx_test, mu_spx)]
    end

    CSV.write("evaluations-spillover/stats-net$netid-rep$rep.csv", df)
    logme("Done with net $netid rep $rep")
end

Distributed.rmprocs(workers())