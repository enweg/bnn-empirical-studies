using Distributed
addprocs(60)
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

################################################################################
#### Data

logme("Preparing Data")

df = DataFrame(CSV.File("./data/wide-data.csv"))
return_columns = names(df)[occursin.("return_", names(df))]
df = @chain df begin
    @rtransform(:Date = Date(:Date[1:10]))
    @orderby(:Date)
end

data = Matrix(df[:, 2:end])
dates = df[!, :Date]
test_from_index = findfirst(d -> d >= Date("2019-01-01"), dates)
# -1 because date column was removed at beginning
return_idx = [findfirst(x -> x==rc, names(df))-1 for rc in return_columns]

# Creating the subsequences. Keep in mind that returns are the second column in
# this tensor
tensor = BFlux.make_rnn_tensor(data, 20 + 1)
tensor = Float32.(tensor)
y = tensor[end, return_idx, :]
x = tensor[1:end-1, :, :]

y_train = y[:, 1:test_from_index-1]
y_test = y[:, test_from_index:end]
x_train = x[:, :, 1:test_from_index-1]
x_test = x[:, :, test_from_index:end]

save(
    "data/multiple-asset.jld", 
    "y-train", y_train, 
    "y-test", y_test, 
    "x-train", x_train, 
    "x-test", x_test
)

logme("Done preparing data.")

################################################################################
#### Model Definition

logme("Preparing Network definitions.")

include("./multi-asset-likelihood-gaussian.jl")

n_in = size(x_train, 2)
n_out = size(y_train, 1)

network_structures = [
    Flux.Chain(RNN(n_in, n_in), Dense(n_in, n_out)),
    Flux.Chain(RNN(n_in, n_out), Dense(n_out, n_out)),
    Flux.Chain(RNN(n_in, 2*n_in), Dense(2*n_in, n_out)),
    Flux.Chain(LSTM(n_in, n_in), Dense(n_in, n_out)),
    Flux.Chain(LSTM(n_in, n_out), Dense(n_out, n_out)),
    Flux.Chain(LSTM(n_in, 2*n_in), Dense(2*n_in, n_out)),
]

@everywhere begin
    function get_bnn(net, x, y)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        like = MultiAssetGaussian(nc, Gamma(2.0, 0.5), size(y_train, 1), 1f0)
        init = InitialiseAllSame(Normal(0, 0.5), like, prior)

        bnn = BNN(x, y, like, prior, init)
    end
end

logme("Done preparing network definitions.")

################################################################################
#### MCMC Estimation and Evaluation

logme("GGMC ==> Starting")

target_accepts = [0.2f0, 0.5f0]
betas = [0.01f0, 0.5f0, 0.99f0]
steps = [1, 3]

ggmc_configs = vec(collect(Iterators.product(network_structures, target_accepts, betas, steps)))

@sync @distributed for config_index=1:lastindex(ggmc_configs)
    logme("GGMC ==> Starting config $config_index")
    try
        config = ggmc_configs[config_index]
        net, target_accept, beta, step = config
        netid = findfirst(x -> x == net, network_structures)

        bnn = get_bnn(net, x_train, y_train)

        Random.seed!(6150533)
        opt = FluxModeFinder(bnn, Flux.RMSProp())
        θmap = find_mode(bnn, 100, 1000, opt; showprogress = false)

        ndraws = 100_000
        sadapter = DualAveragingStepSize(1f-15; adapt_steps = Int(ndraws*0.1), target_accept = target_accept)
        sampler = GGMC(Float32, β = beta, l = 1f-15, sadapter = sadapter, madapter = FixedMassAdapter(), steps = step)
        logme("GGMC ==> MCMC config $config_index")
        chain = mcmc(bnn, 100, ndraws, sampler; θstart = copy(θmap))

        keep = 20_000
        chain = chain[:, end-keep+1:end]
        save("mcmc-chains/chain-$config_index.jld", "netid", netid, "chain", chain)

        logme("GGMC ==> Posterior Predictive config $config_index")
        ypp = sample_posterior_predict(bnn, chain; x = x_test)
        pfolio_return_hat = vec(mean(ypp; dims = 2))
        pfolio_return = vec(mean(y_test; dims = 1))

        rmse = sqrt(mean(abs, pfolio_return .- pfolio_return_hat))

        VaR_levels = SharedArray([0.001, 0.01, 0.05, 0.1])
        get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
        VaRs = Dict(alpha => get_VaR(ypp, alpha) for alpha in VaR_levels)

        df = DataFrame(
            "config" => config_index,
            "netid" => netid,
            "rmse" => rmse
        )
        for (key, v) in zip(keys(VaRs), values(VaRs))
            df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(pfolio_return .< v)
        end

        CSV.write("evaluations/mcmc-single-chain-config$config_index.csv", df)
        logme("GGMC ==> Done config $config_index")
    catch e
        logme("GGMC ==> Error in config $config_index ==> $e")
    end
end

logme("GGMC ==> Merging all evaluation files")
files = readdir("./evaluations/")
files = filter(x -> occursin("mcmc-single-chain-config", x), files)
files = ["./evaluations/$f" for f in files]
evaluation = DataFrame(CSV.File(files))
CSV.write("./evaluations/mcmc-single-chain-all.csv", evaluation)

################################################################################
#### BBB Estimation and Evaluation

logme("Starting BBB")

@sync @distributed for netid=1:lastindex(network_structures)
    logme("BBB ==> Starting network $netid")
    try
        net = network_structures[netid]
        bnn = get_bnn(net, x_train, y_train)

        logme("BBB ==> Estimating network $netid")
        vi = bbb(bnn, 100, 10; opt = Flux.RMSProp(), n_samples_convergence = 1)
        save("./bbb-objects/vi-$netid.jld", "bbb", vi, "netid", netid)

        logme("BBB ==> Evaluating network $netid")
        chain = rand(vi[1], 20_000)
        ypp = sample_posterior_predict(bnn, chain; x = x_test)
        pfolio_return_hat = vec(mean(ypp; dims = 2))
        pfolio_return = vec(mean(y_test; dims = 1))

        rmse = sqrt(mean(abs, pfolio_return .- pfolio_return_hat))

        VaR_levels = SharedArray([0.001, 0.01, 0.05, 0.1])
        get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
        VaRs = Dict(alpha => get_VaR(ypp, alpha) for alpha in VaR_levels)

        df = DataFrame(
            "netid" => netid,
            "rmse" => rmse
        )
        for (key, v) in zip(keys(VaRs), values(VaRs))
            df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(pfolio_return .< v)
        end

        CSV.write("evaluations/bbb-single-chain-net$netid.csv", df)
        logme("BBB ==> Done network $netid")
    catch e
        logme("BBB ==> Error in network $netid ==> $e")
    end
end

logme("BBB ==> Merging all evaluation files")
files = readdir("./evaluations/")
files = filter(x -> occursin("bbb-single-chain-net", x), files)
files = ["./evaluations/$f" for f in files]
evaluation = DataFrame(CSV.File(files))
CSV.write("./evaluations/bbb-single-chain-all.csv", evaluation)
