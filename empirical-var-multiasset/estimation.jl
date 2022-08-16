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

# Scaling makes estimation simpler
scaling_factor = 100f0
y = scaling_factor*y
x = scaling_factor*x

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

n_in = size(x_train, 2)
n_out = size(y_train, 1)

network_structures = [
    Flux.Chain(RNN(n_in, n_in), Dense(n_in, n_out)),
    Flux.Chain(RNN(n_in, n_out), Dense(n_out, n_out)),
    Flux.Chain(LSTM(n_in, n_in), Dense(n_in, n_out)),
    Flux.Chain(LSTM(n_in, n_out), Dense(n_out, n_out)),
]

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

logme("Done preparing network definitions.")

################################################################################
#### MCMC Estimation and Evaluation

logme("GGMC ==> Starting")

target_accepts = [0.5f0]
betas = [0.5f0]
steps = [3]
m0s = collect(20f0:20f0:200f0)

ggmc_configs = vec(collect(Iterators.product(network_structures, target_accepts, betas, steps, m0s)))

# @sync @distributed for config_index=1:lastindex(ggmc_configs)
#     logme("GGMC ==> Starting config $config_index")
#     try
#         config = ggmc_configs[config_index]
#         net, target_accept, beta, step, m0 = config
#         netid = findfirst(x -> x == net, network_structures)

#         bnn = get_bnn(net, x_train, y_train, m0)

#         Random.seed!(6150533)
#         opt = FluxModeFinder(bnn, Flux.ADAM(), ϵ = -Inf, windowlength = 5000)
#         θmap = find_mode(bnn, 100, 1000, opt; showprogress = false)

#         ndraws = 50_000
#         sadapter = DualAveragingStepSize(1f-15; adapt_steps = Int(ndraws*0.1), target_accept = target_accept)
#         sampler = GGMC(Float32, β = beta, l = 1f-15, sadapter = sadapter, madapter = FixedMassAdapter(), steps = step)
#         logme("GGMC ==> MCMC config $config_index")
#         chain = mcmc(bnn, 100, ndraws, sampler; θstart = copy(θmap))

#         keep = 20_000
#         chain = chain[:, end-keep+1:end]
#         save("mcmc-chains/chain-$config_index.jld", "netid", netid, "chain", chain)

#         logme("GGMC ==> Posterior Predictive config $config_index")
#         ypp = sample_posterior_predict(bnn, chain; x = x_test)
#         pfolio_return_hat = vec(mean(ypp; dims = 2))
#         pfolio_return = vec(mean(y_test; dims = 1))

#         rmse = sqrt(mean(abs2, (pfolio_return .- pfolio_return_hat)/scaling_factor))

#         VaR_levels = SharedArray([0.001, 0.01, 0.05, 0.1])
#         get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
#         VaRs = Dict(alpha => get_VaR(ypp, alpha) for alpha in VaR_levels)

#         df = DataFrame(
#             "config" => config_index,
#             "netid" => netid,
#             "rmse" => rmse,
#             "m0" => m0,
#         )
#         for (key, v) in zip(keys(VaRs), values(VaRs))
#             df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(pfolio_return .< v)
#         end

#         CSV.write("evaluations/mcmc-single-chain-config$config_index.csv", df)
#         logme("GGMC ==> Done config $config_index")
#     catch e
#         logme("GGMC ==> Error in config $config_index ==> $e")
#     end
# end

# logme("GGMC ==> Merging all evaluation files")
# files = readdir("./evaluations/")
# files = filter(x -> occursin("mcmc-single-chain-config", x), files)
# files = ["./evaluations/$f" for f in files]
# evaluation = DataFrame(CSV.File(files))
# CSV.write("./evaluations/mcmc-single-chain-all.csv", evaluation)

################################################################################
#### BBB Estimation and Evaluation

logme("Starting BBB")

bbb_configs = vec(collect(Iterators.product(network_structures, m0s)))
@sync @distributed for config=1:lastindex(bbb_configs)
    logme("BBB ==> Starting config $config")
    try
        net, m0 = bbb_configs[config]
        netid = findfirst(x->x==net, network_structures)
        bnn = get_bnn(net, x_train, y_train, m0)

        logme("BBB ==> Estimating config $config")
        Random.seed!(6150533)
        vi = bbb(bnn, 100, 1000; opt = Flux.ADAM(1f-35), n_samples_convergence = 1)
        save("./bbb-objects/vi-$config.jld", "bbb", vi, "netid", netid, "config", config, "m0", m0)

        logme("BBB ==> Evaluating config $config")
        chain = rand(vi[1], 20_000)
        ypp = sample_posterior_predict(bnn, chain; x = x_test)
        pfolio_return_hat = vec(mean(ypp; dims = 2))
        pfolio_return = vec(mean(y_test; dims = 1))

        rmse = sqrt(mean(abs2, (pfolio_return .- pfolio_return_hat)/scaling_factor))

        VaR_levels = SharedArray([0.001, 0.01, 0.05, 0.1])
        get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
        VaRs = Dict(alpha => get_VaR(ypp, alpha) for alpha in VaR_levels)

        df = DataFrame(
            "netid" => netid,
            "rmse" => rmse,
            "m0" => m0,
        )
        for (key, v) in zip(keys(VaRs), values(VaRs))
            df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(pfolio_return .< v)
        end

        CSV.write("evaluations/bbb-single-chain-config$config.csv", df)
        logme("BBB ==> Done config $config")
    catch e
        logme("BBB ==> Error in config $config ==> $e")
    end
end

# logme("BBB ==> Merging all evaluation files")
# files = readdir("./evaluations/")
# files = filter(x -> occursin("bbb-single-chain-net", x), files)
# files = ["./evaluations/$f" for f in files]
# evaluation = DataFrame(CSV.File(files))
# CSV.write("./evaluations/bbb-single-chain-all.csv", evaluation)
