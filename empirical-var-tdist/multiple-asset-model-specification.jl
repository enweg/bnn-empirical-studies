using Distributed
@everywhere using Pkg
addprocs(50) 
# addprocs(5) 
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

    function get_bnn(net, x_train, y_train, nu)
        nc = destruct(net)
        prior = GaussianPrior(nc, 0.5f0)
        like = SeqToOneTDist(nc, Gamma(2.0, 0.5), nu)
        init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
        bnn = BNN(x_train, y_train, like, prior, init)
        return bnn
    end
    logme(msg) = @info("@$(Dates.now()) ==> $msg")
end
logme("All workers have been set up.")


# Loading the wide data and keeping only SPX data which is the asset we will be
# using for our single asset test. Explanatory variables are the lagged return,
# lagged RV and lagged squared return. 
df = DataFrame(CSV.File("./data/wide-data.csv"))
return_columns = names(df)[occursin.("return_", names(df))]
return_mat = Matrix(df[!, return_columns])
equal_weighted_pfolio = mean(return_mat; dims = 2)
df[!, "return_portfolio"] .= equal_weighted_pfolio
df = @chain df begin
    @rtransform(:Date = Date(:Date[1:10]))
    @orderby(:Date)
end
pfolio_column_idx = findfirst(d -> d == "return_portfolio", names(df))

data = Matrix(df[:, 2:end])
dates = df[!, :Date]
test_from_index = findfirst(d -> d >= Date("2019-01-01"), dates)

# Creating the subsequences. Keep in mind that returns are the second column in
# this tensor
tensor = BFlux.make_rnn_tensor(data, 20 + 1)
tensor = Float32.(tensor)
# -1 because we removed the first date column
y = tensor[end, pfolio_column_idx - 1, :]
x = tensor[1:end-1, :, :]
# checking if we took the right column
# approximate because we changed from 64bit to 32bit 
all(last(y, 10) .≈ last(df[!, "return_portfolio"], 10)) ? logme("All good with y") : error("y is not what you wanted")

y_train = y[1:test_from_index-1]
y_test = y[test_from_index:end]
x_train = x[:, :, 1:test_from_index-1]
x_test = x[:, :, test_from_index:end]

save(
    "data/multiple-asset.jld", 
    "y-train", y_train, 
    "y-test", y_test, 
    "x-train", x_train, 
    "x-test", x_test
)

# Network structures
include("./multiple_network_structures.jl")

# target_accepts = [0.1f0, 0.5f0]
# betas = [0.01f0, 0.5f0, 0.99f0]
# steps = [1, 2, 3]
target_accepts = [0.5f0]
betas = [0.5f0]
steps = [3]
nus = collect(3f0:1f0:10f0)
nus = vcat(nus, [20f0, 30f0])

ggmc_configs = vec(collect(Iterators.product(network_structures, target_accepts, betas, steps, nus)))

logme("Preparing $(length(ggmc_configs)) GGMC configurations.")
logme("Starting GGMC")

@sync @distributed for config_index in 1:lastindex(ggmc_configs)
    logme("GGMC: Starting with config $(config_index)")
    try
        Random.seed!(6150533)
        net, target_accept, beta, step, nu = ggmc_configs[config_index]
        netid = findfirst(x -> net == x, network_structures)
        bnn = get_bnn(net, x_train, y_train, nu)
        opt = FluxModeFinder(bnn, Flux.RMSProp())
        θmap = find_mode(bnn, 100, 1_000, opt; showprogress = false)
        # θmap = find_mode(bnn, 100, 10, opt; showprogress = false)
        logme("GGMC: Found MAP for config $(config_index)")
        sadapter = DualAveragingStepSize(1f-15; target_accept = target_accept, adapt_steps = 15000)
        madapter = FixedMassAdapter()
        sampler = GGMC(Float32; β = beta, steps = step, sadapter = sadapter, madapter = madapter)
        @suppress begin 
            ch = mcmc(bnn, 100, 100_000, sampler; θstart = θmap, showprogress = false)
            # ch = mcmc(bnn, 100, 100, sampler; θstart = θmap, showprogress = false)
            ch = ch[:, end-20_000+1:end]
            save("mcmc-chains-multiple/chain-$(config_index).jld", "chain", ch, "netid", netid)
        end
    catch e
        logme("GGMC: Error in config $(config_index): $(e)")
    end
    logme("GGMC: Done with config $(config_index)")
end
logme("Done with GGMC")

logme("Starting BBB")
# Bayes-By-Backprop Estimation
bbb_configs = vec(collect(Iterators.product(network_structures, nus)))
@sync @distributed for config_id in 1:lastindex(bbb_configs)
    try
        net, nu = bbb_configs[config_id]
        netid = findfirst(x -> net == x, network_structures)
        logme("BBB: Starting with config $(config_id)")
        bnn = get_bnn(net, x_train, y_train, nu)
        vi = bbb(bnn, 100, 5000; opt = Flux.RMSProp(), n_samples_convergence = 1, showprogress = false)
        # vi = bbb(bnn, 100, 5; opt = Flux.RMSProp(), n_samples_convergence = 1, showprogress = true)
        save("./bbb-objects-multiple/vi-net$config_id.jld", "bbb", vi, "netid", netid)
        logme("BBB: Done with network $(config_id)")
    catch e
        logme("BBB: Error in network $(config_id): $e")
    end
end