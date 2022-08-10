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

    logme(msg) = @info("$(Dates.now()) ==> $msg")
end

################################################################################
#### Data Preparation

logme("Starting data preparation.")

df = DataFrame(CSV.File("../empirical-var/data/wide-data.csv"))
df = @chain df begin
    @rtransform(:Date = Date(:Date[1:10]))
    @orderby(:Date)
    @select(:Date, :rv5_SPX, :return_SPX, :returnsq_SPX)
end
df[!, "rv5_SPX_weekly"] = vcat(fill(missing, 4), [mean(df[!, "rv5_SPX"][t-5+1:t]) for t=5:size(df, 1)]) 
df[!, "rv5_SPX_monthly"] = vcat(fill(missing, 21), [mean(df[!, "rv5_SPX"][t-22+1:t]) for t=22:size(df, 1)])
# lost 22 observations because of monthly aggregation
df = df[22:end, :]

# we are working with log-rv
rv_columns = filter(x -> contains(x, "rv5"), names(df))
for c in rv_columns
    df[!, c] = log.(df[!, c])
end

# subtract 1 because we remove first column in a second
idx_daily = findfirst(x -> x == "rv5_SPX", names(df)) - 1
idx_weekly = findfirst(x -> x == "rv5_SPX_weekly", names(df)) - 1
idx_monthly = findfirst(x -> x == "rv5_SPX_monthly", names(df)) - 1
data = Matrix(df[:, 2:end])
dates = df[!, :Date]

# Creating the subsequences. Keep in mind that returns are the second column in
# this tensor
seq_len = 60+1
tensor = BFlux.make_rnn_tensor(data, seq_len)
tensor = Float32.(tensor)
y = tensor[end, idx_daily, :]
x = tensor[1:end-1, :, :]

test_from_index = findfirst(d -> d >= Date("2019-01-01"), dates) - seq_len
y_train = y[1:test_from_index-1]
y_test = y[test_from_index:end]
x_train = x[:, :, 1:test_from_index-1]
x_test = x[:, :, test_from_index:end]

save(
    "data/informationset-simple.jld", 
    "y-train", y_train, 
    "y-test", y_test, 
    "x-train", x_train, 
    "x-test", x_test, 
    "idx_daily", idx_daily, 
    "idx_weekly", idx_weekly, 
    "idx_monthly", idx_monthly
)

logme("Finished data preparation")

################################################################################
#### RNN-HAR definition

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

################################################################################
#### GGMC Estimation

@sync @distributed for netid in 1:lastindex(network_structures)
    logme("GGMC: Starting with network $netid")
    try
        Random.seed!(6150533)
        net = network_structures[netid]
        bnn = get_bnn(net, x_train, y_train, idx_daily, idx_weekly, idx_monthly)

        opt = FluxModeFinder(bnn, Flux.RMSProp())
        θmap = find_mode(bnn, 100, 1000, opt; showprogress = false)

        sadapter = DualAveragingStepSize(1f-15; target_accept = 0.5f0, adapt_steps = 10000)
        sampler = GGMC(Float32; β = 0.5f0, l = 1f-15, sadapter = sadapter, madapter = FixedMassAdapter(), steps = 3)
        @suppress begin
            chain = mcmc(bnn, 100, 100000, sampler; θstart = copy(θmap), showprogress = false)
            chain = chain[:, end-20_000+1:end]
            save("chains-simple/chain-$netid.jld", "netid", netid, "chain", chain)
        end
        logme("GGMC: Done with network $netid")
    catch e
        logme("GGMC: Error in network $netid ==> $e")
    end
end

################################################################################
#### BBB

logme("Starting with BBB")

@sync @distributed for netid=1:lastindex(network_structures)
    logme("BBB: Starting with network $netid")
    try
        net = network_structures[1]
        bnn = get_bnn(net, x_train, y_train, idx_daily, idx_weekly, idx_monthly)
        Random.seed!(6150533)
        vi = bbb(bnn, 100, 100; mc_samples = 1, showprogress = false, 
            opt = Flux.RMSProp(), n_samples_convergence = 1)
        save("bbb-objects-simple/vi-$netid.jld", "netid", netid, "bbb", vi)
        logme("BBB: Done with network $netid")
    catch e
        logme("BBB: Error in network $netid ==> $e")
    end
end

logme("All Done !!!!")