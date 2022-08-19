using Random, Distributions
using Turing
using JLD
using LinearAlgebra
using MCMCChains
using StatsPlots
using CSV, DataFrames
include("./utils.jl")

data = load("./data/informationset-simple.jld")
y_train = data["y-train"]
y_test = data["y-test"]
x_train = data["x-train"]
x_train = x_train[:, getindex.([data], ["idx_daily", "idx_weekly", "idx_monthly"]), :]
x_train = x_train[end, :, :]
x_test = data["x-test"]
x_test = x_test[:, getindex.([data], ["idx_daily", "idx_weekly", "idx_monthly"]), :]
x_test = x_test[end, :, :]

function har_yhat(x, coeffs, intercept)
    return x'*coeffs .+ intercept
end

@model function HAR(x, y)
    intercept ~ Normal(0, 0.5)
    coeffs ~ MvNormal(fill(0, 3), 0.5^2*I)
    σ ~ Gamma(2.0, 0.5)

    yhat = har_yhat(x, coeffs, intercept)
    # yhat = x'*coeffs .+ intercept 
    return y ~ MvNormal(yhat, σ^2*I)
end

har = HAR(x_train, y_train)
chain = sample(har, NUTS(0.65), 30_000)
# 10_000 burnin
chain = chain[end-20_000+1:end, :, :]

function posterior_predict(chain, x)
    p = get_params(chain)
    yhats = p.intercept' .+ x'*reduce(hcat, p.coeffs)'
    sigmas = p.σ
    ypp = similar(yhats)
    for i=1:lastindex(sigmas)
        mu = yhats[:, i]
        sigma = sigmas[i]
        ypp[:, i] = rand(MvNormal(mu, sigma^2*I))
    end
    return ypp
end

ypp = posterior_predict(chain, x_test)

yhat = vec(mean(ypp; dims = 2))
rmse_log = sqrt(mean(abs2, yhat .- y_test))
rmse_level = sqrt(mean(abs2, exp.(yhat) .- exp.(y_test)))

df = DataFrame(
    "method" => "HAR", 
    "rmse_log" => rmse_log, 
    "rmse_level" => rmse_level
)

target_q = collect(0.01:0.01:1.0)
qperformance = vec(get_observed_quantiles(y_test, ypp, target_q))
qperformance = DataFrame([qperformance, target_q], [:observed, :target])
CSV.write("./evaluations/baseline-qperformance.csv", qperformance)

function var_performance(alpha, ypp, y, mu)
    VaR = reduce(hcat, [[quantile(Normal(mu, sqrt(exp(rv))), alpha) for rv in ts] for ts in eachcol(ypp)])
    VaR_mean = vec(mean(VaR; dims = 2))
    return mean(y .< VaR_mean)
end

spx_train = data["spx-train"]
spx_test = data["spx-test"]
mu_spx = mean(spx_train)

for (alpha, name) in zip([0.001, 0.01, 0.05, 0.1], ["VaR_0_1", "VaR_1_0", "VaR_5_0", "VaR_10_0"])
    df[!, Symbol(name)] = [var_performance(alpha, ypp, spx_test, mu_spx)]
end

CSV.write("evaluations/baseline-stats.csv", df)