using Turing
using Random, Distributions
using JLD
using LinearAlgebra
using MCMCChains
using DataFrames
using CSV

data = load("./data/multiple-asset.jld")
x_train = data["x-train"]
x_train = x_train[end, :, :]
y_train = data["y-train"]

last(y_train, 10)
x_train[:, end-10+1:end]

@model function garch(y, sigma0)
    mu ~ Normal(0, 10)
    alpha0 ~ TruncatedNormal(0, 10, 0, Inf)
    alpha1 ~ Uniform(0, 1)
    beta1 ~ Uniform(0, 1)

    sigma_t = sigma0
    for t=2:lastindex(y)
        sigma_t = sqrt(alpha0 + alpha1*(y[t-1]-mu)^2 + beta1*sigma_t^2)
        y[t] ~ Normal(mu, sigma_t)
    end
end

sigma0 = sqrt(var(y_train))
model = garch(y_train, sigma0)

chain = sample(model, NUTS(0.65), 20_000)

function predict_garch(y_tm1, mu, sigma0, alpha0, alpha1, beta1)
    y = similar(y_tm1)
    T = length(y_tm1)
    sigma = similar(y)
    sigma_t = sigma0
    for t=1:T
        sigma_t = sqrt(alpha0 + alpha1*(y_tm1[t] - mu)^2 + beta1*sigma_t^2)
        sigma[t] = sigma_t
        y[t] = rand(Normal(mu, sigma_t))
    end
    return y, sigma
end

function garch_posterior_predict(y, chain, sigma0)
    mu = vec(chain[:mu])
    alpha0 = vec(chain[:alpha0])
    alpha1 = vec(chain[:alpha1])
    beta1 = vec(chain[:beta1])

    y_tm1 = y[1:end-1]
    out = predict_garch.([y_tm1], mu, [sigma0], alpha0, alpha1, beta1)
    ys = [o[1] for o in out]
    sigmas = [o[2] for o in out]
    return reduce(hcat, ys), reduce(hcat, sigmas)
end

ys, sigmas = garch_posterior_predict(data["y-test"], chain, sigma0)

yhat = mean(ys; dims = 2)
sigmahat = mean(sigmas; dims = 2)
# plot(sigmahat)
# plot(yhat)

mse = mean(abs2, yhat .- data["y-test"][2:end])
rmse = sqrt(mse)

get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
VaR_levels = [0.001, 0.01, 0.05, 0.1]
VaRs = Dict(alpha => get_VaR(ys, alpha) for alpha in VaR_levels)

df = DataFrame(
    "rmse" => rmse
)
for (key, v) in zip(keys(VaRs), values(VaRs))
    df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(data["y-test"][2:end] .< v)
end
df

CSV.write("./evaluations/garch11-baseline-multiple.csv", df)