using Distributed
addprocs(3)
using Pkg
pmap(i -> Pkg.activate("."), workers())
pmap(i -> Pkg.instantiate(), workers())


@everywhere begin
    using Turing
    using Random, Distributions
    using JLD
    using LinearAlgebra
    using MCMCChains
    using DataFrames
    using CSV
end

@everywhere begin
    data = load("./data/multiple-asset.jld")
    x_train = data["x-train"]
    x_train = x_train[end, :, :]
    y_train = data["y-train"]
    x_train[:, end-10+1:end]
end

@everywhere begin
    function loglike_garch_tdist(nu, y, mu, sigma_t)
        return loglikelihood(TDist(nu), (y-mu)/sigma_t) - log(sigma_t)
    end

    @model function tgarch(nu, y, sigma0)
        mu ~ Normal(0, 10)
        alpha0 ~ TruncatedNormal(0, 10, 0, Inf)
        alpha1 ~ Uniform(0, 1)
        beta1 ~ Uniform(0, 1)

        sigma_t = sigma0
        for t=2:lastindex(y)
            sigma_t = sqrt(alpha0 + alpha1*(y[t-1]-mu)^2 + beta1*sigma_t^2)
            Turing.@addlogprob! loglike_garch_tdist(nu, y[t], mu, sigma_t)
        end
    end

    function predict_tgarch(nu, y_tm1, mu, sigma0, alpha0, alpha1, beta1)
        y = similar(y_tm1)
        T = length(y_tm1)
        sigma = similar(y)
        sigma_t = sigma0
        for t=1:T
            sigma_t = sqrt(alpha0 + alpha1*(y_tm1[t] - mu)^2 + beta1*sigma_t^2)
            sigma[t] = sigma_t
            y[t] = mu + sigma_t*rand(TDist(nu))
        end
        return y, sigma
    end

    function tgarch_posterior_predict(nu, y, chain, sigma0)
        mu = vec(chain[:mu])
        alpha0 = vec(chain[:alpha0])
        alpha1 = vec(chain[:alpha1])
        beta1 = vec(chain[:beta1])

        y_tm1 = y[1:end-1]
        out = predict_tgarch.([nu], [y_tm1], mu, [sigma0], alpha0, alpha1, beta1)
        ys = [o[1] for o in out]
        sigmas = [o[2] for o in out]
        return reduce(hcat, ys), reduce(hcat, sigmas)
    end
end

@everywhere function performance_df(nu)
    println("Working on nu=$nu")
    sigma0 = sqrt(var(y_train))
    model = tgarch(nu, y_train, sigma0)

    chain = sample(model, NUTS(0.65), 20_000)
    ys, sigmas = tgarch_posterior_predict(nu, data["y-test"], chain, sigma0)

    yhat = mean(ys; dims = 2)
    mse = mean(abs2, yhat .- data["y-test"][2:end])
    rmse = sqrt(mse)

    get_VaR(ypp, alpha) = [quantile(r, alpha) for r in eachrow(ypp)]
    VaR_levels = [0.001, 0.01, 0.05, 0.1]
    VaRs = Dict(alpha => get_VaR(ys, alpha) for alpha in VaR_levels)

    df = DataFrame(
        "nu" => nu,
        "rmse" => rmse
    )
    for (key, v) in zip(keys(VaRs), values(VaRs))
        df[!, Symbol("VaR_"*replace(string(key*100), "."=>"_"))] .= mean(data["y-test"][2:end] .< v)
    end
    println("Returning nu=$nu")
    return df
end
dfs = pmap(nu -> performance_df(nu) ,vcat(3:10, [20, 30]))

df = DataFrame()
for i=1:lastindex(dfs)
    append!(df, dfs[i])
end
df

CSV.write("./evaluations/tdist-garch11-baseline-multiple.csv", df)
