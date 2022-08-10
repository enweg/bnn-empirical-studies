using BFlux
import BFlux: BNNLikelihood, NetConstructor, predict
using Random, Distributions
using Bijectors
using LinearAlgebra


struct HARLikelihood{T, F, D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_σ::D
    idx_daily::Int
    idx_weekly::Int
    idx_monthly::Int
end
function HARLikelihood(nc::NetConstructor{T, F}, prior_σ::D, idx_daily::Int, idx_weekly::Int, idx_monthly::Int) where {T,F,D<:Distribution}
    return HARLikelihood(1, nc, prior_σ, idx_daily, idx_weekly, idx_monthly)
end

function calculate_yhat_and_coeffs(l, net, x)
    # x will be of the form 
    # timestep x features x sequences
    # since the network cannot get the current RV values, we go until end-1
    # the har then becomes the current values
    x_net = x[1:end-1, :, :]
    x_har = x[end, [l.idx_daily, l.idx_weekly, l.idx_monthly], :]  # 3xsequences 

    # time steps are rows
    har_params = [net(xx) for xx in eachslice(x_net; dims = 1)][end]
    intercepts = har_params[1, :]
    c_rv_daily = har_params[2, :]
    c_rv_weekly = har_params[3, :]
    c_rv_monthly = har_params[4, :]

    yhat = intercepts .+ c_rv_daily.*x_har[1,:] .+ c_rv_weekly.*x_har[2,:] .+ c_rv_monthly.*x_har[3,:]

    return yhat, (intercept =intercepts, daily = c_rv_daily, weekly = c_rv_weekly, monthly = c_rv_monthly)

end

function (l::HARLikelihood{T, F, D})(x::Array{T, 3}, y::Vector{T}, 
    θnet::AbstractVector{T}, θlike::AbstractVector{T}) where {T, F, D}

    θnet = T.(θnet)
    θlike = T.(θlike)

    # reconstructing net given parameters
    net = l.nc(θnet)

    yhat, _ = calculate_yhat_and_coeffs(l, net, x)

    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    
    # we do lose some data due to taking averages
    return logpdf(MvNormal(yhat, sigma^2*I), y) + logpdf(tdist, θlike[1])
end

function predict(l::HARLikelihood{T, F, D}, x::Array{T, 3}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    net = l.nc(θnet)

    yhat, _ = calculate_yhat_and_coeffs(l, net, x)

    sigma = invlink(l.prior_σ, θlike[1])

    ypp = rand(MvNormal(yhat, sigma^2*I))

    return ypp
end