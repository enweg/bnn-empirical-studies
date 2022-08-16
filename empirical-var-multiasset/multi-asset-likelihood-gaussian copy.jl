using BFlux
import BFlux: BNNLikelihood, NetConstructor, predict
using Random, Distributions
using Bijectors
using LinearAlgebra


# Motivated by https://groups.google.com/g/julia-users/c/UARlZBCNlng?pli=1
function vec2utri(v::AbstractVector{T}, z::T=zero(T)) where {T}
    n = length(v)
    dims = 0.5*(sqrt(8*n+1) + 1)
    dims % 1 != 0 && error("vec2utri: length of vector is not triangular")
    dims = Int(dims)
    return [r >= c ? z : v[Int((c-2)*(c-1)/2+r)] for r=1:dims, c=1:dims]
end


struct MultiAssetGaussian{T, F, D1<:Distribution, D2<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_variance::D1
    num_assets::Int
    lkj_eta::T
    lkj_dist::D2
end
function MultiAssetGaussian(nc::NetConstructor{T, F}, prior_variance::D, num_assets::Int, lkj_eta::Number=T(1)) where {T,F,D<:Distribution}
    lkj_dist = LKJ(num_assets, lkj_eta)
    params_correlation = (num_assets-1)*num_assets/2
    params_variance = num_assets
    num_params_like = Int(params_correlation + params_variance)
    return MultiAssetGaussian(num_params_like, nc, prior_variance, num_assets, lkj_eta, lkj_dist)
end

function get_yhat_covariance(l, θnet::AbstractVector{T}, θlike::AbstractVector{T}, x::Array{T, 3}) where {T}
    num_params_correlation = l.num_params_like - l.num_assets
    params_correlation = θlike[1:num_params_correlation]
    params_variance = θlike[num_params_correlation+1:end]

    # reconstructing net given parameters
    net = l.nc(θnet)

    # is a matrix of dimension num_assets x time_steps
    yhat = [net(xx) for xx in eachslice(x; dims = 1)][end]

    sigmas = invlink.(l.prior_variance, params_variance)

    tcorrelation_mat = vec2utri(params_correlation)
    correlation_mat = invlink(l.lkj_dist, tcorrelation_mat)

    # We need this conversion because otherwise numerical problems occur
    cov_mat = T.(Float64.(diagm(sigmas)).*Float64.(correlation_mat).*Float64.(diagm(sigmas))) + 1f-6*I

    return yhat, cov_mat
end


function (l::MultiAssetGaussian{T, F, D1, D2})(x::Array{T, 3}, y::Matrix{T}, 
    θnet::AbstractVector{T}, θlike::AbstractVector{T}) where {T, F, D1, D2}

    θnet = T.(θnet)
    θlike = T.(θlike)
    num_params_correlation = l.num_params_like - l.num_assets
    params_correlation = θlike[1:num_params_correlation]
    params_variance = θlike[num_params_correlation+1:end]

    yhat, cov_mat = get_yhat_covariance(l, θnet, θlike, x)

    tdist_variance = transformed(l.prior_variance)
    tdist_correlation = transformed(l.lkj_dist)
    tcorrelation_mat = vec2utri(params_correlation)

    # return sum(logpdf(MvNormal(yhat[:, i], cov_mat), y[:, i]) for i=1:size(y, 2)) + sum(logpdf.(tdist_variance, params_variance)) + logpdf(tdist_correlation, Float64.(tcorrelation_mat))
    return sum(logpdf(MvNormal(vec(mean(y_train; dims = 2)), cov_mat), y[:, i]) for i=1:size(y, 2)) + sum(logpdf.(tdist_variance, params_variance)) + logpdf(tdist_correlation, Float64.(tcorrelation_mat))
end

function predict(l::MultiAssetGaussian{T, F, D1, D2}, x::Array{T, 3}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D1, D2}
    θnet = T.(θnet)
    θlike = T.(θlike)


    yhat, cov_mat = get_yhat_covariance(l, θnet, θlike, x)

    ypp = [rand(MvNormal(yhat[:, i], cov_mat)) for i=1:size(yhat, 2)]
    ypp = reduce(hcat, ypp)

    # only returning equal weighted pfolio
    return vec(mean(ypp; dims = 1))
end