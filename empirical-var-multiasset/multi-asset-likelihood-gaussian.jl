using BFlux
import BFlux: BNNLikelihood, NetConstructor, predict
using Random, Distributions
using Bijectors
using LinearAlgebra


# Motivated by https://groups.google.com/g/julia-users/c/UARlZBCNlng?pli=1
function vec2ltri(v::AbstractVector{T}, z::T=zero(T)) where {T}
    n = length(v)
    dims = -0.5*(1-sqrt(1+8*n))
    dims % 1 != 0 && error("vec2ltri: length of vector is not triangular")
    dims = Int(dims)
    return [r >= c ? v[Int(r*(r-1)/2+c)] : z for r=1:dims, c=1:dims]
end

struct MultiAssetGaussian{T, F, D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_cov::D
end
function MultiAssetGaussian(nc::NetConstructor{T, F}, prior_cov::D) where {T,F,D<:Distribution}
    n = size(prior_cov.Ψ, 1)
    num_params_like = Int(n*(n+1)/2)
    return MultiAssetGaussian(num_params_like, nc, prior_cov)
end

function get_yhat_covariance(l, θnet::AbstractVector{T}, θlike::AbstractVector{T}, x::Array{T, 3}) where {T}
    # reconstructing net given parameters
    net = l.nc(θnet)

    # is a matrix of dimension num_assets x time_steps
    yhat = [net(xx) for xx in eachslice(x; dims = 1)][end]

    tcovmat = vec2ltri(θlike)
    covmat = invlink(l.prior_cov, tcovmat)

    return yhat, covmat
end


function (l::MultiAssetGaussian{T, F, D})(x::Array{T, 3}, y::Matrix{T}, 
    θnet::AbstractVector{T}, θlike::AbstractVector{T}) where {T, F, D}

    θnet = T.(θnet)
    θlike = T.(θlike)

    yhat, cov_mat = get_yhat_covariance(l, θnet, θlike, x)

    tcovmat = vec2ltri(θlike)
    tcovdist = transformed(l.prior_cov)

    return sum(logpdf(MvNormal(yhat[:, i], cov_mat), y[:, i]) for i=1:size(y, 2)) + logpdf(tcovdist, tcovmat)
end

function predict(l::MultiAssetGaussian{T, F, D}, x::Array{T, 3}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D}
    θnet = T.(θnet)
    θlike = T.(θlike)


    yhat, cov_mat = get_yhat_covariance(l, θnet, θlike, x)

    ypp = [rand(MvNormal(yhat[:, i], cov_mat)) for i=1:size(yhat, 2)]
    ypp = reduce(hcat, ypp)

    # only returning equal weighted pfolio
    return vec(mean(ypp; dims = 1))
end