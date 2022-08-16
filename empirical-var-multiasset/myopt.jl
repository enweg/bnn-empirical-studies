# Use Flux to find modes
import BFlux:BNNModeFinder

"""
FluxModeFinder(bnn::BNN, opt::O; windowlength = 100, ϵ = 1e-6) where {O<:Flux.Optimise.AbstractOptimiser}

Use one of Flux optimisers to find the mode. Keep track of changes in θ over a window 
of `windowlegnth` and report convergence if the maximum change over the current window is 
smaller than ϵ.
"""
mutable struct MyOpt{B, O, T} <: BNNModeFinder
    bnn::B
    opt::O
    windowlength::Int
    window::Vector{T}
    prevθ::Vector{T}
    ϵ::T
    i::Int
    values::Vector{T}
end
function MyOpt(bnn::BNN, opt::O; windowlength = 100, ϵ = 1e-6) where {O<:Flux.Optimise.AbstractOptimiser}
    T = eltype(bnn.like.nc.θ)
    ϵ = T(ϵ)
    window = fill(T(-Inf), windowlength)
    prevθ = fill(T(Inf), bnn.num_total_params)
    i = 1

    v = BFlux.loglikeprior(bnn, vcat(bnn.init()...), bnn.x, bnn.y)
    return MyOpt(bnn, opt, windowlength, window, prevθ, ϵ, i, T[])
end

function BFlux.step!(fmf::MyOpt, θ::AbstractVector{T}, ∇θ::Function) where {T}

    # Checking for convergence
    if fmf.i == 1
        maxchange = maximum(abs, fmf.window)
        if  maxchange <= fmf.ϵ
            return θ, true
        end
    end

    v, g = ∇θ(θ)
    g = maximum(g) > 1.0 ? g ./ maximum(g) : g

    if length(fmf.values)<1
        append!(fmf.values, v)
    end
    alpha = 1.0f0
    vnext = v
    for i=1:100
        θnext = θ .+ alpha*g 
        vnext, _ = ∇θ(θnext)
        if vnext > fmf.values[end]
            break
        end
        alpha /= 10.0f0
    end
    println(alpha)
    θ .+= alpha*g 
    v, _ = ∇θ(θ)
    append!(fmf.values, v)

    maxchange = maximum(abs, fmf.prevθ - θ)
    fmf.window[fmf.i] = maxchange
    fmf.prevθ = copy(θ)
    fmf.i = fmf.i == fmf.windowlength ? 1 : fmf.i + 1

    # We need optimisation while flux does by default minimisation. 
    # So we negate the gradient.
    # Flux.update!(fmf.opt, θ, -g)

    return θ, false
end

