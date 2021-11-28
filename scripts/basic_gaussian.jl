using Distributions
using Flux
using LaTeXStrings

N = 100; # number of samples
dataDist = Normal(5, 2); #distribution of the data
x = rand(dataDist, N) #training data
noiseDist = Normal(0, 1); #distribution of the noise data
y = rand(noiseDist,
 N) #noise data

"""
Computes log-likelihood for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function Gaussianlogpdf(u::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, c::Vector{Float64})
    if σ >= [0.0]
        l = .- 0.5 .* ((u .- μ) ./ σ) .^ 2 .+ c #log likelihood of the Gaussian      
        return l
    else
        println("σ should be greater than 0")
    end
end

"""
Computes function G for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function G(u::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, c::Vector{Float64})
    if σ >= [0.0]
        G = Gaussianlogpdf(u, μ, σ, c) .- logpdf.(noiseDist, u)
        return G
    else
        println("σ should be greater than 0")
    end
end

"""
Computes function h for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function h(u::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, c::Vector{Float64}) 
    h = 1 ./ (1 .+ exp.(-G(u, μ, σ, c)))
    return h
end

"""
Computes loss function for given vector of data x and noise data y.
"""
function loss(x::Vector{Float64}, y::Vector{Float64}) 
    loss = -sum(log.(h(x, μ, σ, c)) .+ log.(1 .- h(y, μ, σ, c)))
end

# set initial value for training
μ = [0.0]
σ = [1.0]
c = [1.0]


opt = Flux.ADAM(); #optimalization method
Flux.train!((x,y) -> loss(x,y), [μ, σ, c], Iterators.repeated((x, y), 10000), opt) # train
println("μ=$μ, σ=$σ, c=$c") #results