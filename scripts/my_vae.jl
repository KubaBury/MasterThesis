using Plots
using Statistics
using Distributions
using LaTeXStrings
using Flux
using Random
using LinearAlgebra
## Generate data
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
Random.seed!(12394)
i = 100 ;#celkovy pocet y_i;
λ = 10 ;#rate in Poisson distribution;
P = Poisson(λ);
U = Uniform(0,10);
pocet_x = rand(P,i); #pocty x_k prirazene k jednomu y_i
xx = [rand(U, pocet_x[j]) for j =1:i] ##  pole poli, generuje bag
x = mean.(xx); ## prumery jednotlivych instanci
y = 1 .- 30.0x .+ 5.0x.^2 .+ 5.0*randn(length(x)); ## kvadraticka zavislost + noise
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

x./=maximum(x)
y./=maximum(y)

p=100
B=rand(MvNormal([2,2], [3 2; 2 3]), p)
x1=B[1,:]
x2=B[2,:]
x=[x1 x2]'


## NN == inicializace??
nx = 2
nz = 2
nh = 32
#A, μ, logσ = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
A, μ, logσ = Dense(nx, nh), Dense(nh, nz,selu), Dense(nh, nz,selu)

function encoder(x)
    h = A(x)
    return h
end

#f = Chain(Dense(nz,nh,swish),Dense(nh,nx)) 
f = Chain(Dense(nz,nh,selu),Dense(nh,nx)) 
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)


s = 0.01;#param(rand(2,1))
#loss(x,z) = Flux.mse(f(mu(x)+exp.(0.5*lsig(x)).*z),x) + KL(mu(x),lsig(x))
function loss(x)
    #s+exp.(s).*0.5*sum((x.-f(z_given_x)).^2) + KL(mu(x),lsig(x))
    HID = A(x);
    zsample = z(μ(HID),logσ(HID));
    0.5*sum((x.-f(zsample)).^2) .+ s*KL(μ(HID),logσ(HID))
end

ps = Flux.params(A, μ, logσ, f)
da = Iterators.repeated(([x y]',),10000)
opt = Flux.ADAM()
Flux.train!(loss,ps,da,opt);
#println(loss([x y]'))

Zs=randn(nz,p)
Xg=f(Zs) 
Zg=g([x y]')
scatter(x,y, label = "true", markersize = 8, legend =:topleft, xlabel = "x", ylabel = "y", markershape=:cross, c=:red)
scatter!(Xg[1,:], Xg[2,:], label = "estimated", markersize = 8, markershape=:cross,c=:green)


plt(x1,x2) = pdf(MvNormal([2,2], [3 2; 2 3]), [x1,x2])
plt2(x1,x2) = pdf.(MvNormal(mean(f(z(μ(encoder(x[:,1])),exp.(logσ(encoder(x[:,1]))))), dims=2)[:],[1 0; 0 1]), [x1,x2])

gr()
#c1=contour(-2.:0.01:6.2, -2.:0.01:6.2, (x1,x2)->plt(x1,x2), linewidth=1, legendfontsize=30,yguidefontsize=20, xguidefontsize=20,
xtickfontsize=20,ytickfontsize=20,levels=6,alpha=0.8,c =:blues,colorbar=false,lw=5)
scatter(x[1,:], x[2,:],label = "true", markersize = 8, markershape=:cross,c=:green)
scatter!(Xg[1,:], Xg[2,:],label = "estimated", markersize = 8, markershape=:cross,c=:red)

#contour!(c1,-2:0.1:6.2, -2.:0.1:6.2, (x1,x2) ->plt2(x1,x2), linewidth=5, levels=6,linestyle=:dash,alpha=1,lw=5, c=:reds)



#scatter!(Zg[2][1,:],Zg[1][1,:])
#k = histogram(Zg[2]', nbins = 50, normalize=:pdf, color =:lightgreen, label = false)
#c = -3:0.01:3
#ng = 1/(2*pi)^(1/2)*exp.(-c.^2/2)
#l = plot!(k ,c,ng, linewidth = 5, color =:brown, xlabel = "x", ylabel = "f(x)", label = L"N(0,1)")
 