using Plots;
## Generate data
ndat = 1000
k=[x./100 for x in 1:1000]'
x = 4 .+[randn(Int(ndat/2));1 .+ randn(Int(ndat/2))]'

## normalize data!!!
p_hx=Plots.histogram(x',nbins=40)#,xlabel='x',ylabel='count')

using Flux

## NN == inicializace??
nx = 1
nz = 1
nh = 1;
#A, μ, logσ = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
A, μ, logσ = Dense(nx, nh), Dense(nh, nz), Dense(nh, nz)
g(X) = (h = A(X); (μ(h), logσ(h)))
#f = Chain(Dense(nz,nh,swish),Dense(nh,nx)) 
f = Chain(Dense(nz,nh),Dense(nh,nx)) 
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)


s = 100;#param(rand(2,1))
#loss(x,z) = Flux.mse(f(mu(x)+exp.(0.5*lsig(x)).*z),x) + KL(mu(x),lsig(x))
function loss(x)
    #s+exp.(s).*0.5*sum((x.-f(z_given_x)).^2) + KL(mu(x),lsig(x))
    HID = A(x);
    zsample = z(μ(HID),logσ(HID));
    0.5*sum((x.-f(zsample)).^2) .+ s*KL(μ(HID),logσ(HID))
end
function loss2(x)
    #s+exp.(s).*0.5*sum((x.-f(z_given_x)).^2) + KL(mu(x),lsig(x))
    HID = A(x);
    zsample = z(μ(HID),logσ(HID));
    (0.5*sum((x.-f(zsample)).^2) , (s*KL(μ(HID),logσ(HID))))
end
ps = Flux.params(A, μ, logσ, f)
da = Iterators.repeated((x,),1000)
opt = Flux.ADAM(1e-2)
for n=1:10
    Flux.train!(loss,ps,da,opt);
    println(loss(x))
end


xg = -3:0.1:7; # grid of x
zg = -2:0.1:2; # grid of z


z_xg =μ(A(xg'));

x_zg = f(zg');
plot(xg,z_xg[:];label="encoder");
plot!(x_zg[:],zg;label="decoder")