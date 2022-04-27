using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using EvalMetrics
using Random
using Plots

function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

function csv2mill(problem)
	x=readdlm("$(problem)/data.csv",'\t',Float32)
	bagids = readdlm("$(problem)/bagids.csv",'\t',Int)[:]
	bags = seqids2bags(bagids)
	y = readdlm("$(problem)/labels.csv",'\t',Int)
	y = map(b -> maximum(y[b]), bags)
	(samples = BagNode(ArrayNode(x), bags), labels = y)
end

function labels2instances(x)
    ym=map(x->x[1]*ones(x[2]),zip(softmax(model(x)),length.(x.bags)))
	y=vcat(ym...)
	return [y';(1.0 .-y)']
end


data = "D:/VU/SCRIPTS/DataSets/Musk1"
(x,y) = csv2mill(data)
y_oh= Flux.onehotbatch((y.+1)[:],1:2) 
y_oh_i = labels2instances(x)

v=size(x.data.data)[2]

model = BagModel(
    Dense(size(x.data.data)[1], 10, Flux.tanh),
    BagCount(SegmentedMeanMax(10)),
    Chain(Dense(21, 20, Flux.tanh),Dense(20,10,Flux.tanh), Dense(10, 2)))

# define loss function
loss(x, y_oh) = Flux.logitcrossentropy(model(x), y_oh) 

## NN == inicializace
nx = size(x.data.data)[1]
nz = 16 #(4 8 16 32 64)
nh = 20 #prizpusobit

q_yz =  Chain(Dense(nx+2,nh+5,tanh),Dense(nh+5,nh)) # encoder
μ, logσ = Dense(nh, nz,tanh), Dense(nh, nz,tanh)
f = Chain(Dense(nz+2,nh,selu),Dense(nh,nx)) #decoder

function encoder(x,y_oh)
    v=vcat(x.data.data,y_oh)
    h = q_yz(v)
    return h
end

function decoder(z,y_oh_i)
    v=vcat(z,y_oh_i)
    h = f(v)
    return h
end

z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ)) #samples
KL(μ, logσ) = 0.5 *sum(-1.0 .-  2 .*logσ .+ (exp.(logσ)).^2 .+ μ.^2, dims=1) #KLdiv



s=0.02 #(0.02 pro hybrid loss1, 0.08 pro hybrodloss2)
function L(x,y_oh_i)      #VAE loss    , x jsou instance      
    HID = encoder(x,y_oh_i)
    zsample = z(μ(HID),logσ(HID))
    K = 0.5*diag((x.data.data .-decoder(zsample,y_oh_instances))' *(x.data.data .-decoder(zsample,y_oh_instances))) .+ s*KL(μ(HID),logσ(HID))'
    return K
end

function hybridVAE(x)   
    yp = labels2instances(x)
    y_oh1 = Flux.onehotbatch((ones(v).+1)[:],1:2)
    y_oh2 = Flux.onehotbatch((zeros(v).+1)[:],1:2)
    Li = yp[1,:].*L(x,y_oh1) + yp[2,:].*L(x,y_oh2) 
    LL = sum(Li) - Flux.logitcrossentropy(yp,yp,agg=sum)
   return LL
end

β = 10000
function hybridloss2(x,y_oh)
    hybridVAE(x) + β*loss(x,y_oh)
end

ps = Flux.params(q_yz, μ, logσ, f,model)
da = Iterators.repeated((x,y_oh),10000)
opt = Flux.ADAM()
Flux.train!(hybridloss2,ps,da,opt)