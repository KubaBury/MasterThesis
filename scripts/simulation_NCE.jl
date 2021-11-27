using Statistics
using Distributions
using Flux
using Random 
using StatsBase

### Model definition ###
X = randperm(100); #feature space X, mx = 100
Y=[1.0, 2.0, 3.0, 4.0] #labels
θ = [1.05, 2.1, 3.9, 4.0]; #parameters
N = 1000;#number of samples
x = sample(X, N); #samples

s(x,θ)= x'.*θ; #score 
sampleDist = exp.(s(x,θ)) ./ sum(exp.(s(x,θ)),dims=1)  #distribution used to sample labels

y=copy(x)
for i = 1:size(w)[2]    
    y[i]= sample(Y,Weights(sampleDist[:,i]))   
end
###MLE### 

y_oh=Flux.onehotbatch((y)[:],1:4) #one-hot encoding
loss(x,y_oh) = Flux.logitcrossentropy(s(x,θ), y_oh) # loss function
opt = Flux.ADAM(); #optimalization method
Flux.train!(loss, [θ], Iterators.repeated((x, y_oh), 10000), opt) # train

θ₁  = θ
s(x,θ₁)

### NCE - ranking ###
negSampleDist = Binomial(1,0.9);
y_neg = rand(Binomial(1,0.9),N);
K = 1000 - sum(y_neg);
logp_N = logpdf(negSampleDist,y)
s1(x,θ)=s(x,θ) .-logp_N'  
A = exp.(s(x,θ)) ./ sum(exp.(s(x,θ)), dims=1)


