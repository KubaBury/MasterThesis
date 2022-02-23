using Distributions
using Flux
using LaTeXStrings

N = 100; # number of samples
K = 5 # number of negative examples for each x

dataDist = Normal(5, 2); #distribution of the data
x = rand(dataDist, N) #training data

y1 = hcat(zeros(1,Int(N/2)),ones(1,Int(N/2))) #labels
y2 = gen_neg_examples(N,K)
y = vcat(y1,y2)

"""
Generates negative examples.
"""
function gen_neg_examples(N::Int,K::Int)
    y2=zeros(K,N)
    for i in 1:K
        y2[i,:] = wsample([0,1], [0.4,0.6], N)
    end
    return y2   
end


"""
Defines score function.
"""
function score()

end

function loss_ranking()

end



opt = Flux.ADAM(); #optimalization method
Flux.train!((x, y) -> loss_ranking(x, y), [μ, σ, c], Iterators.repeated((x, y), 10000), opt) # train binary
println("μ=$μ, σ=$σ, c=$c") #results

