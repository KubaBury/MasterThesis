using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using Plots

data = "D:/VU/SCRIPTS/DataSets/Musk2"
###BrownCreeper, CorelAfrican,CorelBeach, Elephant,Fox, Musk1, Musk2, Mutagenesis1(2)
###Newsgroups1, Newsgroups2, Newsgroups3, Protein, Tiger, Web1(2,3,4), WinterWren


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

###### load data ######

(x,y) = csv2mill(data)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)

###### cross-validation - specify number of folds and dense layer ###### 


function KFCV(K,W,h)
	b = length(y) #number of bags
	A = length(x.data.data[:,1]) # for dense layer
	n = floor(Int, length(y)/K); # number of bags in each fold
	L2 = zeros(W,K); # initial position of loss matrix
	L3 = zeros(W,K); # initial position of loss matrix
	L6 = zeros(K)
	L7 = zeros(K)
	train_sets = zeros(Int, b-n, K);
	test_sets = zeros(Int, n, K);

	opt = Flux.ADAM()
K=5
	### define random training and validation samples
	for j = 1:K
		l1 = (1:b)
		l2 = (1+n*(j-1):n+n*(j-1))
		q = symdiff(l1,l2)
		train_sets[:,j] = q
		test_sets[:,j] = l2
	end

	for i = 1:W
		# create the model
		model = BagModel(
    	ArrayModel(Dense(A, h*i, Flux.tanh)),                      			# model on the level of Flows
    	meanmax_aggregation(h*i),                                      		# aggregation
		ArrayModel(Chain(Dense(2*h*i+1, h*i, Flux.tanh), Dense(h*i, 2)))) ; 	# model on the level of bags
		
		for j = 1:K
			#loss function
			loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);
			#train
			Flux.train!(loss, Flux.params(model), repeated((x[train_sets[:,j]], y_oh[:,train_sets[:,j]]), 1000), opt);
			# calculate loss for each fold and dense layer
			L2[i,j] = loss(x[test_sets[:,j]], y_oh[:,test_sets[:,j]])
			L3[i,j] = loss(x[train_sets[:,j]], y_oh[:,train_sets[:,j]])	
			L6[j]= 1-mean(mapslices(argmax, model(x[train_sets[:,j]]).data, dims=1)' .!= y[train_sets[:,j]].+1)
			L7[j]= 1-mean(mapslices(argmax, model(x[test_sets[:,j]]).data, dims=1)' .!= y[test_sets[:,j]].+1)
		end;
	
	end

	#visulazation
	x2 = mean(L4, dims = 2)
	x3 = mean(L5, dims = 2)
	x1 = 1:W
	plot(x1, x2, xlabel = "Model Complexity", ylabel = "Prediction Error", 
	label = "Testing data",lw=3,
	legend=:topleft, color=:blue, title="KfoldCV, Musk2",
	xtickfont=font(11), 
    ytickfont=font(11),
    guidefont=font(14),
    legendfont=font(11))
	plot!(x1, x3,label = "Training data", lw=3, legend=:topleft, color=:red)
	L6,L7
end

