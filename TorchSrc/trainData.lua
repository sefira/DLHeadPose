require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions
require 'optim'

print '==> defining trainning procedure'
-- params/gradients
x, dl_dx = regresModel:getParameters()


epochs = 1e3  -- number of times to cycle over our training data

----------------------------------------------------------------------
-- 4.b. Train the model (Using L-BFGS)

regresModel:reset()

feval = function(x_new)
	-- set x to x_new, if differnt
	-- (in this simple example, x_new will typically always point to x,
	-- so the copy is really useless)
	if x ~= x_new then
		x:copy(x_new)
	end

	-- reset gradients (gradients are always accumulated, to accomodate 
	-- batch methods)
	dl_dx:zero()

	-- and batch over the whole training dataset:
	local loss_x = 0
	for i = 1,trainData.size() do
		-- select a new training sample
		_nidx_ = (_nidx_ or 0) + 1
		if _nidx_ > trainData.size() then _nidx_ = 1 end
		
		
		--target = trainData.labels[_nidx_]
		inputs = trainNNout[_nidx_]
		target = torch.Tensor(tablelength(labels_id)):zero()
		--print(target)
		target[trainData.labels[_nidx_]] = 1
		--print(#inputs)
		--print(target)
		-- evaluate the loss function and its derivative wrt x, for that sample
		loss_x = loss_x + criterion:forward(regresModel:forward(inputs), target)
		regresModel:backward(inputs, criterion:backward(regresModel.output, target))
	end

	-- normalize with batch size
	loss_x = loss_x / trainData.size()
	dl_dx = dl_dx:div( trainData.size() )

	-- return loss(x) and dloss/dx
	return loss_x, dl_dx
end

lbfgs_params = {
	lineSearch = optim.lswolfe,
	maxIter = epochs,
	verbose = true,
	learningRate = 0.1
}

function train()
	print '==> training start under lbfgs_params:'
	print(lbfgs_params)
	_,fs = optim.lbfgs(feval,x,lbfgs_params)
	print '==> training finished'
end

-- classes
classes = {}
for k,v in pairs(labels_id)do
	classes[v] = k
end

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
function testinTrainData()
	print '==> test in trainData'
	for t = 1,trainData:size() do
		-- disp progress
		xlua.progress(t, trainData:size())

		-- get new sample
		target = trainData.labels[t]
		inputs = trainNNout[t]

		-- test sample
		local pred = regresModel:forward(inputs)
		confusion:add(pred, target)
	end
	print(confusion)
	confusion:zero()
end
