require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions
require 'optim'

print '==> test in testData'


testNNout = {}
for i = 1,testData:size() do
	xlua.progress(i, testData:size())
	-- select a new training sample

	inputs = testData.data[i]
	inputs = inputs:double()
	outputNN = throughNN(inputs)
	testNNout[i] = outputNN:clone()
end


function testinTestData()
	for t = 1,testData:size() do
		-- disp progress
		xlua.progress(t, testData:size())

		-- get new sample
		target = testData.labels[t]
		inputs = testNNout[t]

		-- test sample
		local pred = regresModel:forward(inputs)
		confusion:add(pred, target)
	end
	print(confusion)
	confusion:zero()
end
