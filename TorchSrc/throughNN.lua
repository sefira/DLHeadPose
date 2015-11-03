require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'gnuplot' -- display a image
----------------------------------------------------------------------
print '==> trainData through NN'

--define data which from NN output
trainNNout = {}
for i = 1,trainData:size() do
	xlua.progress(i, trainData:size())
	-- select a new training sample

	inputs = trainData.data[i]
	inputs = inputs:double()
	outputNN = throughNN(inputs)
	trainNNout[i] = outputNN:clone()
end
