require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------

criterion = nn.MSECriterion()
criterion.sizeAverage = false

criterion:cuda()

-- Compared to the other losses, the MSE criterion needs a distribution
-- as a target, instead of an index. Indeed, it is a regression loss!
-- So we need to transform the entire label vectors:

tanhTheResult = false

if tanhTheResult then
	-- convert training labels:
	local trsize = (#trainData.labels)[1]
	local trlabels = torch.Tensor( trsize, noutputs )
	trlabels:fill(-1)
	for i = 1,trsize do
		trlabels[{ i,trainData.labels[i] }] = 1
	end
	trainData.labels = trlabels

	-- convert test labels
	local tesize = (#testData.labels)[1]
	local telabels = torch.Tensor( tesize, noutputs )
	telabels:fill(-1)
	for i = 1,tesize do
		telabels[{ i,testData.labels[i] }] = 1
	end
	testData.labels = telabels
end