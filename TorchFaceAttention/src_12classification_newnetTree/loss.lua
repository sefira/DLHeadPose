--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
model:add(nn.LogSoftMax())
if ClassNLL then
	criterion = nn.ClassNLLCriterion()
else 
	criterion = nn.DistKLDivCriterion()
end

-- and move it to the GPU:
if enableCuda then
	model:cuda()
	criterion:cuda()
end
