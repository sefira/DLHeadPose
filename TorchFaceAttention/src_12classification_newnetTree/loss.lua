--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
if ClassNLL then
	criterion = nn.ClassNLLCriterion()
else 
	criterion = nn.DistKLDivCriterion()
end

-- and move it to the GPU:
if enableCuda then
	criterion:cuda()
end
