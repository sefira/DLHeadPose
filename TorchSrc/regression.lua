require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------

-- multi-class problem

----------------------------------------------------------------------
print '==> define regression model'
local input = noutputs
local output = tablelength(labels_id)
regresModel = nn.Sequential()
regresModel:add(nn.Linear(input,output))
regresModel:add(nn.LogSoftMax())

print '==> define loss function'
--criterion = nn.ClassNLLCriterion()
criterion = nn.DistKLDivCriterion()
----------------------------------------------------------------------
print '==> here is the regression model:'
print(regresModel)
print '==> here is the loss function:'
print(criterion)

