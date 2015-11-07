--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'
require 'nn'
require 'cunn'

----------------------------------------------------------------------

-- define model to train
model = nn.Sequential()

-- stage 1
model:add(nn.SpatialConvolution(1, 6, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Tanh())

-- stage 2 
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Tanh())

-- stage 3 
model:add(nn.SpatialConvolution(16, 120, 5, 5))
model:add(nn.Tanh())
model:add(nn.Reshape(120))
model:add(nn.Linear(120, 2))
-- model:add(nn.Tanh())
 
-- and move it to the GPU:
model:cuda()

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

--visualize = true
visualize = false
if visualize then
	if itorch then
		print '==> visualizing ConvNet filters'
		print('Layer 1 filters:')
		itorch.image(model:get(1).weight)
		print('Layer 2 filters:')
		itorch.image(model:get(5).weight)
		print('Layer 3 filters:')
		itorch.image(model:get(9).weight)
	else
		print '==> To visualize filters, start the script in itorch notebook'
	end
end