--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'
require 'nn'

----------------------------------------------------------------------
classicalModel = false

-- define model to train
model = nn.Sequential()

if classicalModel then
	-- stage 1
	model:add(nn.SpatialConvolution(1, 6, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialAveragePooling(2,2,2,2))

	-- stage 2 
	model:add(nn.SpatialConvolution(6, 16, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialAveragePooling(2,2,2,2))

	-- stage 3 
	model:add(nn.SpatialConvolution(16, 120, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.Reshape(120))
	model:add(nn.Linear(120, 2))
else
	-- stage 1
	model:add(nn.SpatialConvolution(1, 16, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2,2,2,2))

	-- stage 2 
	model:add(nn.SpatialConvolution(16, 32, 3, 3))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2,2,2,2))

	-- stage 3 
	model:add(nn.SpatialConvolution(32, 64, 3, 3))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(64, 128, 3, 3))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2,2,2,2))
	model:add(nn.Reshape(128))
	model:add(nn.Linear(128, 2))
end

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
