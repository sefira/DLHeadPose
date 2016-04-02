--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'
require 'nn'

----------------------------------------------------------------------
loadModel = loadModel or true

-- define model to train
model = nn.Sequential()
modelNode11 = nn.Sequential()
modelNode21 = nn.Sequential()
modelNode22 = nn.Sequential()
modelNode31 = nn.Sequential()
modelNode32 = nn.Sequential()
modelNode33 = nn.Sequential()
modelNode34 = nn.Sequential()

if (loadModel) then
	modelNode11 = torch.load("results/modelNode11.net")
	modelNode21 = torch.load("results/modelNode21.net")
	modelNode22 = torch.load("results/modelNode22.net")
	modelNode31 = torch.load("results/modelNode31.net")
	modelNode32 = torch.load("results/modelNode32.net")
	modelNode33 = torch.load("results/modelNode33.net")
	modelNode34 = torch.load("results/modelNode34.net")
end

if (not loadModel) then
	-- stage 1
	modelNode11:add(nn.SpatialConvolution(1, 16, 5, 5))
	modelNode11:add(nn.ReLU())
	modelNode11:add(nn.SpatialMaxPooling(2,2,2,2))

	-- stage 2 
	modelNode21:add(nn.SpatialConvolution(16, 32, 3, 3))
	modelNode21:add(nn.ReLU())
	modelNode21:add(nn.SpatialMaxPooling(2,2,2,2))

	modelNode22 = modelNode21:clone()

	-- stage 3
	modelNode31:add(nn.SpatialConvolution(32, 64, 3, 3))
	modelNode31:add(nn.ReLU())
	modelNode31:add(nn.SpatialConvolution(64, 128, 3, 3))
	modelNode31:add(nn.ReLU())
	modelNode31:add(nn.SpatialMaxPooling(2,2,2,2))
	modelNode31:add(nn.Reshape(128))
	modelNode31:add(nn.Linear(128, 2))

	modelNode32 = modelNode31:clone()
	modelNode33 = modelNode31:clone()
	modelNode34 = modelNode31:clone()
end

model:add(modelNode11)

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
