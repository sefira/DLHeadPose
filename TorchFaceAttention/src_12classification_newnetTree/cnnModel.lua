--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'
require 'nn'

----------------------------------------------------------------------
loadModel = loadModel or false

if (loadModel) then	
	modelNode = {}
	modelNode[1] = {nn.Sequential()}
	modelNode[2] = {nn.Sequential(),nn.Sequential()}
	modelNode[3] = {nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential()}
	modelNode[1][1] = torch.load("results/modelNode11.net")
	modelNode[2][1] = torch.load("results/modelNode21.net")
	modelNode[2][2] = torch.load("results/modelNode22.net")
	modelNode[3][1] = torch.load("results/modelNode31.net")
	modelNode[3][2] = torch.load("results/modelNode32.net")
	modelNode[3][3] = torch.load("results/modelNode33.net")
	modelNode[3][4] = torch.load("results/modelNode34.net")

	decisionTreeNode = {}
	decisionTreeNode[1] = {nn.Sequential()}
	decisionTreeNode[2] = {nn.Sequential(),nn.Sequential()}
	decisionTreeNode[1][1] = torch.load("results/decisionTreeNode11.net")
	decisionTreeNode[2][1] = torch.load("results/decisionTreeNode21.net")
	decisionTreeNode[2][2] = torch.load("results/decisionTreeNode22.net")
end

if (not loadModel) then	
	modelNode = {}
	modelNode[1] = {nn.Sequential()}
	modelNode[2] = {nn.Sequential(),nn.Sequential()}
	modelNode[3] = {nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential()}

	-- stage 1
	modelNode[1][1]:add(nn.SpatialConvolution(1, 16, 5, 5))
	modelNode[1][1]:add(nn.ReLU())
	modelNode[1][1]:add(nn.SpatialMaxPooling(2,2,2,2))

	-- stage 2 
	modelNode[2][1]:add(nn.SpatialConvolution(16, 32, 3, 3))
	modelNode[2][1]:add(nn.ReLU())
	modelNode[2][1]:add(nn.SpatialMaxPooling(2,2,2,2))

	modelNode[2][2] = modelNode[2][1]:clone()

	-- stage 3
	modelNode[3][1]:add(nn.SpatialConvolution(32, 64, 3, 3))
	modelNode[3][1]:add(nn.ReLU())
	modelNode[3][1]:add(nn.SpatialConvolution(64, 128, 3, 3))
	modelNode[3][1]:add(nn.ReLU())
	modelNode[3][1]:add(nn.SpatialMaxPooling(2,2,2,2))
	modelNode[3][1]:add(nn.Reshape(128))
	modelNode[3][1]:add(nn.Linear(128, 2))
	modelNode[3][1]:add(nn.LogSoftMax())

	modelNode[3][2] = modelNode[3][1]:clone()
	modelNode[3][3] = modelNode[3][1]:clone()
	modelNode[3][4] = modelNode[3][1]:clone()

	-- Decision Tree Node
	decisionTreeNode = {}
	decisionTreeNode[1] = {nn.Sequential()}
	decisionTreeNode[2] = {nn.Sequential(),nn.Sequential()}

	decisionTreeNode[1][1]:add(nn.Reshape(14*14*16))
	decisionTreeNode[1][1]:add(nn.Linear(14*14*16,1))

	decisionTreeNode[2][1]:add(nn.Reshape(6*6*32))
	decisionTreeNode[2][1]:add(nn.Linear(6*6*32,1))

	decisionTreeNode[2][2] = decisionTreeNode[2][1]:clone()


	-- and move these to the GPU:
	if enableCuda then
		for i = 1,#modelNode do
			for j = 1,#modelNode[i] do 
				modelNode[i][j]:cuda()
			end
		end
		for i = 1,#decisionTreeNode do
			for j = 1,#decisionTreeNode[i] do
				decisionTreeNode[i][j]:cuda()
			end
		end
	end

end

model = {nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential()}
function modelGeneraterImplements(branchNum)
	model[branchNum].add(modelNode[1][math.ceil(branchNum/4)])
	model[branchNum].add(modelNode[2][math.ceil(branchNum/2)])
	model[branchNum].add(modelNode[3][math.ceil(branchNum/1)])

	if enableCuda then
		model[branchNum]:cuda()
	end

end

for i = 1,4 do 
	modelGeneraterImplements(i)
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)
