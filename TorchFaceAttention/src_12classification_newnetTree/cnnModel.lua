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

    if (inheritModel) then
        print("inherit model node from parent model that a CNN model trained without tree")
        parentModel = torch.load("results/model.net")

        -- stage 1
        modelNode[1][1]:add(parentModel:get(1)) -- nn.SpatialConvolution(1, 16, 5, 5)
        modelNode[1][1]:add(parentModel:get(2)) -- nn.ReLU()
        modelNode[1][1]:add(parentModel:get(3)) -- nn.SpatialMaxPooling(2,2,2,2)

        -- stage 2
        modelNode[2][1]:add(parentModel:get(4)) -- nn.SpatialConvolution(16, 32, 3, 3)
        modelNode[2][1]:add(parentModel:get(5)) -- nn.ReLU()
        modelNode[2][1]:add(parentModel:get(6)) -- nn.SpatialMaxPooling(2,2,2,2)

        modelNode[2][2] = modelNode[2][1]:clone()

        -- stage 3
        modelNode[3][1]:add(parentModel:get(7)) -- nn.SpatialConvolution(32, 64, 3, 3)
        modelNode[3][1]:add(parentModel:get(8)) -- nn.ReLU()
        modelNode[3][1]:add(parentModel:get(9)) -- nn.SpatialConvolution(64, 128, 3, 3)
        modelNode[3][1]:add(parentModel:get(10)) -- nn.ReLU()
        modelNode[3][1]:add(parentModel:get(11)) -- nn.SpatialMaxPooling(2,2,2,2)
        modelNode[3][1]:add(parentModel:get(12)) -- nn.Reshape(128)
        modelNode[3][1]:add(parentModel:get(13)) -- nn.Linear(128, 2)
        modelNode[3][1]:add(parentModel:get(14)) -- nn.LogSoftMax()

        modelNode[3][2] = modelNode[3][1]:clone()
        modelNode[3][3] = modelNode[3][1]:clone()
        modelNode[3][4] = modelNode[3][1]:clone()
    else
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
    end

    -- Decision Tree Node
    decisionTreeNode = {}
    decisionTreeNode[1] = {nn.Sequential()}
    decisionTreeNode[2] = {nn.Sequential(),nn.Sequential()}

    decisionTreeNode[1][1]:add(nn.Reshape(14*14*16))
    decisionTreeNode[1][1]:add(nn.Linear(14*14*16,1))

    decisionTreeNode[2][1]:add(nn.Reshape(6*6*32))
    decisionTreeNode[2][1]:add(nn.Linear(6*6*32,1))

    decisionTreeNode[2][2] = decisionTreeNode[2][1]:clone()
end

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

TreeModels = {nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential()}
function modelGenerater(branchNum)
    TreeModels[branchNum]:add(modelNode[1][1])
    TreeModels[branchNum]:add(modelNode[2][math.ceil(branchNum/2)])
    TreeModels[branchNum]:add(modelNode[3][branchNum])

    if enableCuda then
        TreeModels[branchNum]:cuda()
    end

end

for i = 1,4 do 
    modelGenerater(i)
end

----------------------------------------------------------------------
print '==> here is the model:'
print(TreeModels)

function TreeModelForward(input,training)
    local firstLayerOutput = modelNode[1][1]:forward(input)
    local firstDecisionOutput = decisionTreeNode[1][1]:forward(firstLayerOutput)
    local firstRoute
    if (firstDecisionOutput[1] >= 0) then
        firstRoute = 1
        print(firstRoute)
    else
        firstRoute = 2
        print(firstRoute)
    end
    local secondLayerOutput = modelNode[2][firstRoute]:forward(firstLayerOutput)
    local secondDecisionOutput = decisionTreeNode[2][firstRoute]:forward(secondLayerOutput)
    local secondRoute
    if (secondDecisionOutput[1] >= 0) then
        secondRoute = (firstRoute * 2 - 1)
        print(secondRoute)
    else
        secondRoute = (firstRoute * 2)
        print(secondRoute)
    end
    local thirdLayerOutput = modelNode[3][secondRoute]:forward(secondLayerOutput)
    return thirdLayerOutput,firstDecisionOutput,secondDecisionOutput,firstRoute,secondRoute
end
