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
        for i=1,#modelNode[1] do
            modelNode[1][i]:add(nn.SpatialConvolution(1, 16, 5, 5))
            modelNode[1][i]:add(nn.ReLU())
            modelNode[1][i]:add(nn.SpatialMaxPooling(2,2,2,2))
        end

        -- stage 2
        for i=1,#modelNode[2] do
            modelNode[2][i]:add(nn.SpatialConvolution(16, 32, 3, 3))
            modelNode[2][i]:add(nn.ReLU())
            modelNode[2][i]:add(nn.SpatialMaxPooling(2,2,2,2))
        end

        -- stage 3
        for i=1,#modelNode[3] do
            modelNode[3][i]:add(nn.SpatialConvolution(32, 64, 3, 3))
            modelNode[3][i]:add(nn.ReLU())
            modelNode[3][i]:add(nn.SpatialConvolution(64, 128, 3, 3))
            modelNode[3][i]:add(nn.ReLU())
            modelNode[3][i]:add(nn.SpatialMaxPooling(2,2,2,2))
            modelNode[3][i]:add(nn.Reshape(128))
            modelNode[3][i]:add(nn.Linear(128, 2))
            modelNode[3][i]:add(nn.LogSoftMax())
        end

    end

    -- Decision Tree Node
    decisionTreeNode = {}
    decisionTreeNodeReshape = {}
    decisionTreeNode[1] = {nn.Sequential()}
    decisionTreeNode[2] = {nn.Sequential(),nn.Sequential()}
    decisionTreeNodeReshape[1] = {}
    decisionTreeNodeReshape[2] = {}

    for i=1,#decisionTreeNode[1] do
        decisionTreeNodeReshape[1][i] = nn.Reshape(14*14*16)
        decisionTreeNode[1][i]:add(decisionTreeNodeReshape[1][i])
        decisionTreeNode[1][i]:add(nn.Linear(14*14*16,1))
    end

    for i=1,#decisionTreeNode[2] do
        decisionTreeNodeReshape[2][i] = nn.Reshape(6*6*32)
        decisionTreeNode[2][i]:add(decisionTreeNodeReshape[2][i])
        decisionTreeNode[2][i]:add(nn.Linear(6*6*32,1))
    end

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
    local firstDecisionInput = decisionTreeNodeReshape[1][1].output
    local firstRoute
    if (firstDecisionOutput[1] >= 0) then
        firstRoute = 1
        --print(firstRoute)
        --print(firstDecisionOutput[1])
    else
        firstRoute = 2
        --print(firstRoute)
        --print(firstDecisionOutput[1])
    end
    local secondLayerOutput = modelNode[2][firstRoute]:forward(firstLayerOutput)
    local secondDecisionOutput = decisionTreeNode[2][firstRoute]:forward(secondLayerOutput)
    local secondDecisionInput = decisionTreeNodeReshape[2][firstRoute].output
    local secondRoute
    if (secondDecisionOutput[1] >= 0) then
        secondRoute = (firstRoute * 2 - 1)
        --print(secondRoute)
        --print(secondDecisionOutput[1])
    else
        secondRoute = (firstRoute * 2)
        --print(secondRoute)
        --print(secondDecisionOutput[1])
    end
    local thirdLayerOutput = modelNode[3][secondRoute]:forward(secondLayerOutput)
    return thirdLayerOutput,firstDecisionInput,firstDecisionOutput[1],secondDecisionInput,secondDecisionOutput[1],firstRoute,secondRoute
end

function DecisionNodesBackward(statusOfDecision, arrivedCount, gradParameters)
    -- greater than 7 indicate there are dicision tree node parameters
    if #gradParameters > 7 then
        -- local statusOfDecision = {}
        -- statusOfDecision[1] = {
        --     sumOfPhi = {torch.zeros(1),torch.zeros(1)},
        --     sumOfPhiX = torch.zeros(14*14*16),
        --     sumOfPhiSqrt = torch.zeros(1),
        --     sumOfX = {torch.zeros(14*14*16),torch.zeros(14*14*16)},
        --     gradWeight = torch.Tensor(14*14*16),
        --     gradBias = torch.Tensor(1),
        --     err = 100
        -- }

        for i=1,#statusOfDecision do
            local sumOfPhiLeft = statusOfDecision[i].sumOfPhi[1][1]
            local sumOfPhiRight = statusOfDecision[i].sumOfPhi[2][1]
            local _diffAvgSumOfPhi = (sumOfPhiLeft / arrivedCount[i*2]) - (sumOfPhiRight / arrivedCount[i*2 + 1])
            if _diffAvgSumOfPhi == 0 then
                --print(i.." not arrived")
                statusOfDecision[i].gradWeight:zero()
                statusOfDecision[i].gradBias:zero()
                gradParameters[#gradParameters - 3 + i]:zero()
                --statusOfDecision[i].err = 0
            else 
                local _sumOfPhiX = statusOfDecision[i].sumOfPhiX
                local _sumOfPhiSqrt = statusOfDecision[i].sumOfPhiSqrt[1]
                local sumOfXLeft = statusOfDecision[i].sumOfX[1][1]
                local sumOfXRight = statusOfDecision[i].sumOfX[2][1]
                local _diffAvgSumOfX = (sumOfXLeft / arrivedCount[i*2]) - (sumOfXRight / arrivedCount[i*2 + 1])
                local _sumOfPhi = sumOfPhiLeft + sumOfPhiRight

                statusOfDecision[i].gradWeight:copy((_sumOfPhiX * _diffAvgSumOfPhi) - (_diffAvgSumOfX * _sumOfPhiSqrt))
                statusOfDecision[i].gradWeight:mul(2):div(arrivedCount[i]):div(_diffAvgSumOfPhi^3)
                statusOfDecision[i].gradBias:fill(2 * _sumOfPhi / arrivedCount[i] / (_diffAvgSumOfPhi^2))

                gradParameters[#gradParameters - 3 + i]:copy(torch.cat(statusOfDecision[i].gradWeight, statusOfDecision[i].gradBias))
                local old_err = statusOfDecision[i].err
                statusOfDecision[i].err = _sumOfPhiSqrt / arrivedCount[i] / (_diffAvgSumOfPhi ^ 2)
                --print("statusOfDecision[" ..i.."].err")
                if lossOfDecisionNodes then
                    lossOfDecisionNodes[i][#lossOfDecisionNodes[i]+1] = statusOfDecision[i].err
                end
            end
        end
    end
end