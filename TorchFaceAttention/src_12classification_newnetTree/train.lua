require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat('results', 'train.log'))
testLogger = optim.Logger(paths.concat('results', 'test.log'))

----------------------------------------------------------------------
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters = {}
gradParameters = {}
-- 1: parameters of first Node, 2~3: parameters of second Node, 4~7: parameters of third Node, 8~10: parameters of decisionTree Node
-- 1: gradParameters of first Node, 2~3: gradParameters of second Node, 4~7: gradParameters of third Node, 8~10: gradParameters of decisionTree Node
if TreeModels then
    countParameters = 1
    for i = 1,#modelNode do
        for j = 1,#modelNode[i] do 
            parameters[countParameters],gradParameters[countParameters] = modelNode[i][j]:getParameters()
            countParameters = countParameters + 1
        end
    end
    for i = 1,#decisionTreeNode do
        for j = 1,#decisionTreeNode[i] do
            parameters[countParameters],gradParameters[countParameters] =  decisionTreeNode[i][j]:getParameters()
            countParameters = countParameters + 1
        end
    end
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
    learningRate = 1e-3,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
}
--optimMethod = optim.sgd
optimMethod = multiModelSgd

----------------------------------------------------------------------
print '==> defining training procedure'
batchSize = 1000
function train()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set TreeModels to training mode (for modules that differ in training and testing, like Dropout)
    for i = 1,4 do 
        TreeModels[i]:training()
    end

    -- shuffle at each epoch
    shuffle = torch.randperm(trsize)

    -- do one epoch
    current_loss = 0
    print('\n\n#######################################################')
    print('#######################################################')
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
    for t = 1,trsize,batchSize do
        -- disp progress
        xlua.progress(t, trsize)

        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+batchSize-1,trsize) do
            -- load new sample
            local input = train_data[shuffle[i]].data
            local target = train_data[shuffle[i]].labels
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
                        -- get new parameters
                        for i=1,#parameters do
                            if x[i] ~= parameters[i] then
                                parameters[i]:copy(x[i])
                            end
                        end

                        -- reset gradients
                        for i=1,#gradParameters do
                            gradParameters[i]:zero()
                        end
                        -- f is the average of all criterions, actually won't be used in sgd.lua
                        local f = 0
                        local arrivedCount = {}
                        for i=1,#gradParameters do
                            arrivedCount[i] = 0
                        end

                        statusOfDecision = {}
                        statusOfDecision[1] = {
                            sumOfPhi = {torch.zeros(1),torch.zeros(1)},
                            sumOfPhiX = torch.zeros(14*14*16),
                            sumOfPhiSqrt = torch.zeros(1),
                            sumOfX = {torch.zeros(14*14*16),torch.zeros(14*14*16)},
                            gradWeight = torch.Tensor(14*14*16),
                            gradBias = torch.Tensor(1),
                            err = 100
                        }
                        for i=2,3 do
                            statusOfDecision[i] = {
                                sumOfPhi = {torch.zeros(1),torch.zeros(1)},
                                sumOfPhiX = torch.zeros(6*6*32),
                                sumOfPhiSqrt = torch.zeros(1),
                                sumOfX = {torch.zeros(6*6*32),torch.zeros(6*6*32)},
                                gradWeight = torch.Tensor(6*6*32),
                                gradBias = torch.Tensor(1),
                                err = 100
                            }
                        end
                        if enableCuda then
                            for i=1,#statusOfDecision do
                                statusOfDecision[i].sumOfPhi[1] = statusOfDecision[i].sumOfPhi[1]:cuda()
                                statusOfDecision[i].sumOfPhi[2] = statusOfDecision[i].sumOfPhi[2]:cuda()
                                statusOfDecision[i].sumOfPhiX = statusOfDecision[i].sumOfPhiX:cuda()
                                statusOfDecision[i].sumOfPhiSqrt = statusOfDecision[i].sumOfPhiSqrt:cuda()
                                statusOfDecision[i].sumOfX[1] = statusOfDecision[i].sumOfX[1]:cuda()
                                statusOfDecision[i].sumOfX[2] = statusOfDecision[i].sumOfX[2]:cuda()
                                statusOfDecision[i].gradWeight = statusOfDecision[i].gradWeight:cuda()
                                statusOfDecision[i].gradBias = statusOfDecision[i].gradBias:cuda()
                            end
                        end

                        -- evaluate function for complete mini batch
                        for i = 1,#inputs do
                            -- estimate f
                            thirdLayerOutput,firstDecisionInput,firstDecisionOutput,secondDecisionInput,secondDecisionOutput,firstRoute,secondRoute = TreeModelForward(inputs[i])
                            local err = criterion:forward(thirdLayerOutput, targets[i])
                            f = f + err
                            
                            arrivedCount[1] = arrivedCount[1] + 1
                            arrivedCount[firstRoute + 1] = arrivedCount[firstRoute + 1] + 1
                            arrivedCount[secondRoute + 3] = arrivedCount[secondRoute + 3] + 1

                            statusOfDecision[1].sumOfPhi[firstRoute]:   add(firstDecisionOutput)
                            statusOfDecision[1].sumOfPhiX:              add(firstDecisionInput * firstDecisionOutput)
                            statusOfDecision[1].sumOfPhiSqrt:           add(firstDecisionOutput * firstDecisionOutput)
                            statusOfDecision[1].sumOfX[firstRoute]:     add(firstDecisionInput)
                            --print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                            --print(firstRoute)
                            --print(firstDecisionOutput)
                            --print(firstDecisionOutput * firstDecisionOutput)
                            --print(firstDecisionInput[1])

                            statusOfDecision[firstRoute + 1].sumOfPhi[((secondRoute + 1) % 2 + 1)]: add(secondDecisionOutput)
                            statusOfDecision[firstRoute + 1].sumOfPhiX:                             add(secondDecisionInput * secondDecisionOutput)
                            statusOfDecision[firstRoute + 1].sumOfPhiSqrt:                          add(secondDecisionOutput * secondDecisionOutput)
                            statusOfDecision[firstRoute + 1].sumOfX[((secondRoute + 1) % 2 + 1)]:   add(secondDecisionInput)

                            -- estimate df/dW
                            local df_do = criterion:backward(thirdLayerOutput, targets[i])
                            TreeModels[secondRoute]:backward(inputs[i], df_do)

                            -- update confusion
                            confusion:add(thirdLayerOutput, targets[i])
                        end

                        for i=1,#gradParameters do
                            if arrivedCount[i] <= 0 then
                                arrivedCount[i] = 1
                            end
                        end
                        -- decision tree Nnode
                        DecisionNodesBackward(statusOfDecision,arrivedCount,gradParameters)

                        -- normalize gradients and f(X)
                        --print(arrivedCount)
                        for i=1,#gradParameters do
                            gradParameters[i]:div(arrivedCount[i])
                        end
                        f = f/#inputs

                        -- return f and df/dX
                        return f,gradParameters
                    end

        -- optimize on current mini-batch
        _,fs = optimMethod(feval, parameters, optimState)
        
        current_loss = current_loss + fs[1]
    end
    current_loss = current_loss / trsize
    print("\n==> loss per sample = " .. (current_loss))

    -- time taken
    time = sys.clock() - time
    time = time / trsize
    print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    -- update logger/plot
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    current_confusion_totalValid = confusion.totalValid * 100
    if liveplot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
    end

    -- save/log current net
    local filename = 'results'
    --os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)

    torch.save("results/modelNode11.net",modelNode[1][1])
    torch.save("results/modelNode21.net",modelNode[2][1])
    torch.save("results/modelNode22.net",modelNode[2][2])
    torch.save("results/modelNode31.net",modelNode[3][1])
    torch.save("results/modelNode32.net",modelNode[3][2])
    torch.save("results/modelNode33.net",modelNode[3][3])
    torch.save("results/modelNode34.net",modelNode[3][4])

    torch.save("results/decisionTreeNode11.net",decisionTreeNode[1][1])
    torch.save("results/decisionTreeNode21.net",decisionTreeNode[2][1])
    torch.save("results/decisionTreeNode22.net",decisionTreeNode[2][2])
    
    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end
