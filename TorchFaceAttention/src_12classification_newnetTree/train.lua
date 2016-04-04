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
if TreeModels then
    parameters[1],gradParameters[1] = TreeModels[1]:getParameters()
    parameters[2],gradParameters[2] = TreeModels[2]:getParameters()
    parameters[3],gradParameters[3] = TreeModels[3]:getParameters()
    parameters[4],gradParameters[4] = TreeModels[4]:getParameters()
    --parameters[5],gradParameters[5] = decisionTreeNode[1][1]:getParameters()
    --parameters[6],gradParameters[6] = decisionTreeNode[2][1]:getParameters()
    --parameters[7],gradParameters[7] = decisionTreeNode[2][2]:getParameters()
end

parametersList = flattenParameters(parameters)
gradParametersList = flattenParameters(gradParameters)

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
    learningRate = 1e-3,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
}
optimMethod = optim.sgd

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
                        -- get new parametersList
                        if x ~= parametersList then
                            parametersList:copy(x)
                        end

                        -- reset gradients
                        gradParametersList:zero()

                        -- f is the average of all criterions, actually won't be used in sgd.lua
                        local f = 0

                        local arrivedCount = {}
                        for i=1,4 do
                            arrivedCount[i] = 0
                        end
                        -- evaluate function for complete mini batch
                        for i = 1,#inputs do
                            -- estimate f
                            local thirdLayerOutput,firstDecisionOutput,secondDecisionOutput,firstRoute,secondRoute = TreeModelForward(inputs[i])
                            local err = criterion:forward(thirdLayerOutput, targets[i])
                            f = f + err
                            arrivedCount[secondRoute] = arrivedCount[secondRoute] + 1
                            -- estimate df/dW
                            local df_do = criterion:backward(thirdLayerOutput, targets[i])
                            TreeModels[secondRoute]:backward(inputs[i], df_do)

                            -- update confusion
                            confusion:add(thirdLayerOutput, targets[i])
                        end

                        -- normalize gradients and f(X)
                        gradParametersList:div(#inputs)
                        f = f/#inputs

                        -- return f and df/dX
                        return f,gradParametersList
                    end

        -- optimize on current mini-batch
        _,fs = optimMethod(feval, parametersList, optimState)
        
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
