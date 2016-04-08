--------------------------------
--TODO transforms tensor to cuda
--------------------------------
require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

-------------------configuration------------------
liveplot = false
ClassNLL = true -- use classNLL or KL
enableCuda = false --*************************
loadModel = false -- load model node from saved nodefile
inheritModel = false -- inherit model node from parent model that a CNN model trained without tree
trainModel = true -- determine the model whether need to be trained

if enableCuda then
    print "CUDA enable"
    require 'cunn'
    require 'cutorch'
end
-------------------configuration------------------
dofile 'utils.lua'
dofile 'readImage.lua'
dofile 'cnnModel.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
current_confusion_totalValid = 0
old_loss = 1000
current_loss = 0
-- target to optimization
loss_target = 0.01
loss_difference_target = 0.0001
confusion_totalValid_target = 95

if trainModel then
    -- optimization
    epoch = 1
    for i = 1, 1000 do
    --while true do
        train()
        print(old_loss)
        print(current_loss)
        print(current_confusion_totalValid)
        print(torch.abs(old_loss - current_loss))
        if (i % 100 == 0) then
            --testInTrainData()
            testInTestData()
        end

        --if (current_loss < loss_target) and (torch.abs(old_loss - current_loss) < loss_difference_target) and (current_confusion_totalValid > 95) then 
        if (torch.abs(old_loss - current_loss) < loss_difference_target) and (current_confusion_totalValid > confusion_totalValid_target) then 
            testInTestData()
            print("############## final test ######################")
            break
        end
        old_loss = current_loss
    end
end
