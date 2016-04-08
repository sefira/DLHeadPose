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
enableCuda = true --*************************
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
lossOfDecisionNodes = {{10},{10},{10}}
old_decisionloss = 100
current_decisionloss = 0

-- target to optimization
loss_target = 0.01
loss_difference_target = 0.0001
confusion_totalValid_target = 95
decisionloss_diff_target = 0.0001

if trainModel then
    -- optimization
    epoch = 1
    for i = 1, 100 do
    --while true do
        train()
        print(current_confusion_totalValid)
        print(old_loss)
        print(current_loss)
        print("loss_difference: "..torch.abs(old_loss - current_loss))
        current_decisionloss = lossOfDecisionNodes[1][#lossOfDecisionNodes[1]] + lossOfDecisionNodes[2][#lossOfDecisionNodes[2]] + lossOfDecisionNodes[3][#lossOfDecisionNodes[3]]
        print(old_decisionloss)
        print(current_decisionloss)
        print("decisionloss_difference: "..torch.abs(old_decisionloss - current_decisionloss))
        if (i % 100 == 0) then
            --testInTrainData()
            testInTestData()
        end

        --if (current_loss < loss_target) and (torch.abs(old_loss - current_loss) < loss_difference_target) and
        -- (current_confusion_totalValid > 95) and (torch.abs(old_decisionloss - current_decisionloss) < decisionloss_diff_target) then 
        if (torch.abs(old_loss - current_loss) < loss_difference_target) and (current_confusion_totalValid > confusion_totalValid_target) and
         (torch.abs(old_decisionloss - current_decisionloss) < decisionloss_diff_target) then 
            testInTestData()
            print("############## final test ######################")
            break
        end
        old_loss = current_loss
        old_decisionloss = current_decisionloss
    end
end
