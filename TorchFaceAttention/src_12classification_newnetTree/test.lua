require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function testInTrainData()
    -- local vars
    local time = sys.clock()

    -- set TreeModels to evaluate mode (for modules that differ in training and testing, like Dropout)  
    for i = 1,4 do 
        TreeModels[i]:evaluate()
    end
    -- test over test data
    print('\n==> testing on train set:')
    for t = 1,trsize do
        -- disp progress
        xlua.progress(t, trsize)

        -- get new sample
        local input = train_data[t].data
        local target = train_data[t].labels

        -- test sample
        local pred,_,_,_,_,_,_ = TreeModelForward(input)
        confusion:add(pred, target)
    end

    -- timing
    time = sys.clock() - time
    time = time / trsize
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    -- update log/plot
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    if liveplot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
    end

    -- next iteration:
    confusion:zero()
end

function testInTestData()
    -- local vars
    local time = sys.clock()

    -- set TreeModels to evaluate mode (for modules that differ in training and testing, like Dropout)
    for i = 1,4 do 
        TreeModels[i]:evaluate()
    end

    -- test over test data
    print('\n==> testing on test set:')
    for t = 1,tesize do
        -- disp progress
        xlua.progress(t, tesize)

        -- get new sample
        local input = test_data[t].data
        local target = test_data[t].labels

        -- test sample
        local pred,_,_,_,_,_,_ = TreeModelForward(input)
        confusion:add(pred, target)
    end
    
    -- timing
    time = sys.clock() - time
    time = time / tesize
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    -- update log/plot
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    if liveplot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
    end

    -- next iteration:
    confusion:zero()
end
