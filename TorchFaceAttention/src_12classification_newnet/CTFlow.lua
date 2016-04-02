--------------------------------
--TODO transforms tensor to cuda
--------------------------------
require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

-------------------configuration------------------
liveplot = false
enableCuda = true 
ClassNLL = true

if enableCuda then
	print "CUDA enable"
	require 'cunn'
	require 'cutorch'
end
-------------------configuration------------------

dofile 'readImage.lua'
dofile 'cnnModel.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'
dofile 'writeModel.lua'

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
		print("write the model weight to txt for C++ loader")
		--writeModel(i)
	end

	--if (current_loss < loss_target) and (torch.abs(old_loss - current_loss) < loss_difference_target) and (current_confusion_totalValid > 95) then 
	if (torch.abs(old_loss - current_loss) < loss_difference_target) and (current_confusion_totalValid > confusion_totalValid_target) then 
		testInTestData()
		print("############## final write ######################")
		print("write the model weight to txt for C++ loader")
		--writeModel(i)
		break
	end
	old_loss = current_loss
end

function equal(a,b)
	res = torch.eq(a,b)
	minV = torch.min(res)
	if minV == 1 then
		return "EQUAL"
	else
		return "NOT EQUAL"
	end
end
