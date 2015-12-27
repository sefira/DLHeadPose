require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

-------------------configuration------------------
liveplot = false
enableCuda = true

if enableCuda then
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
epoch = 1
for i = 1, 1000 do
--while true do
	train()
	testInTrainData()
	testInTestData()
	if (i % 100 == 0) then
		print("write the model weight to txt for C++ loader")
		writeModel(i)
	end
	--if (current_confusion_totalValid > 95) then  -- current_confusion_totalValid > 95%
	if (torch.abs(old_loss - current_loss) < 0.0001) then 
		print("############## final write ######################")
		print("write the model weight to txt for C++ loader")
		writeModel(i)
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
