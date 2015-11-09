require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

dofile 'readImage.lua'
dofile 'cnnModel.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'
dofile 'writeModel.lua'


old_loss = 1000
for i = 1, 1000 do
--while true do
	train()
	testInTrainData()
	testInTestData()
	if (i % 50 == 0) then
		print("write the model weight to txt for C++ loader")
		writeModel(i)
	end
	if (torch.abs(old_loss - current_loss) < 0.00001) then 
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


