require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

dofile 'readImage.lua'
dofile 'cnnModel.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

for i = 1, 300 do
--while true do
	train()
	testInTrainData()
	testInTestData()
	if (i % 50 == 0) then
		print("write the model weight to txt for C++ loader")
		dofile 'writeModel.lua'
	end 
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


