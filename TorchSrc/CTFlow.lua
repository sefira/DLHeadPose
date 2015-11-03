require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> executing all'

dofile 'readImage.lua'
dofile 'defineNN.lua'
dofile 'regression.lua'
dofile 'throughNN.lua'

dofile 'trainData.lua'
train()
testinTrainData()

dofile 'testData.lua'
testinTestData()

function equal(a,b)
	res = torch.eq(a,b)
	minV = torch.min(res)
	if minV == 1 then
		return "EQUAL"
	else
		return "NOT EQUAL"
	end
end


