require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> executing all'

dofile 'readImage.lua'
dofile 'cnnModel.lua'
dofile 'loss.lua'
dofile 'train.lua'

-- dofile 'test.lua'
while true do
   train()
   test()
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


