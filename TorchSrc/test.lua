require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function testInTrainData()
	-- local vars
	local time = sys.clock()

	-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
	model:evaluate()
	local loss = 0
	
	-- test over test data
	print('==> testing on train set:')
	for t = 1,trsize do
		-- disp progress
		xlua.progress(t, trsize)

		-- get new sample
		local input = train_data[t].data
		local target = train_data[t].labels

		-- test sample
		local pred = model:forward(input)
		loss = loss + torch.abs(target - pred)
	end
	print("\n==> loss per sample = " .. (loss / trsize) .. 'ms')

	-- timing
	time = sys.clock() - time
	time = time / trsize
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

end

function testInTestData()
	-- local vars
	local time = sys.clock()

	-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
	model:evaluate()
	local loss = 0

	-- test over test data
	print('==> testing on test set:')
	for t = 1,tesize do
		-- disp progress
		xlua.progress(t, tesize)

		-- get new sample
		local input = test_data[t].data
		local target = test_data[t].labels

		-- test sample
		local pred = model:forward(input)
		loss = loss + torch.abs(target - pred)
	end
	print("\n==> loss per sample = " .. (loss / tesize) .. 'ms')

	-- timing
	time = sys.clock() - time
	time = time / tesize
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

end
