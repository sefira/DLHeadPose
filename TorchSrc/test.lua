require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
	-- local vars
	local time = sys.clock()

	-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
	model:evaluate()

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
		print(string.format("%6.2f %6.2f", i, pred, target))
	end

	-- timing
	time = sys.clock() - time
	time = time / tesize
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

end
