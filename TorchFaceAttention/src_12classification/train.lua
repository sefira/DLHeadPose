require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat('results', 'train.log'))
testLogger = optim.Logger(paths.concat('results', 'test.log'))

----------------------------------------------------------------------
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
	learningRate = 1e-3,
	weightDecay = 0,
	momentum = 0,
	learningRateDecay = 1e-7
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'
batchSize = 1000
function train()

	-- epoch tracker
	epoch = epoch or 1

	-- local vars
	local time = sys.clock()

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	model:training()

	-- shuffle at each epoch
	shuffle = torch.randperm(trsize)

	-- do one epoch
	current_loss = 0
	print('\n\n#######################################################')
	print('#######################################################')
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
	for t = 1,trsize,batchSize do
		-- disp progress
		xlua.progress(t, trsize)

		-- create mini batch
		local inputs = {}
		local targets = {}
		for i = t,math.min(t+batchSize-1,trsize) do
			-- load new sample
			local input = train_data[shuffle[i]].data
			local target = train_data[shuffle[i]].labels
			table.insert(inputs, input)
			table.insert(targets, target)
		end
		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
						-- get new parameters
						if x ~= parameters then
							parameters:copy(x)
						end

						-- reset gradients
						gradParameters:zero()

						-- f is the average of all criterions
						local f = 0

						-- evaluate function for complete mini batch
						for i = 1,#inputs do
							-- estimate f
							local output = model:forward(inputs[i])
							local err = criterion:forward(output, targets[i])
							if (targets[i] == 1) then
								f = f + (err * 5)
							else
								f = f + err
							end
							-- estimate df/dW
							local df_do = criterion:backward(output, targets[i])
							model:backward(inputs[i], df_do)

                          	-- update confusion
                          	confusion:add(output, targets[i])
						end

						-- normalize gradients and f(X)
						gradParameters:div(#inputs)
						f = f/#inputs

						-- return f and df/dX
						return f,gradParameters
					end

		-- optimize on current mini-batch
		_,fs = optimMethod(feval, parameters, optimState)
		
		current_loss = current_loss + fs[1]
	end
	current_loss = current_loss / trsize
	print("\n==> loss per sample = " .. (current_loss))

	-- time taken
	time = sys.clock() - time
	time = time / trsize
	print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)

	-- update logger/plot
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	current_confusion_totalValid = confusion.totalValid * 100
	if liveplot then
	  trainLogger:style{['% mean class accuracy (train set)'] = '-'}
	  trainLogger:plot()
	end

	-- save/log current net
	local filename = paths.concat('results', 'model.net')
	--os.execute('mkdir -p ' .. sys.dirname(filename))
	print('==> saving model to '..filename)
	torch.save(filename, model)

   	-- next epoch
   	confusion:zero()
	epoch = epoch + 1
end
