----------------------------------------
-------------pure Lua-------------------
----------------------------------------
function writeModel(epoch)
	-- open or create a file then return the handle
	file = io.open("xbu-weights_" .. epoch, "w")

	writeCNN(1)
	writePooling(6)
	writeCNN(5)
	writePooling(16)
	writeCNN(9)
	writeLinear(12)

	-- colse the handle
	file:close()
end

-- write CNN weight and bias
function writeCNN(layer_num) 
	local count = 0
	local weights = model:get(layer_num).weight
	local weight_str = ''	
	for i = 1,(#weights)[1] do
		for l = 1, (#weights)[2] do
			for j = 1, (#weights)[3] do
				for k = 1, (#weights)[4] do
					weight_str = ''
					count = count + 1
					weight_str = weights[i][l][j][k] .. ' '
					file:write(weight_str)
				end
			end
		end
	end
	print("in ".."cnn "..layer_num.." write "..count.." weights")
	
	local count = 0
	local biases = model:get(layer_num).bias	
	local bias_str = '';
	for i = 1,(#biases)[1] do
		count = count + 1
		bias_str = bias_str .. biases[i] .. ' '
	end
	print("in ".."cnn "..layer_num.." write "..count.." biases")
	file:write(bias_str)
end

-- write pooling layer weight and bias
function writePooling(output_num)
	local count = 0
	local weight_str = ''
	for i = 1,output_num do
		count = count + 1
		weight_str = weight_str .. 1 .. ' '
	end 
	print("in ".."pooling".." write "..count.." weights")
	file:write(weight_str)
	
	local count = 0
	local bias_str = ''
	for i = 1,output_num do
		count = count + 1
		bias_str = bias_str .. 0 .. ' '
	end
	print("in ".."pooling".." write "..count.." biases")
	file:write(bias_str)
end

--write Linear layer weight and bias
function writeLinear(layer_num)
	local count = 0
	local weights = model:get(layer_num).weight
	local weight_str = ''
	for i = 1,(#weights)[2] do 
		for j = 1,(#weights)[1] do
			count = count + 1
			weight_str = weight_str .. weights[j][i] .. ' '
		end
	end
	print("in ".."linear "..layer_num.." write "..count.." weights")
	file:write(weight_str)
	
	local count = 0
	local biases = model:get(layer_num).bias
	local bias_str = ''
	for i = 1,(#weights)[1] do 
		count = count + 1
		bias_str = bias_str .. biases[i] .. ' '
	end
	print("in ".."linear "..layer_num.." write "..count.." biases")
	file:write(bias_str)
end