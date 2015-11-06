require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'gnuplot' -- display a image
--gmagent = require 'graphicsmagick'

----------------------------------------------------------------------

--see if the file exists
function file_exists(file)
	local f = io.open(file, "rb")
	if f then 
		f:close() 
	end
	return f ~= nil
end
function read_file (file)
	if not file_exists(file) then 
		return {} 
	end
	lines = {}
	for line in io.lines(file) do
		lines[#lines + 1] = line
	end
	return lines
end


height = 32
width = 32
-- read train data. iterate train.txt
train_txt = read_file("../data/train.txt")
train_data = {}
for i = 1, #train_txt do
	local train_image = torch.Tensor(width, height) 
	local train_labels = torch.Tensor(2) -- yaw and pitch
	local imageread = image.load("../data/" .. train_txt[i])
	print(imageread:max())
	train_image = imageread:mul(2):mul(1/255):add(-1)
	local res = {}
	--s = "1.3 3.5 xingyuan/sfalskdf/45.jpg"
	for v in string.gmatch(train_txt, "[^%s]+") do
		res[#res + 1] = v
	end
	train_labels[1] = res[1] -- yaw
	train_labels[2] = res[2] -- pitch
	
	local train_data_temp = {
		--data = train_image:float(),
		--labels = train_labels:float()
		data = train_image:cuda(),
		labels = train_labels:cuda()
	}
	train_data[#train_data + 1] = train_data_temp
	if(i % 100 == 0) then
		print("train data: " .. i)
	end
end

-- read test data. iterate test.txt
test_txt = read_file("../data/test.txt")
test_data = {}
for i = 1, #test_txt do
	local test_image = torch.Tensor(width, height) 
	local test_labels = torch.Tensor(2) -- yaw and pitch
	local imageread = image.load("../data/" .. test_txt[i])
	test_image = imageread:mul(2):mul(1/255):add(-1)
	local res = {}
	--s = "1.3 3.5 xingyuan/sfalskdf/45.jpg"
	for v in string.gmatch(test_txt, "[^%s]+") do
		res[#res + 1] = v
	end
	test_labels[1] = res[1] -- yaw
	test_labels[2] = res[2] -- pitch
	
	local test_data_temp = {
		--data = test_image:float(),
		--labels = test_labels:float()
		data = test_image:cuda(),
		labels = test_labels:cuda()
	}
	test_data[#test_data + 1] = test_data_temp
	if(i % 100 == 0) then
		print("test data: " .. i)
	end
end

trsize = #train_txt
tesize = #test_lines










