--------------------------------
--TODO transforms tensor to cuda
--------------------------------

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
	local train_labels = torch.Tensor(2) -- pitch and yaw
	local res = {}
	--s = "up_down/image0/1_1.jpg -45 0"
	for v in string.gmatch(train_txt[i], "[^%s]+") do
		res[#res + 1] = v
	end
	filename = res[1]
	train_labels[1] = res[2] -- pitch
	train_labels[2] = res[3] -- yaw
	-- here need to mul(255) due to torch will auto mul(1/255) for a jpg
	local imageread = image.load("../data/" .. filename):mul(255)
	--print(imageread:max())
	local train_image = imageread:mul(2):mul(1/255):add(-1)
	
	local train_data_temp = {
		data = train_image:double(),
		labels = train_labels:double()
		--data = train_image:cuda(),
		--labels = train_labels:cuda()
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
	local test_labels = torch.Tensor(2) -- pitch and yaw
	local res = {}
	--s = "up_down/image0/1_1.jpg -45 0"
	for v in string.gmatch(test_txt[i], "[^%s]+") do
		res[#res + 1] = v
	end
	filename = res[1]
	test_labels[1] = res[2] -- pitch
	test_labels[2] = res[3] -- yaw
	-- here need to mul(255) due to torch will auto mul(1/255) for a jpg
	local imageread = image.load("../data/" .. filename):mul(255)
	--print(imageread:max())
	local test_image = imageread:mul(2):mul(1/255):add(-1)
	
	local test_data_temp = {
		data = test_image:double(),
		labels = test_labels:double()
		--data = test_image:cuda(),
		--labels = test_labels:cuda()
	}
	test_data[#test_data + 1] = test_data_temp
	if(i % 100 == 0) then
		print("test data: " .. i)
	end
end

trsize = #train_txt
tesize = #test_txt











