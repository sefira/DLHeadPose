require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

-- multi-class problem
function tablelength(T)
	local count = 0
	for _ in pairs(T) do count = count +  1 end
	return count
end
noutputs = 6400
ninputs = nfeats*width*height


----------------------------------------------------------------------
print '==> construct model'

--LayerOne NN model in function type:
function defineOne()
	local oneDim = nfeats
	local oneNFilter = 64
	local oneFilterKN = 9
	local oneGaussKN = 9
	local oneGauss = image.gaussian1D(oneGaussKN)
	local onePoolingKN = 10
	local onePoolingDN = 1
	local oneSubsamKN = 1
	local oneSubsamDN = 5
	--convolution kernel 9*9 , step 1
	--3D or 4D(batch mode) tensor expected
	--------------inputImage:double()
	oneConvolutionLayer = nn.SpatialConvolution(oneDim, oneDim*oneNFilter, oneFilterKN, oneFilterKN)
	oneConvolutionLayer.weight:uniform(-0.11,0.11)
	--abs
	oneAbsLayer = nn.Abs()
	--Contrastive Normalization 
	oneContrastiveNormalizationLayer = nn.SpatialContrastiveNormalization(oneDim*oneNFilter, oneGauss):double()
	--oneContrastiveNormalizationLayer = nn.SpatialSubtractiveNormalization(oneDim*oneNFilter, oneGauss):double()
	--oneContrastiveNormalizationLayer = nn.SpatialDivisiveNormalization(oneDim*oneNFilter, oneGauss):double()
	--Average Pooling kernel 10*10 , step 1
	oneAveragePoolingLayer = nn.SpatialAveragePooling(onePoolingKN, onePoolingKN, onePoolingDN, onePoolingDN)
	--oneAveragePoolingLayer = nn.SpatialMaxPooling(onePoolingKN, onePoolingKN, onePoolingDN, onePoolingDN)
	--SubSampling kernel 1*1 , step 5
	oneSubSamplingLayer = nn.SpatialSubSampling(oneNFilter,oneSubsamKN,oneSubsamKN,oneSubsamDN,oneSubsamDN)
	print(oneContrastiveNormalizationLayer)
end
defineOne()

function throughOne(inputImage)
	--print(inputImage)
	local oneCon = oneConvolutionLayer:forward(inputImage)
	local oneAbs = oneAbsLayer:forward(oneCon)
	local oneConNor = oneContrastiveNormalizationLayer:forward(oneAbs)
	local oneAve = oneAveragePoolingLayer:forward(oneConNor)
	local oneSub = oneSubSamplingLayer:forward(oneAve)
	return oneSub
end

--define firstStage output
function defineOneOutput()
	local oneOPoolingKN = 6
	local oneOPoolingDN = 1
	local oneONFilter = 64
	local oneOSubsamKN = 1
	local oneOSubsamDN = 4
	local oneOsizeOutput = 6*6*64
	--Average Pooling kernel 6*6 , step 1
	oneOAveragePoolingLayer = nn.SpatialAveragePooling(oneOPoolingKN, oneOPoolingKN, oneOPoolingDN, oneOPoolingDN)
	--oneOAveragePoolingLayer = nn.SpatialMaxPooling(oneOPoolingKN, oneOPoolingKN, oneOPoolingDN, oneOPoolingDN)
	--SubSampling kernel 1*1 , step 5
	oneOSubSamplingLayer = nn.SpatialSubSampling(oneONFilter,oneOSubsamKN,oneOSubsamKN,oneOSubsamDN,oneOSubsamDN)
	oneOReSh = nn.Reshape(oneOsizeOutput)
end
defineOneOutput()

--calculate firstStage output
function handleOneOutput(input)
	local oneOAve = oneOAveragePoolingLayer:forward(input)
	local oneOSub = oneOSubSamplingLayer:forward(oneOAve)
	local oneOSub = torch.Tensor(oneOSub)
	local output = oneOReSh:forward(oneOSub)
	return output 
end

--LayerTwo NN model in function type:
function defineTwo()
	local twoDim = 64
	local twoNFilter = 4
	local twoFilterKN = 9
	local twoGaussKN = 9
	local twoGauss = image.gaussian1D(oneGaussKN)
	local twoPoolingKN = 6
	local twoPoolingDN = 1
	local twoSubsamKN = 1
	local twoSubsamDN = 4
	--convolution kernel 9*9 , step 1
	--twoConvolutionLayer = nn.SpatialConvolution(twoDim, twoDim*twoNFilter, twoFilterKN, twoFilterKN)
	twoConvolutionLayer = nn.SpatialConvolutionMap(nn.tables.random(twoDim, twoDim*twoNFilter, 4096/twoDim/twoNFilter), twoFilterKN, twoFilterKN)
	twoConvolutionLayer.weight:uniform(-0.11,0.11)
	--abs
	twoAbsLayer = nn.Abs()
	--Contrastive Normalization 
	twoContrastiveNormalizationLayer = nn.SpatialContrastiveNormalization(twoDim*twoNFilter, twoGauss):double()
	--Average Pooling kernel 10*10 , step 1
	twoAveragePoolingLayer = nn.SpatialAveragePooling(twoPoolingKN, twoPoolingKN, twoPoolingDN, twoPoolingDN)
	--twoAveragePoolingLayer = nn.SpatialMaxPooling(twoPoolingKN, twoPoolingKN, twoPoolingDN, twoPoolingDN)
	--SubSampling kernel 1*1 , step 5
	twoSubSamplingLayer = nn.SpatialSubSampling(twoDim*twoNFilter,twoSubsamKN,twoSubsamKN,twoSubsamDN,twoSubsamDN)
end
defineTwo()

function throughTwo(inputImage)
	local twoCon = twoConvolutionLayer:forward(inputImage)
	local twoAbs = twoAbsLayer:forward(twoCon)
	local twoConNor = twoContrastiveNormalizationLayer:forward(twoAbs)
	local twoAve = twoAveragePoolingLayer:forward(twoConNor) 
	local twoSub = twoSubSamplingLayer:forward(twoAve)
	return twoSub
end

--define secondStage output
function defineTwoOutput()
	local twoOsizeOutput = 4*4*256
	twoOReSh = nn.Reshape(twoOsizeOutput)
end
defineTwoOutput()

--calculate secondStage output
function handleTwoOutput(input)
	local output = twoOReSh:forward(input)
	return output 
end

function throughNN(inputImage)
	local oneOutput = throughOne(inputImage)
	--print(#oneOutput)
	local firstStage = handleOneOutput(oneOutput)
	--print(#firstStage)
	local twoOutput = throughTwo(oneOutput)
	--print(#twoOutput)
	local secondStage = handleTwoOutput(twoOutput)
	--print(#secondStage)
	output = torch.Tensor(noutputs):zero()
	output[{{1,2304}}] = firstStage
	output[{{2305,noutputs}}] = secondStage
	return output
end



