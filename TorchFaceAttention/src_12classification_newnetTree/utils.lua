function equal(a,b)
   res = torch.eq(a,b)
   minV = torch.min(res)
   if minV == 1 then
      return "EQUAL"
   else
      return "NOT EQUAL"
   end
end

_flattenTensorBuffer = {}
function flattenParameters(parameters)

   -- returns true if tensor occupies a contiguous region of memory (no holes)
   local function isCompact(tensor)
      local sortedStride, perm = torch.sort(
            torch.LongTensor(tensor:nDimension()):set(tensor:stride()), 1, true)
      local sortedSize = torch.LongTensor(tensor:nDimension()):set(
            tensor:size()):index(1, perm)
      local nRealDim = torch.clamp(sortedStride, 0, 1):sum()
      sortedStride = sortedStride:narrow(1, 1, nRealDim):clone()
      sortedSize   = sortedSize:narrow(1, 1, nRealDim):clone()
      local t = tensor.new():set(tensor:storage(), 1,
                                 sortedSize:storage(),
                                 sortedStride:storage())
      return t:isContiguous()
   end

   if not parameters or #parameters == 0 then
      return torch.Tensor()
   end
   local Tensor = parameters[1].new
   local TmpTensor = _flattenTensorBuffer[torch.type(parameters[1])] or Tensor

   -- 1. construct the set of all unique storages referenced by parameter tensors
   local storages = {}
   local nParameters = 0
   local parameterMeta = {}
   for k = 1,#parameters do
      local param = parameters[k]
      local storage = parameters[k]:storage()
      local storageKey = torch.pointer(storage)

      if not storages[storageKey] then
         storages[storageKey] = {storage, nParameters}
         nParameters = nParameters + storage:size()
      end

      parameterMeta[k] = {storageOffset = param:storageOffset() +
                                          storages[storageKey][2],
                          size          = param:size(),
                          stride        = param:stride()}
   end

   -- 2. construct a single tensor that will hold all the parameters
   local flatParameters = TmpTensor(nParameters):zero()

   -- 3. determine if there are elements in the storage that none of the
   --    parameter tensors reference ('holes')
   local tensorsCompact = true
   for k = 1,#parameters do
      local meta = parameterMeta[k]
      local tmp = TmpTensor():set(
         flatParameters:storage(), meta.storageOffset, meta.size, meta.stride)
      tmp:fill(1)
      tensorsCompact = tensorsCompact and isCompact(tmp)
   end

   local maskParameters  = flatParameters:byte():clone()
   local compactOffsets  = flatParameters:long():cumsum(1)
   local nUsedParameters = compactOffsets[-1]

   -- 4. copy storages into the flattened parameter tensor
   for _, storageAndOffset in pairs(storages) do
      local storage, offset = table.unpack(storageAndOffset)
      flatParameters[{{offset+1,offset+storage:size()}}]:copy(Tensor():set(storage))
   end

   -- 5. allow garbage collection
   storages = nil
   for k = 1,#parameters do
       parameters[k]:set(Tensor())
   end

   -- 6. compact the flattened parameters if there were holes
   if nUsedParameters ~= nParameters then
      assert(tensorsCompact,
         "Cannot gather tensors that are not compact")

      flatParameters = TmpTensor(nUsedParameters):copy(
            flatParameters:maskedSelect(maskParameters))
      for k = 1,#parameters do
        parameterMeta[k].storageOffset =
              compactOffsets[parameterMeta[k].storageOffset]
      end
   end

   if TmpTensor ~= Tensor then
      flatParameters = Tensor(flatParameters:nElement()):copy(flatParameters)
   end

   -- 7. fix up the parameter tensors to point at the flattened parameters
   for k = 1,#parameters do
      parameters[k]:set(flatParameters:storage(),
          parameterMeta[k].storageOffset,
          parameterMeta[k].size,
          parameterMeta[k].stride)
   end

   return flatParameters
end


--[[ A plain implementation of SGD
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
(Clement Farabet, 2012)
]]
function multiModelSgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) weight decay with single or individual parameters
   if wd ~= 0 then
      print("here!!!!!!!!!!!!!!!!wd!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      dfdx:add(wd, x)
   elseif wds then
      print("there!!!!!!!!!!!!!!!wds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      print("here!!!!!!!!!!!!!!!!!mom!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   --print(nevals)
   -- (5) parameter update with single or individual learning rates
   if lrs then
      print("here!!!!!!!!!!!!!!!!!lrs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      for i=1,#x do
         x[i]:add(-clr, dfdx[i])
      end
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end

