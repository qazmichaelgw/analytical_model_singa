--Train
kTrain = 'kTrain'
--LayerType 
kCSVInput = 100
kImagePreprocess = 101
kRecordInput = 103
kLMDBData = 190
kLabel = 191
kMnist = 192
kRGBImage = 193
kShardData = 194
kCharRNN = 195
kRNNLabel = 196
kOneHot = 197
kConvolution = 201
kCConvolution = 202
kDropout = 203
kDummy = 204
kInnerProduct = 205
kLRN = 206
kPooling = 207
kCPooling = 208
kRBMHid = 209
kRBMVis = 210
kReLU = 211
kSTanh = 212
kSigmoid = 213
kSoftmax = 214
kGRU = 215
kEmbedding = 216
kActivation = 217
kBM = 218
kCudnnConv = 250
kCudnnPool = 251
kCudnnLRN = 252
kCudnnSoftmax = 253
kCudnnActivation = 254
kCudnnBM = 255
kEuclideanLoss = 300
kSoftmaxLoss = 301
kCudnnSoftmaxLoss = 350
kAccuracy = 400
kArgSort = 401
kCSVOutput = 402
kRecordOutput = 403
kCharRNNOutput = 404
kBridgeDst = 500
kBridgeSrc = 501
kConcate = 502
kSlice = 503
kSplit = 504
kRNNDummy = 505
kUserLayer = 600
--Activation Type
RELU = 1
SIGMOID = 2
TANH = 3
STANH = 4

--Define Const Time Estimator
c_add = 1.51112e-11
c_mul = 4.55988e-11 
c_comp = 3.1234e-11
c_muladd = c_add + c_mul
c_exp = 7.12344e-10
c_act = 9.12e-10
c_err = c_add + 3.142e-10
--batch_size
batch_size = 1

local function forward(w, neuron, layer, extra_param)
    if layer and layer:find('conv') then
        return batch_size*((c_muladd*w + c_add)*neuron/extra_param + neuron*c_act)
    elseif layer and layer:find('pool') then
        print (extra_param)
        return batch_size*((c_muladd*w + c_comp*extra_param + c_add)*neuron + neuron*c_act)
    elseif layer and layer:find('loss') then
        --for linear parameters, we can see the neuron as the linear weights
        --refer http://cs231n.github.io/neural-networks-case-study/
        return batch_size*(c_muladd*w + c_addmul*neuron+c_exp*neuron+c_mul*neuron)
    else
        return batch_size*(c_muladd*w + c_add + neuron*c_act)
    end
end
local function backward(w, neuron, layer, extra)
    if layer and layer:find('conv') then
        print ('conv', w)
        return batch_size*(c_err*w*neuron/extra)
    elseif layer and layer:find('pool') then
        print (extra_param)
        return batch_size*(c_err*neuron)
    elseif layer and layer:find('loss') then
        --for linear parameters, we can see the neuron as the linear weights
        --refer http://cs231n.github.io/neural-networks-case-study/
        return batch_size*(c_err*neuron+c_exp*neuron+c_mul*neuron)
    else
        return batch_size*(c_err*neuron)
    end
end
local function update(w)
    return batch_size*(c_muladd*w)
end
local function back(w, neuron, layer, extra)
    return backward(w, neuron, layer, extra) + update(w)
end

--Layer Time for different layer
monitor_layer_time = {}
monitor_back_layer_time = {}
net_param = {}
conf = {}
time = {}
time[kRecordInput] = function(input, param, layer) 
    print ('Input ') 
    local shape={'shape0', 'shape1', 'shape2'}
    local output={'d', 'w', 'h'}
    local o = {}
    for k, v in pairs(shape) do
        o[output[k]] = param[v]
    end
    --batch_size=param.batchsize
    batch_size= 64
    return 0, o  
end
conf[kRecordInput] = 'store_conf'
time[kCConvolution] = function(input, param, layer) 
    local stride = param.stride or 1
    local pad = param.pad or 0
    local w = (input.w-param.kernel+2*pad)/stride + 1
    local h = (input.h-param.kernel+2*pad)/stride + 1 
    local o = {w=w, h=h, d=param.num_filters} 
    print ('Conv f:'..param.num_filters..' k:'..param.kernel..' s:'..stride..' p:'..pad) 
    local weights = param.kernel*param.kernel*input.d*param.num_filters
    local neuron = o.w*o.h*o.d
    net_param[layer] = {w=weights, n=neuron, extra=o.d*input.d}
    return forward(weights, neuron, 'conv', o.d), o 
end
conf[kCConvolution] = 'convolution_conf'
time[kCPooling] = function(input, param, layer) 
    pool = param.pool or 'MAX' 
    print ('Pool p:'..pool..' k:'..param.kernel..' s:'..param.stride) 
    local w = (input.w-param.kernel)/param.stride + 1
    local h = (input.h-param.kernel)/param.stride + 1 
    local o = {w=w, h=h, d=input.d} 
    local weights = 0
    local neuron = o.w*o.h*o.d
    net_param[layer] = {w=weights, n=neuron}
    return forward(weights, neuron, 'pool', param.kernel*param.kernel*o.d), o 
end 
conf[kCPooling] = 'pooling_conf'
time[kInnerProduct] = function(input, param, layer) 
    print ('InnerProduct '..param.num_output) 
    local o = {n=param.num_output}
    local i = 1
    for _, v in pairs(input) do
        i = i * v 
    end
    local weights = i*param.num_output
    local neuron = param.num_output
    net_param[layer] = {w=weights, n=neuron}
    return forward(weights, neuron), o 
end
conf[kInnerProduct] = 'innerproduct_conf'
time[kSoftmaxLoss] = function(input, param, layer) 
    print ('SoftMax ', input) 
    local neuron = 1
    for _, v in pairs(input) do
        neuron = neuron * v 
    end
    local o = input
    local weights = input.n*neuron 
    net_param[layer] = {w=weights, n=neuron}
    return forward(weights, neuron), o 
end
conf[kSoftmaxLoss] = 'softmaxloss_conf'
time[kReLU] = function(input, param, layer) 
    print 'ReLU' 
    local o = input
    local weights = 0
    local neuron = 1
    for _, v in pairs(input) do
        neuron = neuron * v 
    end
    net_param[layer] = {w=weights, n=neuron}
    return forward(weights, neuron), o 
end

function string.starts(String,Start)
    return string.sub(String,1,string.len(Start))==Start
end

function string.ends(String,End)
    return End=='' or string.sub(String,-string.len(End))==End
end

function parse(fn)
    local file = io.open(fn)
    if file then
        local layers_str = {}
        -- used to address the table with the same index
        local i = 0
        local j = 0
        for line in file:lines() do
            line = line:gsub("%s+", "")
            if not string.starts(line, "#") and line:len() > 0 then
                if line:find('srclayers') then
                    line = line:gsub('srclayers', 'srclayers'..string.format('%04d', j))
                    j = j + 1
                end
                if line:find('{') then
                    if not line:find('neuralnet') and (not line:find('conf')) then
                        line = line:gsub('{', string.format('%04d', i)..'{')
                    end
                    i = i + 1
                    table.insert(layers_str, line)
                else
                    table.insert(layers_str, line..',')
                end
            end
        end
        local str = table.concat(layers_str, ' ')
        local table_str = 'net={'..string.gsub(string.gsub(str, '{', '={'), ':', '=')..'}'
        loadstring(table_str)()
        --print (net)
        --remapping by the name of the layer
        local nnet = {}
        net = net['neuralnet']
        for k, v in pairs(net) do
            nnet[v.name] = v
        end
        return nnet
    end
end

function layer_time(input, layer, net)
    print ('#####################################')
    print ('layer_time', layer, input)
    local l_net = net[layer]
    local param = l_net[conf[l_net.type]]
    local t, o = time[l_net.type](input, param, layer) 
    monitor_layer_time[layer] = t
    return t, o 
end

function compute_time(net)
    local time = 0
    --construct graph structure
    local path = {}
    local pre_path = {}
    for k, v in pairs(net) do
        for k_src, v_src in pairs(v) do
            if k_src:find('srclayers') then
                if pre_path[k] then
                    table.insert(pre_path[k], v[k_src])
                else
                    pre_path[k] = {v[k_src]}
                end
            end
        end
    end
    -- mute data layer to loss
    for k, v in pairs(pre_path) do
        if k:find('loss') then
            for k1, v1 in pairs(v) do
                if v1:find('data') then
                    v[k1] = nil
                end
            end
        end
    end

    for k, v in pairs(pre_path) do
        for k1, v1 in pairs(v) do
            if path[v1] then
                table.insert(path[v1], k)
            else
                path[v1] = {k}
            end
        end
    end
    --print (pre_path)
    --print (path)

    --data layer
    local queue = {}
    local head = 1
    local tail = 1
    local function push(e)
        queue[tail] = e
        tail = tail + 1
    end
    local function pop()
        if head == tail then return nil end
        local e = queue[head]
        head = head + 1
        return e
    end
    local layer = 'data'
    local input = {}
    while layer do
        local t, o = layer_time(input[layer], layer, net)
        -- assert sequence model, only support sequencial model now
        time = time + t 
        if path[layer] then
            input[path[layer][1]] = o
            for _, v in pairs(path[layer]) do
                push(v)
            end
        end
        layer = pop()
    end

    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print ('forward time', time)
    --backward
    layer = path['data'][1]
    while layer do
        local weights = net_param[layer].w
        local neuron = net_param[layer].n
        local extra = net_param[layer].extra
        local t = back(weights, neuron, layer, extra)
        monitor_back_layer_time[layer] = t
        time = time + t 
        if path[layer] then
            for _, v in pairs(path[layer]) do
                push(v)
            end
        end
        layer = pop()
    end

    return time
end

--local net = parse('lt-singa-worker1')
local net = parse('conv_mnist.conf')
--local net = parse('cudnn_cifar10.conf')
--print (net)
local t = compute_time(net)
print ('batch size', batch_size)
print (net_param)
print (monitor_layer_time)
print (monitor_back_layer_time)
print ('time', t)
