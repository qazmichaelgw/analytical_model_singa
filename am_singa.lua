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
c_muladd = 1
c_act = 2

--Layer Time for different layer
conf = {}
time = {}
time[kRecordInput] = function(param) print ('Input ') return 1 end
conf[kRecordInput] = 'store_conf'
time[kCConvolution] = function(param) print ('Conv f:'..param.num_filters..' k:'..param.kernel..' s:'..param.stride) return 1 end
conf[kCConvolution] = 'convolution_conf'
time[kCPooling] = function(param) pool = param.pool or 'MAX' print ('Pool p:'..pool..' k:'..param.kernel..' s:'..param.stride) return 1 end 
conf[kCPooling] = 'pooling_conf'
time[kInnerProduct] = function(param) print ('InnerProduct '..param.num_output) return 1 end
conf[kInnerProduct] = 'innerproduct_conf'
time[kSoftmaxLoss] = function(param) print ('SoftMax ') return 1 end
conf[kSoftmaxLoss] = 'softmaxloss_conf'
time[kReLU] = function() print 'ReLU' return 1 end
print (time)

function parse(fn)
    local file = io.open(fn)
    if file then
        local layers_str = {}
        -- used to address the table with the same index
        local i = 0
        local j = 0
        for line in file:lines() do
            line = line:gsub("%s+", "")
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

function layer_time(layer, net)
    local l_net = net[layer]
    local param = l_net[conf[l_net.type]]
    return time[l_net.type](param) 
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
    while layer do
        time = time + layer_time(layer, net)
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
local net = parse('cudnn.conf')
--print (net)
local t = compute_time(net)
print (t)
