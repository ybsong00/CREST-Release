function [ net ] = InitFeatureExtractionNet( net,net1)

net.addLayer('conv1', dagnn.Conv('size', [3,3,3,64], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'input', 'conv_1', {'conv1_f', 'conv1_b'});
net.addLayer('relu1', dagnn.ReLU(), 'conv_1', 'relu_1');

f=net.getParamIndex('conv1_f');
net.params(f).value=net1.layers{1}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv1_b');
net.params(f).value=net1.layers{1}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv2', dagnn.Conv('size', [3,3,64,64], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_1', 'conv_2', {'conv2_f', 'conv2_b'});
net.addLayer('relu2', dagnn.ReLU(), 'conv_2', 'relu_2');

f=net.getParamIndex('conv2_f');
net.params(f).value=net1.layers{3}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv2_b');
net.params(f).value=net1.layers{3}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('pool1', dagnn.Pooling('poolSize', [2,2], 'pad', [0,0,0,0],...
    'stride', [2,2]),'relu_2','pool_1');

%----------------------------------
net.addLayer('conv3', dagnn.Conv('size', [3,3,64,128], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'pool_1', 'conv_3', {'conv3_f', 'conv3_b'});
net.addLayer('relu3', dagnn.ReLU(), 'conv_3', 'relu_3');

f=net.getParamIndex('conv3_f');
net.params(f).value=net1.layers{6}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv3_b');
net.params(f).value=net1.layers{6}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv4', dagnn.Conv('size', [3,3,128,128], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_3', 'conv_4', {'conv4_f', 'conv4_b'});
net.addLayer('relu4', dagnn.ReLU(), 'conv_4', 'relu_4');

f=net.getParamIndex('conv4_f');
net.params(f).value=net1.layers{8}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv4_b');
net.params(f).value=net1.layers{8}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('pool2', dagnn.Pooling('poolSize', [2,2], 'pad', [0,0,0,0],...
    'stride', [2,2]),'relu_4','pool_2');

%------------------------------
net.addLayer('conv5', dagnn.Conv('size', [3,3,128,256], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'pool_2', 'conv_5', {'conv5_f', 'conv5_b'});
net.addLayer('relu5', dagnn.ReLU(), 'conv_5', 'relu_5');

f=net.getParamIndex('conv5_f');
net.params(f).value=net1.layers{11}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv5_b');
net.params(f).value=net1.layers{11}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv6', dagnn.Conv('size', [3,3,256,256], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_5', 'conv_6', {'conv6_f', 'conv6_b'});
net.addLayer('relu6', dagnn.ReLU(), 'conv_6', 'relu_6');

f=net.getParamIndex('conv6_f');
net.params(f).value=net1.layers{13}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv6_b');
net.params(f).value=net1.layers{13}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv7', dagnn.Conv('size', [3,3,256,256], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_6', 'conv_7', {'conv7_f', 'conv7_b'});
net.addLayer('relu7', dagnn.ReLU(), 'conv_7', 'relu_7');

f=net.getParamIndex('conv7_f');
net.params(f).value=net1.layers{15}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv7_b');
net.params(f).value=net1.layers{15}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv8', dagnn.Conv('size', [3,3,256,256], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_7', 'conv_8', {'conv8_f', 'conv8_b'});
net.addLayer('relu8', dagnn.ReLU(), 'conv_8', 'relu_8');

f=net.getParamIndex('conv8_f');
net.params(f).value=net1.layers{17}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv8_b');
net.params(f).value=net1.layers{17}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('pool3', dagnn.Pooling('poolSize', [2,2], 'pad', [0,0,0,0],...
    'stride', [2,2]),'relu_8','pool_3');

net.addLayer('conv9', dagnn.Conv('size', [3,3,256,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'pool_3', 'conv_9', {'conv9_f', 'conv9_b'});
net.addLayer('relu9', dagnn.ReLU(), 'conv_9', 'relu_9');

f=net.getParamIndex('conv9_f');
net.params(f).value=net1.layers{20}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv9_b');
net.params(f).value=net1.layers{20}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv10', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_9', 'conv_10', {'conv10_f', 'conv10_b'});
net.addLayer('relu10', dagnn.ReLU(), 'conv_10', 'relu_10');

f=net.getParamIndex('conv10_f');
net.params(f).value=net1.layers{22}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv10_b');
net.params(f).value=net1.layers{22}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv11', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_10', 'conv_11', {'conv11_f', 'conv11_b'});
net.addLayer('relu11', dagnn.ReLU(), 'conv_11', 'relu_11');

f=net.getParamIndex('conv11_f');
net.params(f).value=net1.layers{24}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv11_b');
net.params(f).value=net1.layers{24}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv12', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_11', 'conv_12', {'conv12_f', 'conv12_b'});
net.addLayer('relu12', dagnn.ReLU(), 'conv_12', 'relu_12');

f=net.getParamIndex('conv12_f');
net.params(f).value=net1.layers{26}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv12_b');
net.params(f).value=net1.layers{26}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('pool4', dagnn.Pooling('poolSize', [2,2], 'pad', [0,0,0,0],...
    'stride', [2,2]),'relu_12','pool_4');

net.addLayer('conv13', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'pool_4', 'conv_13', {'conv13_f', 'conv13_b'});
net.addLayer('relu13', dagnn.ReLU(), 'conv_13', 'relu_13');

f=net.getParamIndex('conv13_f');
net.params(f).value=net1.layers{29}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv13_b');
net.params(f).value=net1.layers{29}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv14', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_13', 'conv_14', {'conv14_f', 'conv14_b'});
net.addLayer('relu14', dagnn.ReLU(), 'conv_14', 'relu_14');

f=net.getParamIndex('conv14_f');
net.params(f).value=net1.layers{31}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv14_b');
net.params(f).value=net1.layers{31}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv15', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_14', 'conv_15', {'conv15_f', 'conv15_b'});
net.addLayer('relu15', dagnn.ReLU(), 'conv_15', 'relu_15');

f=net.getParamIndex('conv15_f');
net.params(f).value=net1.layers{33}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv15_b');
net.params(f).value=net1.layers{33}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('conv16', dagnn.Conv('size', [3,3,512,512], 'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_15', 'conv_16', {'conv16_f', 'conv16_b'});
net.addLayer('relu16', dagnn.ReLU(), 'conv_16', 'relu_16');

f=net.getParamIndex('conv16_f');
net.params(f).value=net1.layers{35}.filters;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

f=net.getParamIndex('conv16_b');
net.params(f).value=net1.layers{35}.biases;
net.params(f).learningRate=0;
net.params(f).weightDecay=0;

net.addLayer('pool5', dagnn.Pooling('poolSize', [2,2], 'pad', [0,0,0,0],...
    'stride', [2,2]),'relu_16','pool_5');

end

