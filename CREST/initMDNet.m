function [ net1, avgImg ] = initMDNet( )
%VGG-M finetuned by MDNet

net2=load('../0203/exp/model/mdnet_vot-otb.mat');

net1=dagnn.DagNN();
avgImg=[128;128;128];

net1.addLayer('conv1', dagnn.Conv('size', [7,7,3,96],...
    'hasBias', true, 'pad',...
    [3,3,3,3], 'stride', [2,2]), 'input', 'conv_1', {'conv1_f', 'conv1_b'});

f = net1.getParamIndex('conv1_f') ;
net1.params(f).value=net2.layers{1}.filters;
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

f = net1.getParamIndex('conv1_b') ;
net1.params(f).value=net2.layers{1}.biases;
net1.params(f).learningRate=0;
net1.params(f).weightDecay=0;

net1.addLayer('relu1', dagnn.ReLU(), 'conv_1', 'relu_1');
%net1.addLayer('lrn1', dagnn.LRN('param',[5 2 1e-4 0.75]), 'relu_1','lrn_1');
net1.addLayer('pool1', dagnn.Pooling('poolSize', [3,3], 'pad',...
    [1,1,1,1], 'stride', [2,2]),'relu_1','pool_1');


net1.addLayer('conv2', dagnn.Conv('size', [5,5,96,256],...
    'hasBias', true, 'pad',...
    [2,2,2,2], 'stride', [1,1]), 'pool_1', 'conv_2', {'conv2_f', 'conv2_b'});

f = net1.getParamIndex('conv2_f') ;
net1.params(f).value=net2.layers{5}.filters;
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

f = net1.getParamIndex('conv2_b') ;
net1.params(f).value=net2.layers{5}.biases;
net1.params(f).learningRate=0;
net1.params(f).weightDecay=0;

net1.addLayer('relu2', dagnn.ReLU(),'conv_2','relu_2');
%net1.addLayer('lrn2', dagnn.LRN('param',[5 2 1e-4 0.75]), 'relu_2','lrn_2');
% net1.addLayer('pool2', dagnn.Pooling('poolSize', [3,3], 'pad',...
%     [1,1,1,1], 'stride', [2,2]),'lrn_2','pool_2');


net1.addLayer('conv3', dagnn.Conv('size', [3,3,256,512],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_2', 'conv_3', {'conv3_f', 'conv3_b'});
f = net1.getParamIndex('conv3_f') ;
net1.params(f).value=net2.layers{9}.filters;
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

f = net1.getParamIndex('conv3_b') ;
net1.params(f).value=net2.layers{9}.biases;
net1.params(f).learningRate=0;
net1.params(f).weightDecay=0;

net1.addLayer('relu3', dagnn.ReLU(), 'conv_3', 'relu_3');

end

