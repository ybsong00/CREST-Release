function net_online = initNet(target_sz1)
%Init network

channel=64;

rw=ceil(target_sz1(2)/2);
rh=ceil(target_sz1(1)/2);
fw=2*rw+1;
fh=2*rh+1;

net_online=dagnn.DagNN();

net_online.addLayer('conv11', dagnn.Conv('size', [fw,fh,channel,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1]), 'input1', 'conv_11', {'conv11_f', 'conv11_b'});

f = net_online.getParamIndex('conv11_f') ;
net_online.params(f).value=single(randn(fh,fw,channel,1) /...
    sqrt(rh*rw*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv11_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;

net_online.addLayer('conv21', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input1', 'conv_21', {'conv21_f', 'conv21_b'});
net_online.addLayer('relu1', dagnn.ReLU(), 'conv_21', 'relu_1');

f = net_online.getParamIndex('conv21_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv21_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv22', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_1', 'conv_22', {'conv22_f', 'conv22_b'});
net_online.addLayer('relu2', dagnn.ReLU(), 'conv_22', 'relu_2');

f = net_online.getParamIndex('conv22_f') ;
net_online.params(f).value=single(randn(1,1,channel,channel) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv22_b') ;
net_online.params(f).value=single(zeros(channel,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;

net_online.addLayer('conv23', dagnn.Conv('size', [1,1,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'relu_2', 'conv_23', {'conv23_f', 'conv23_b'});

f = net_online.getParamIndex('conv23_f') ;
net_online.params(f).value=single(randn(1,1,channel,1) /...
    sqrt(1*1*channel));
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1;

f = net_online.getParamIndex('conv23_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1;


net_online.addLayer('conv31', dagnn.Conv('size', [1,1,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input2', 'conv_31', {'conv31_f', 'conv31_b'});

f = net_online.getParamIndex('conv31_f') ;
net_online.params(f).value=single(randn(1,1,channel,1) /...
    sqrt(1*1*channel))/1e10;
net_online.params(f).learningRate=1e-2;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv31_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2e-2;
net_online.params(f).weightDecay=1e3;


net_online.addLayer('sum1',dagnn.Sum(),{'conv_23','conv_11','conv_31'},'sum_1');

net_online.addLayer('L2Loss',...
    RegressionL2Loss(),{'sum_1','label'},'objective');

end
