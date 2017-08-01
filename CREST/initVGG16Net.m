function [net1, avgImg] = initVGG16Net( )
%VGG16
opts.gpus=1;
cold=true;
prepareGPUs(opts,cold);

net2=load('./exp/model/imagenet-vgg-verydeep-16.mat');
net2.layers(31:end) = [];

avgImg=net2.meta.normalization.averageImage;

net1=dagnn.DagNN();

net1.addLayer('conv1_1', dagnn.Conv('size', [3,3,3,64],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'input', 'conv_11', {'conv11_f', 'conv11_b'});
net1.addLayer('relu1_1', dagnn.ReLU(),'conv_11','relu_11');

f = net1.getParamIndex('conv11_f') ;
net1.params(f).value=net2.layers{1}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv11_b') ;
net1.params(f).value=net2.layers{1}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

net1.addLayer('conv1_2', dagnn.Conv('size', [3,3,64,64],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_11', 'conv_12', {'conv12_f', 'conv12_b'});
net1.addLayer('relu1_2', dagnn.ReLU(),'conv_12','relu_12');
net1.addLayer('pool1', dagnn.Pooling('poolSize', [2,2], 'pad',...
    [0,1,0,1], 'stride', [2,2]),'relu_12','pool_1');

f = net1.getParamIndex('conv12_f') ;
net1.params(f).value=net2.layers{3}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv12_b') ;
net1.params(f).value=net2.layers{3}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;



%-------------------------covn 2--------------------
net1.addLayer('conv2_1', dagnn.Conv('size', [3,3,64,128],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'pool_1', 'conv_21', {'conv21_f', 'conv21_b'});
net1.addLayer('relu2_1', dagnn.ReLU(),'conv_21','relu_21');

f = net1.getParamIndex('conv21_f') ;
net1.params(f).value=net2.layers{6}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv21_b') ;
net1.params(f).value=net2.layers{6}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

net1.addLayer('conv2_2', dagnn.Conv('size', [3,3,128,128],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_21', 'conv_22', {'conv22_f', 'conv22_b'});
net1.addLayer('relu2_2', dagnn.ReLU(),'conv_22','relu_22');
net1.addLayer('pool2', dagnn.Pooling('poolSize', [2,2], 'pad',...
    [0,1,0,1], 'stride', [2,2]),'relu_22','pool_2');

f = net1.getParamIndex('conv22_f') ;
net1.params(f).value=net2.layers{8}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv22_b') ;
net1.params(f).value=net2.layers{8}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

%-------------------------covn 3--------------------
net1.addLayer('conv3_1', dagnn.Conv('size', [3,3,128,256],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'pool_2', 'conv_31', {'conv31_f', 'conv31_b'});
net1.addLayer('relu3_1', dagnn.ReLU(),'conv_31','relu_31');

f = net1.getParamIndex('conv31_f') ;
net1.params(f).value=net2.layers{11}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv31_b') ;
net1.params(f).value=net2.layers{11}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

net1.addLayer('conv3_2', dagnn.Conv('size', [3,3,256,256],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_31', 'conv_32', {'conv32_f', 'conv32_b'});
net1.addLayer('relu3_2', dagnn.ReLU(),'conv_32','relu_32');

f = net1.getParamIndex('conv32_f') ;
net1.params(f).value=net2.layers{13}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv32_b') ;
net1.params(f).value=net2.layers{13}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

net1.addLayer('conv3_3', dagnn.Conv('size', [3,3,256,256],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_32', 'conv_33', {'conv33_f', 'conv33_b'});
net1.addLayer('relu3_3', dagnn.ReLU(),'conv_33','relu_33');

f = net1.getParamIndex('conv33_f') ;
net1.params(f).value=net2.layers{15}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv33_b') ;
net1.params(f).value=net2.layers{15}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

%-------------------------covn 4--------------------
net1.addLayer('conv4_1', dagnn.Conv('size', [3,3,256,512],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_33', 'conv_41', {'conv41_f', 'conv41_b'});
net1.addLayer('relu4_1', dagnn.ReLU(),'conv_41','relu_41');

f = net1.getParamIndex('conv41_f') ;
net1.params(f).value=net2.layers{18}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv41_b') ;
net1.params(f).value=net2.layers{18}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

net1.addLayer('conv4_2', dagnn.Conv('size', [3,3,512,512],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_41', 'conv_42', {'conv42_f', 'conv42_b'});
net1.addLayer('relu4_2', dagnn.ReLU(),'conv_42','relu_42');

f = net1.getParamIndex('conv42_f') ;
net1.params(f).value=net2.layers{20}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv42_b') ;
net1.params(f).value=net2.layers{20}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

net1.addLayer('conv4_3', dagnn.Conv('size', [3,3,512,512],...
    'hasBias', true, 'pad',...
    [1,1,1,1], 'stride', [1,1]), 'relu_42', 'conv_43', {'conv43_f', 'conv43_b'});
net1.addLayer('relu4_3', dagnn.ReLU(),'conv_43','relu_43');

f = net1.getParamIndex('conv43_f') ;
net1.params(f).value=net2.layers{22}.weights{1};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1e3;

f = net1.getParamIndex('conv43_b') ;
net1.params(f).value=net2.layers{22}.weights{2};
net1.params(f).learningRate=0;
net1.params(f).weightDecay=1;

% net1.addLayer('pool2_show', dagnn.Crop('crop',[1,1]),...
%     {'pool_2','pool_2'}, 'pool2show');
% net1.addLayer('relu33_show', dagnn.Crop('crop',[1,1]),...
%     {'relu_33','relu_33'}, 'relu33_show');



net1.move('gpu');
clear net2;
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tflow vl_imreadjpeg ;
end

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end
end

