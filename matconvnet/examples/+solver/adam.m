function [w, state] = adam(w, state, grad, opts, lr)
%SGD
%   Example SGD solver, with momentum, for use with CNN_TRAIN and
%   CNN_TRAIN_DAG.
%
%   The convergence of SGD depends heavily on the learning rate (set in the
%   options for CNN_TRAIN and CNN_TRAIN_DAG).
%
%   If called without any input argument, returns the default options
%   structure.
%
%   Solver options: (opts.train.solverOpts)
%
%   `momentum`:: 0.9
%      Parameter for Momentum SGD; set to 0 for standard SGD.
%
%   Note: for backwards compatibility, the parameter can also be set in
%   opts.train.momentum.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 0 % Return the default solver options
  w = struct('momentum', 0.9);
  return;
end
if isempty(state)
  state.m = 0;
  state.v = 0;
  state.momentum1t = 1;
  state.momentum2t = 1;
%   momentum = 0 ;
end

opts.momentum1 = 0.9;
opts.momentum2 = 0.999;

state.momentum1t = state.momentum1t*opts.momentum1;
state.momentum2t = state.momentum2t*opts.momentum2;
state.m = opts.momentum1*state.m + (1-opts.momentum1)*grad;
state.v = opts.momentum2*state.v + (1-opts.momentum2)*grad.^2;

momentum = - state.m/(1-state.momentum1t);
momentum = momentum ./ (sqrt(state.v/(1-state.momentum2t)) + 10^(-8));
w = w + lr * momentum ;
