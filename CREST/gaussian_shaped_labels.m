function labels = gaussian_shaped_labels(sigma, sz)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ. The output will have size SZ, representing
%   one label for each possible shift. The labels will be Gaussian-shaped,
%   with the peak at 0-shift (top-left element of the array), decaying
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


% 	%as a simple example, the limit sigma = 0 would be a Dirac delta,
% 	%instead of a Gaussian:
% 	labels = zeros(sz(1:2));  %labels for all shifted samples
% 	labels(1,1) = magnitude;  %label for 0-shift (original sample)


%evaluate a Gaussian with the peak at the center element
[rs, cs] = ndgrid((1:sz(1)) - ceil(sz(1)/2), (1:sz(2)) - ceil(sz(2)/2));

global objSize;
sizeMax=max(objSize);
sizeMin=min(objSize);
if sizeMax/sizeMin<1.025&&sizeMax>120
    alpha=0.2;
else
    alpha=0.3;
end

labels = exp(-alpha*(rs.^2/sigma(1)^2 + cs.^2/sigma(2)^2));

end

