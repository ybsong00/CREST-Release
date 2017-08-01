function out = get_subwindow(im, pos, sz)
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

if isscalar(sz),  %square sub-window
    sz = [sz, sz];
end

ys = floor(pos(1)) + (1:sz(1)) - ceil(sz(1)/2);
xs = floor(pos(2)) + (1:sz(2)) - ceil(sz(2)/2);

% Check for out-of-bounds coordinates, and set them to the values at the borders
xs = clamp(xs, 1, size(im,2));
ys = clamp(ys, 1, size(im,1));

%extract image
out = im(ys, xs, :);

end

function y = clamp(x, lb, ub)
% Clamp the value using lowerBound and upperBound

y = max(x, lb);
y = min(y, ub);

%y=x;
% idx=find(x<lb);
% for i=1:length(idx)
%     y(idx(i))=min(2*lb-x(idx(i)),ub);
% end


% idx=find(x>ub);
% for i=1:length(idx)
%     y(idx(i))=max(2*ub-x(idx(i)),lb);
% end

end