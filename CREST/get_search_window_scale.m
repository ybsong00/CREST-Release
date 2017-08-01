function window_sz = get_search_window_test( target_sz, im_sz)
% GET_SEARCH_WINDOW

% if(target_sz(1)/target_sz(2) > 2)
%     % For objects with large height, we restrict the search window with padding.height
%     window_sz = floor(target_sz.*[1+padding.height, 1+padding.generic]);
%     
% elseif(prod(target_sz)/prod(im_sz(1:2)) > 0.05)
%     % For objects with large height and width and accounting for at least 10 percent of the whole image,
%     % we only search 2x height and width
%     window_sz=floor(target_sz*(1+padding.large));
%     
% else
%     %otherwise, we use the padding configuration
%     window_sz = floor(target_sz * (1 + padding.generic));

ratio=target_sz(1)/target_sz(2);
if ratio>1    
    window_sz=round(target_sz.*[2,2*ratio]);
else
    window_sz=round(target_sz.*[2/ratio,2]);
end
% 
% %window_sz=round(target_sz.*[5,9]);
window_sz=window_sz-mod(window_sz,2)+1;

end

