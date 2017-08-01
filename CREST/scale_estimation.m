function target_szU = scale_estimation(im,pos,target_sz,window_szo,...
    net1,net_online,coeff,meanImg,featPCA1st)

num_channels=64;
threshold=0.15;

[h,w,~]=size(im);
im_sz=[h w];

cell_size=4;
l1_patch_num = ceil(window_szo/ cell_size);
l1_patch_num=l1_patch_num-mod(l1_patch_num,2)+1;
cos_window = hann(l1_patch_num(1)) * hann(l1_patch_num(2))';
sz_window=size(cos_window);

%%  ---------scale refinement------------
scale=[1 0.95 1.05];
value=zeros(3,1);
for i=1:length(scale)
    target_sz1=round(target_sz*scale(i));
    window_sz=get_search_window(target_sz1,im_sz);
    patch = get_subwindow(im, pos, window_sz);    
    patch1=double(imresize(patch,window_szo));
       
    net1.eval({'input',gpuArray(single(patch1-meanImg))});
    feat=gather(net1.vars(23).value);
    feat = imResample(feat, sz_window(1:2));
    feat = bsxfun(@times, feat, cos_window);

    [hf,wf,cf]=size(feat);
    feat_=reshape(feat,hf*wf,cf);
    feat_=feat_*coeff;
    featPCA=reshape(feat_,hf,wf,num_channels);
    net_online.eval({'input1',gpuArray(featPCA),'input2',gpuArray(featPCA1st)});
    regression_map=gather(net_online.vars(10).value);
    value(i)=max(regression_map(:));
    
    if value(1)<threshold        
        target_szU=target_sz;        
        return;
    end
end
[~,id]=max(value);

if id==1
    target_szU=target_sz;    
    return;    
end

target_szU=0.4*target_sz+0.6*round(target_sz*scale(id));

end

