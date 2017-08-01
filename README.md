This is the implementation of our CREST paper. The project page can be found here:
http://www.cs.cityu.edu.hk/~yibisong/iccv17/index.html

There is three folders in this repository where matconvnet toolbox and Skiing sequences are contained. Our main development is kept in the folder CREST.

Before running our code, check if you have a state-of-the-art GPU. I develop this code using Titan Black. Make sure yours are better than mine :-).

Please download the VGG-16 model and put it on the 'CREST/exp/model/' directory. You can download VGG-16 model via http://www.vlfeat.org/matconvnet/pretrained/.

Meanwhile, please configure matconvnet on your side.

Try 'CREST/demo.m' to see the tracker performance on the Skiing sequences.

<p>If you find the code useful, please cite:</p>
<pre><code>@inproceedings{song-iccv17-CREST,
    author    = {Song, Yibing and Ma, Chao and Gong, Lijun and Zhang, Jiawei and Lau, Rynson and Yang, Ming-Hsuan}, 
    title     = {CREST: Convolutional Residual Learning for Visual Tracking}, 
    booktitle = {IEEE International Conference on Computer Vision},
    year      = {2017}
}
</code></pre>
