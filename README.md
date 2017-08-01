This is the implementation of our CREST paper. It contains matconvnet toolbox and Skiing sequences from the OTB dataset. Our main development is kept in the folder CREST.

Before running our code, make sure you have a state-of-the-art GPU. I develop this code using Titan Black. Make sure yours are better than mine :-).

Please download the VGG-16 model and put it on the 'CREST/exp/model/' directory. You can download VGG-16 model via http://www.vlfeat.org/matconvnet/pretrained/.

Meanwhile, please configure matconvnet on your side.

Try 'CREST/demo.m' to see the tracker performance on the Skiing sequences.

<p>If you find the code and datasets useful in your research, please cite:</p>
<pre><code>@inproceedings{LapSRN,
    author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
    title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
    booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
    year      = {2017}
}
</code></pre>
