Summary of One Stage Model Structure
===

<center>
    <img
    src=https://i.imgur.com/jKwAdSL.png
    width=700>
    Photo from Angèle Kamp
    <br></br>
</center>

Table of Content
[Toc]


## Introduction

One stage method object detection has became the main stream of computer vision field. The success of [YOLOv4](https://arxiv.org/abs/2004.10934) (or [YOLOv5](https://github.com/ultralytics/yolov5)) does not emerge out of the void, instead, there are many fascinating and inspired work before this great hit.

In this serise of notes, I want to share the logic flow and evolution of one stage model with you guys. Since there are already many well written review blogs about this topic, I decide to take another approch. I seperate the content into two parts:
1. [Summary of One Stage Model Structure](https://hackmd.io/LYDus2SFSLu2UjXTQJWctg?view): focus on the evolution of model structure only, that is, the whole picture of various models. In here, we can understand the overall idea of model evolution without going through many technique detail.
2. [Summary for One Stage Model Loss](https://hackmd.io/GQeRNWGhTQ6e9pUbvDSIhA?view): we will dive into the heart of machine learning model; the objective (loss) function. There are many calculation detail for encoding ground truth or decoding prediction in the ground of anchor-based method. Furthermore, I give some historical review of loss functions before YOLOv3. 

Before we start, there are three points I want to clearify:
1. Some contents are taken from other amazing blogs, I will try my best to give the reference for readers (check [Reference section](https://hackmd.io/LYDus2SFSLu2UjXTQJWctg?view#Reference)), but if you find that I miss the credit or reference, please do not hesitate to contact me (rch0421@gmail.com) or leave it in the comment. My sole purpose is to share my reading note for the journey of understanding those fine work. These blogs real help me along the path.
2. I am appreciated for readers pointing out which part I might give a wrong impression or a completely false concept, the discussion will only make this note become better.
3. Recently, anchor-free method in detection is catching up the anchor-based method. Interest reads can checkout [FCOS](https://hackmd.io/IX9MuFDwQXSYPDaeciwO8w) and [YOLOX](https://hackmd.io/-ag13jn4S4OvW_gccBDPzw?view)(in progress).


## Prerequisite

Class imbalance issue in object detection is a long standing problem. The number of negative samples (background) are usually much larger than positive samples (foreground). In one-stage models, it is the main obstacle for increasing accuracy, since many easy targets (negative sample) overwhelmingly control the flow of gradient, leading to low performance and training instability.

In the two stages model, this issue is largely alleviated by the first stage processing; the region of proposal network. However, this advantage comes with the price. Since the existance of two stages slows down the inference speed (one can roughly consider as passing through two object detection models per inference). If one stage models want to domain the field, they have to solve the class imbalance problem properly in some way.

As we dive deeper to the story below, we will find out that even though directly modifying loss function in order to favour hard examples (will be explained in the below, be patience~) can solve the issue; researchs find out that using objectness as an indicator for positive/nagative samples and combining advance model structure and feature fusing is actaully the key to solve the imbalance problem.

## You Only Look Once v1 (YOLOv1)

The model structure of YOLOv1:

<center>
    <img
    src=https://i.imgur.com/H2DX0Xe.png
    width=700>
    Taken from Joseph Redmon et al. 2016
    <br></br>
</center>

Let's look closer to the prediction module (annotated as "head" hereafter)

<center>
    <img
    src=https://i.imgur.com/oRLNVHl.png
    width=400>
    <br></br>
</center>


YOLOv1 proposes a brand new idea that using every pixel (or called a grid cell) in the "head" of the network (like the green box in above) as a region of proposal. The so called _unified detection_ means to compose the prediction of bounding box (bbox) and class probabilies together in the channel dimension of that grid cell.

To be more specified, for each grid cell, it will predict
* A set of class probabilities (note that it is for per grid cell, not per $B$, this point will be changed in the incoming models)
* $x, y$ gives the center of prediction bbox relative to bounds of grid cell
    <center>
        <br>
        <img
        src=https://i.imgur.com/JuEEPSQ.png
        width=200>
        <br></br>
    </center>
* $w, h$ relate to the whole image
* The confidence (we usually denote it as objectness) score is calculated from the IoU of predicted bbox and ground truth box.

While inference, the prediction score for a bbox is calculated through

\begin{equation}
\text{score} = P(\text{class}) \times \text{Objectness},
\end{equation}

in right hand side, the first term is the class probablity. 

In summary, YOLOv1 discards the idea of proposal network, therefore, the inference time can be greatly reduced. However, there is still a gap in the accuracy of v1 between two-stage methods. There are several points we can foresee (well, in some sense) might cause the drop of performance:

1. Fully-connect (FC) layers will break the spatial information (valuable for object detection) from feature extractor.
2. Even though, the max-pooling procedure can quickly reduce the spatial size, it hurts the information from small object
3. There are no limitation for a grid cell's prediction, a grid cell has to deal with various size of ground truth
4. Does not solve the issue of class imbalance 

Single shot multibox detector solves above issues at once. The result also shows that those improvements increase not only speed but accuracy.


## Single Shot Multibox Dectector (SSD)

The structure of SSD:
<center>
    <img
    src=https://i.imgur.com/dL43dbt.png
    width=700>
    Taken from Wei Liu et al. 2016
    <br></br>
</center>

SSD adapted multi-scale feature maps for prediction, which soon becomes the standard procedure in detection, and they abandon FC layer which reduces the number of parameters while keeping the spatial information. At this point, one stage model is fully convolutional.

SSD also uses the similrar concept of anchor bbox (they called it default box) from Fast-RCNN. It is a reference box for both network to make prediction and ground truth to be encoded. I will give a more thorough concept in the [ next part](https://hackmd.io/GQeRNWGhTQ6e9pUbvDSIhA), please bear with me for now. In addition, SSD author use online hard example mining loss (OHEM) to deal with class imbalance issue.

OHEM is a way to pick **hard examples/samples** in order to reduce computation cost and improve your network performance. It is mostly used for object detection. When you find yourself in the situation of many negatives (false positive, like background detection) are present as oppose to relatively much small positives (correct detection). It will be clever to pick a subset of negatives that are the most informative for your network to train with, i.e. **_hard examples mining_**.

> To be explicitly, hard examples can be considered as those anchor boxes have realtively larger compared to most of the population (but smaller than the thresold to be postive example) IoU with ground true boxes.

How exactly to find/mine the hard exmaples, one can check [here](https://hackmd.io/tGIMAyl0RqatLA3WBGoDlA?view) (most of the content is from [Online Hard Example Mining on PyTorch](https://erogol.com/online-hard-example-mining-pytorch/)).

In the above design, it indeed increases the detection ability across different scales. However, for the prediction in small scale (recetive field is small), the semantic information is not enough due to the shallow convolution in that scale. In the later, we will find out that **the feature fusion across scales**  can help the information flow while detecting small object.

Inspired by SSD, YOLO inherited several concepts from it and experiment many state-of-art methods at its time and give the v2 of YOLO.

## YOLOv2

The structure of YOLOv2:
<center>
    <img src=https://i.imgur.com/5w3KALg.png
     width=700>
     Taken from Joseph Redmon, Ali Farhadi 2016
    <br></br>
</center>

Based on the idea of YOLOv1 and SSD, YOLOv2 make several improvements:

1. ***Batch Normalization (BN)***
    which is used on all convolutional layers in YOLOv2. This makes gradient flow statistic more steady which in turn let a model to be coveraged faster. Furthremore, BN provides a regularization effect for noise.

2. ***Removing FC layer***,
    spatial information is kept, and there is no need for fixing input image size anymore (originally, neurons of FC layer have to match the flatted feature map comes from previous convolution), the input size only has to be the multiples of 32 (due to convolution strides). Due to the reduction of parameters, the model can be trained in ***higher resolution $(416×416)$***.

3. ***Using k-mean cluster on $1 - \text{IOU}$***. Check here for [detail](https://hackmd.io/e2lKiB4PTfSLc560eZRk7Q).

4. YOLOv2 used a custom deep architecture ***darknet-19***, an originally 19-layer network supplemented with 11 more layers for object detection (total 30).
    With a 30-layer architecture, YOLOv2 often struggled with small object detections. The reason is because as the layers downsampled the input, the loss of fine-grained features becomre greater. To remedy above, they ***add passthrough layer*** (The one that $26×26×256$ skip connect to $13×13×3072$), for information fusing.

5. In YOLOv2, they incorporates the anchor logic into the framework. 
    YOLOv1 does not have constraints on location prediction which makes the model unstable at early iterations. The predicted bounding box can be far from the original grid location. YOLOv2 bounds the location using logistic activation $\sigma$, which makes the value between 0 to 1.

6. Removing pooling, keep the prediction dense


Note that in YOLOv2, the scale information does not come from the model structure but changing the input image resolution while training. Ideally, the scale information  should be learned by the model itself. In the RetinaNet, we can see that how the model learn the concept of multi-scale by model design.


## RetinaNet

The structure of RetinaNet:
<center>
    <img src=https://i.imgur.com/Eh7DbgZ.png
     width=700>
     Taken from Tsung-Yi Lin et al. 2018
    <br></br>
</center>

In SSD, the problem of class imbalance is solved by OHEM. In RetinaNet, they find that the [focal loss](https://hackmd.io/GQeRNWGhTQ6e9pUbvDSIhA?view#RetinaNet-Loss) can further surpressing the problem of easy targets, and focus on the hard targets. In addition, RetinaNet adopted Feature Pyramid Networks (FPN) for feature fusion along with multi-scale predictions**. In result, their method are more sensitive to small object.


### Feature Pyramid Networks (FPN)

Since target's size often varied largely in object detection task, it is crucial to have a detector that is insensitive to the scale of the object. One of the methods is to predict the object in each pre-define scales, in SSD, they idea can be represented as

<center>
    <img src=https://i.imgur.com/EKA5FjE.png
    width=400>
    <br>
    Take from Tsung-Yi Lin et al. 2017
    <br></br>
</center>

It makes predictions for various scales along the way of downsampling. However, as we mentioned before, the small scale predictions are suffer from lack of semnatic information, while large scales are lake of spatial resolution. A top-down pathway can bring the information from semantic rich layers

<center>
    <img src=https://i.imgur.com/QY5ZOi8.png
    width=400>
    <br>
    Take from Tsung-Yi Lin et al. 2017
    <br></br>
</center>

While the reconstructed layers are semantic strong but the locations of objects are not precise after all the downsampling and upsampling procedures. We add lateral connections between reconstructed layers and the corresponding feature maps from backbone (which have abundent location information) to help the detector to predict the location betters. It also acts as skip connections to make training easier.

<center>
    <img src=https://i.imgur.com/KDaD4TC.png
    width=600>
    <br>
    Take from Tsung-Yi Lin et al. 2017
    <br></br>
</center>

To more precise, since their feature map sizes for semantic rich (smaller) and location rich (larger) are different. We need to upsamples (2x) the previous top-down stream and add it with the neighboring layer of the bottom-up stream (see the diagram below). The result is passed into a 3×3 convolution to reduce upsampling artifacts and create the feature maps P4 below for the head to furth processing.

<center>
    <img src=https://i.imgur.com/IQyzG8T.png
     width=400>
     <br>
     Taken from Jonathan Hui blog
    <br></br>
</center>

As whole

<center>
    <img src=https://i.imgur.com/qUapObw.png
     width=400>
    <br>
    Taken from Jonathan Hui blog
    <br></br>
</center>

In summary, RetinaNet use FPN greatly enchance the detection performance for small object. With this success, YOLOv3 also adopted the FPN into the network along with bunch of improvements.


## YOLOv3


The structure of YOLOv3:
<center>
    <img src=https://i.imgur.com/0Cqj4MM.jpg
     width=700>
    <br>
    Taken from Qiwei Wang et al. 2020
    <br></br>
</center>


Let's check it in detail:

<center>
    <img src=https://i.imgur.com/Btj8dRP.png
     width=700>
    <br>
    Taken from Haikuan Wang et al. 2020
    <br></br>
</center>

In YOLOv2, it suffers from the information loss of deep structure. Residual blocks, skip connections, and non-sigmoid function are added in YOLOv3 to solve above issues. Based on YOLOv2, ***YOLOv3 enhances the feature fusion like RetinaNet through FPN.*** The design of detection head is same as RetinaNet. That is, YOLOv3 makes predictions on 3 different scales.

Noitce that in YOLOv3, they also experiment with focal loss in their framework. However, contrary to expectations, usage of modified loss hurts performace around 2 mAP. In my own explanation, there are roughly 2 possible reasons:
1. RetinaNet (9 predictions per grid cell) class imbalance is more severe than YOLOv3
    > From Sik-Ho Tsang blog, total number of anchors:
    > * YOLOv1: 98 "boxes" (not anchor)
    > * YOLOv2: ~1k
    > * SSD: ~8–26k
    > * RetinaNet: ~100k
    > * YOLOv3: ~10k

    If not suppress the numerous negative samples, model will be hard to converge.
2. In YOLOv3, object, non-object masks and objectiveness (check [here](https://hackmd.io/GQeRNWGhTQ6e9pUbvDSIhA?view#YOLOv3-Loss)) help model get rid of many irrelevant anchors. Which greatly reduces the class imbalance problems.

In my naive guessing (no experiment support, btw), under YOLOv3 framework, focal loss might be cable of squeezing some performance gain in the finetuning stage of training, due to the fact that it pays more attention to those hard cases. On the other hand, it is not suitable for training from scratch, since in the begining of the training, punishment from both YOLOv3 obj/no-obj masks and focal loss might hurt the process of model seeking good cluster of information.


In the next section, we finaly arrive YOLOv4. Based on the success of YOLOv3, authors of v4 include and experiment great amount of new techniques into one stage detection framework. Most importantly, it is the start point for one stage models have noticeable accuracy gain over two stage models across different tasks while increasing high inference speed.

<center>
    <img src=https://i.imgur.com/dE1TiEK.png
     width=500>
    <br>
    Taken from Jeong-ah Kim et al. 2020
    <br></br>
</center>


## YOLOv4

There are mainly three parts in YOLOv4:

<center>
    <img src=https://i.imgur.com/AsyTh2n.png
     width=700>
    <br>
    Taken from Alexey Bochkovskiy et al. 2020
    <br></br>
</center>


1. CSPDarknet/Densenet backbone
    * Feature Extractor
2. Neck (SPP + PANet)
    * Enrich the feature from backbone before going to the classifier
3. YOLO Head (Same as YOLOv3)
    * Classifier/Regressor

The detail structure is like below:


<center>
    <img src=https://i.imgur.com/okWfYmx.png
     width=700>
    <br>
    Taken from Bubbliiiing blog
    <br></br>
</center>


In the following, I will focus on the backbone and neck structure, since the head structure does not changed from v3 (but the regression of $xywh$ is changed, check [here](https://hackmd.io/GQeRNWGhTQ6e9pUbvDSIhA?view#Bounding-Box-Regression)).

### Backbone

YOLOv4 backbone can be understanded by the combination of the following work:
* DenseNet (parameter reuse)
* Inception (model widthwise enlarge)
* Cross-Stage-Partial-connections net (CSPNet) and
* Darknet53 (whose earlier version already used in YOLOv2).

#### DenseNet

DenseNet gives the idea of parameters reusing in order to solve vanishing-gradient problem. Reusing parameters also reduce the number of parameters, furthermore, in DenseNet structure, the model encourages reusing feature from previous layers, which keep the low level information propagate to high level.

<center>
    <img src=https://i.imgur.com/hHJgOGr.png
     width=400>
    <br>
    Taken from Gao Huang et al. 2016
    <br></br>
</center>

From the idea of DenseNet, evey connectation will increase the number of feature maps, we needs to use $1\times 1$ kernal size convolution to reshape dimension (the so called transition). Note the previous layers are concated to the later, not like resnet is adding operation.


#### Inception

Previous CNN modules are focus on the depth of models, inception network focus on the width of model. When inception first come out, the problem it wants to solve is the recetive field is small for shallow layer while deep layer is easily over-fit or facing gradient vanishing problem.

<center>
    <img src=https://i.imgur.com/zR2YJqR.png
     width=600>
    <br>
    Taken from Christian Szegedy et al. 2015
    <br></br>
</center>

As the development of new techniques solving the gradient vanishing problem, the idea of model width is then used in CSPNet for reducing number of parameters.

#### Cross-Stage-Partial-connections Net (CSPNet)
> A little be more detail can be found in [here](https://hackmd.io/9yBn8bfnQcm86Mou_63S3A)

Since every output in DenseNet will be reused in the next layer, the increasing channels still requires great amount of computation. If we can decrease channels by dividing the channels into two parts, one part passes through original convolutions and the other just skipping to the end, the computation burden can be reduced. This the main idea of CSPNet

<center>
    <img src=https://i.imgur.com/zwKUjGm.png
     width=600>
     <br>
     Taken from Chien-Yao Wang et al. 2019
     <br></br>
</center>

CSPNet is a structure concept can be applied to many kinds of blocks (ResNet, ResNext or DenseNet). Researchers find out that above method reduces computations by $20\%$ while keeping or even surpass previous structure. The reason of CSP will have less computing burden than DenseNet is because the split and merge techquie paritially avoid the gradient repeatedly calculated for every channels in the previous layers.

#### Darknet53

Note that above is all belonged to one kind of module/block and these blocks can be composed by a framework. Darknet53 is a framework like below, 


<center>
    <img src=https://i.imgur.com/qz4icR7.png
     width=300>
     <br>
     Taken from Joseph Redmon et al. 2018
     <br></br>
</center>

By replacing every block in the above diagram by CSPDense block (please ignore all the output numbers and all the texts below Avgpool), then we have the backbone of YOLOv4.


### Neck (SPP + PANet)

Even though, the backbone can give us the best set of features, to add/cat/mix or manipulate those features become an important job before feeding to the Head part.

To enrich the information before feeding into the head, neighboring feature maps coming from the bottom-up stream and the top-down stream are added together element-wise or concatenated before feeding into the head.

:::info
Therefore, ***the head’s input will contain spatial rich information from the bottom-up stream and the semantic rich information from the top-down stream.*** This part of the system is called a neck.
:::

In YOLOv4 design, the last feature map from the backbone is fed into SPP, and the output of SPP is fed into PANet. At the same time, feature maps from the middle layer of backbone are also taken as input of PANet. PANet does the job of upsampling/downsampling those added/concated features.

#### Spatial Pyramid Pooling layer (SPP)

SPP applies a slightly different strategy from FPN for detecting objects of different scales:

1. it is applied at the last feature maps. Since the last layer has the largest receptive field.
2. it replaces the last pooling layer (after the last convolutional layer from backbone) with a spatial pyramid pooling layer.

<center>
    <img src=https://i.imgur.com/Pzu02F6.png
     width=600>
     <br>
     Taken from Kaiming He et al. 2014
     <br></br>
</center>

***The feature maps are spatially divided into $m\times m$ bins with $m$, say, equals 1, 2, and 4 respectively. Then a maximum pool is applied to each bin for each channel.*** ***This forms a fixed-length representation.*** For classification CNN model, it can be further analyzed with FC-layers.

Note that, in the usual CNN model, FC-layers are connect to the last layer of conv layers, which are then used as score for classification. For object detection, not only the semantic informaion is important ***but also spatial information***, convert 2-D feature maps into a fixed-size 1-D vector is not necessarily desirable. Therefore, SPP strucutre in YOLO is modified.

#### SPP in YOLO

The reason of adding SPP structure only to the last feature map, I believe, is because the last layer contains the highest level information, largest receptive field, and complete information compared to other scales (Of course, one can argue that apply SPP on the other scales need more matrix operations since their feature maps are larger). With this crucial information at hand, we can further extract the most important information meanwhile get a larger receptive file by max pooling with varied kernal and stride size.


In SPP-YOLOv3 (YOLOV4 follows this procedure), ***SPP is modified to retain the output spatial dimension. Max-pooling is applied with kernel sizes, 1×1, 5×5, 9×9, 13×13 and stride one. The spatial dimension is then preserved. The features maps from different kernel sizes are concatenated together as output.*** Note that the stride of pooling is all one, therefore, their output size remain constant.

<center>
    <img src=https://i.imgur.com/uORoEya.png
     width=400>
     <br>
     Taken from Zhanchao Huang et al. 2019.
     <br></br>
</center>

The concatenated output goes through a 3x3 convolution and along with other two scales (also through 3x3 convolution) are fed into feature fusing part.

#### Path Aggregation Network (PAN)

:::info
Note that below is the original formulation in PANet. The concept can be applied to YOLOv4 structure. The exact procedure can be refered to the daigram at the begining of this section.
:::

The diagram below is Path Aggregation Network (PAN) for object detection. ***A bottom-up path (b) is augmented to make low-layer information easier to propagate to the top.*** In FPN, the localized spatial information traveled upward in the red arrow. While not clearly demonstrates in the diagram, the red path goes through about 100+ layers (since the backbone design, ResNet101). PAN introduces a short-cut path (the green path) which only takes about 10 layers to go to the top $N_5$ layer.

***This short-circuit concepts make fine-grain localized information available to top layers.*** Since top layers' neurons are sensitive to the whole object, while does not very activate for edges which low layers are responsible. The short cut connection can bring high resolution information to the high level, the prediction will become easy while predicting the localization of the object.

They use $\{N2, N3, N4, N5\}$ to denote newly generated feature maps corresponding to $\{P2, P3, P4, P5\}$. Note that $N2$ is simply $P2$, without any processing.

Each feature map $N_i$ first goes through a $3 \times 3$ convolutional layer with stride 2 to reduce the spatial size. Then each element of feature map $P_{i+1}$ and the down-sampled map are "added" (which change to concatenate) through lateral connection. 

The fused feature map is then processed by another 3 × 3 convolutional layer to generate $N_{i+1}$ for following sub-networks.  They consistently use channel 256 of feature maps.

<center>
    <img src=https://i.imgur.com/1XRkWXZ.png
     width=500>
     <br>
     Taken from Shu Liu et al. 2018
    <br></br>
</center>

#### PAN in YOLOv4

In YOLOv4, instead of adding neighbor layers together (e.g. $P_3$ and $N_2$ to $N_3$), features maps are concatenated together.

<center>
    <img src=https://i.imgur.com/TcNKih6.png
     width=450>
     <br>
     Taken from Alexey Bochkovskiy et al. 2020
    <br></br>
</center>

Finally, we can check the structure adapted in YOLOv4. Like PANet suggested, the short-cut path make the information from largest scale to the smallest scale through only 10 convolutions, vice versa.

<center>
    <img src=https://i.imgur.com/XyXjkdu.png
     width=300>
    <br>
    Taken from Bubbliiiing blog
    <br></br>
</center>


Even though there many other feature fusing techniques like below

<center>
    <img src=https://i.imgur.com/X68ygn4.png
    width=600>
    <br>
    Taken from Mingxing Tan at el. 2020
    <br></br>
</center>

In YOLOv4, they claim that they has experimented above all structures. Not sure why they do not give a comparison result. BiFPN actually gives mutiple top-down and bottom-up pathways for better feature fusion and it uses less parameters than PAN. It will be nice to see the future development of changing the neck structure.


I am fully aware that there are still bunch of techniques YOLOv4 used or experimented which are not included in here. Interest readers might go to [here](https://hackmd.io/x1acXk9bQAGu9xK_kLczng?view) (Self-Attention Module, SAM) and [here](https://hackmd.io/dr_pmo47TUq79dqmCsJCNw?view) (bag of everthings), but I beleive that there are already plenty of blogs give wonderful reviews for each topic.

:::info
Do notice that above two links might not as complete or well organized like this one.
:::

In my personally experience, attention modules like SAM or [SE block](https://hackmd.io/K0S4fzYrQd20352FdGVsRg?view) indeed help big/deep model get extra accuracy or make a model captures some fine detail if in your case detail matters. If one wishes to squeeze bouns mAP, why not give them a shot.

## In Summary

Okay, what a journay isn't it? Congrats to whoever make to the end! Let's summarize what we have learn along the way:

YOLOv1 brings the object detection model into the era of realtime. However, in the earlier stage of development in one stage model, class imblanace problem is always a pain in the ass for controlling gradient flow. YOLOv1 to YOLOv2 are both suffer from the training instability and low performance despite many hard work is incoperated in.

After RetinaNet introduced, the multi-scale prediction finally is included into the YOLO framework. With the combination of (non)object mask, YOLOv3 is further utilizing objectness to replace loss penalty terms (positive/negative). At this point, class imbalance issue starts to become a minor problem in one stage model. Furthermore, a mature framework is formed: backbone, neck and head.

Standing on the shoulders of giants, with many excellent new ideas in community, and great amount of experiments are conducted (so we don't have to), YOLOv4 authors gives us the state-of-art progress in one stage model. 

## Backlog

The first part is finished. As we get a whole picture and history of YOLO development, we can start to dive into a lit bit detail of how machine learns the world through the objetctive function. Hope you guys like this serise :baby_chick: 

BTW...
You Only Learn One Representation is out! Checkout their [fanciating work](https://paperswithcode.com/paper/you-only-learn-one-representation-unified).


## Reference and Link

##### Related Note
* [Summary of One Stage Model Loss](/GQeRNWGhTQ6e9pUbvDSIhA)
* [FCOS](https://hackmd.io/IX9MuFDwQXSYPDaeciwO8w)
* [YOLOX](https://hackmd.io/-ag13jn4S4OvW_gccBDPzw?view) (in progress)
* [OHEM](https://hackmd.io/tGIMAyl0RqatLA3WBGoDlA?view)
* [K-mean for anchors](https://hackmd.io/e2lKiB4PTfSLc560eZRk7Q)
* [Cross Stage Partial Network, CSPNet](https://hackmd.io/9yBn8bfnQcm86Mou_63S3A)
* [Self-Attention Module, SAM](https://hackmd.io/x1acXk9bQAGu9xK_kLczng?view)
* [Bag of Freebies and Bag of Specials](https://hackmd.io/dr_pmo47TUq79dqmCsJCNw?view)
* [SE block](https://hackmd.io/K0S4fzYrQd20352FdGVsRg?view)


##### Blog
* [Notes for Object Detection: One Stage Methods](https://www.yuthon.com/post/tutorials/notes-for-object-detection-one-stage-methods/)
* [Online Hard Example Mining on PyTorch](https://erogol.com/online-hard-example-mining-pytorch/)
* [YOLOv2--論文學習筆記（算法詳解）](https://www.twblogs.net/a/5bafd96b2b7177781a0f6394)
* [Review: YOLOv2 & YOLO9000 — You Only Look Once (Object Detection)](https://towardsdatascience.com/review-yolov2-yolo9000-you-only-look-once-object-detection-7883d2b02a65)
* [Understanding Feature Pyramid Networks for object detection (FPN)](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
* [Review: RetinaNet — Focal Loss (Object Detection)](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)
* [YOLOv3 Structure Diagram](https://widgets.figshare.com/articles/8322632/embed?)
* [A Real-Time Safety Helmet Wearing Detection Approach Based on CSYOLOv3](https://www.researchgate.net/publication/345784529_A_Real-Time_Safety_Helmet_Wearing_Detection_Approach_Based_on_CSYOLOv3)
* [YOLOv3](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
* [YOLOv3 architecture.](https://plos.figshare.com/articles/YOLOv3_architecture_/8322632/1)
* [Dive Really Deep into YOLO v3: A Beginner’s Guide](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)
* [Comparison of Faster-RCNN, YOLO, and SSD for Real-Time Vehicle Type Recognition](https://ieeexplore.ieee.org/document/9277040)
* [DenseNet 學習心得](https://medium.com/%E5%AD%B8%E4%BB%A5%E5%BB%A3%E6%89%8D/dense-cnn-%E5%AD%B8%E7%BF%92%E5%BF%83%E5%BE%97-%E6%8C%81%E7%BA%8C%E6%9B%B4%E6%96%B0-8cd8c65a6f3f)
* [A Simple Guide to the Versions of the Inception Network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
* [YOLOv4](https://medium.com/@jonathan_hui/yolov4-c9901eaa8e61)
* [睿智的目标检测30——Pytorch搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106214657#1Backbone_47)


###### Paper
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640v5)
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
* [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242v1)
* [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
* [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
* [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
* [YOLOv4: Optimal Speed and Accuracy of Object Detection
](https://arxiv.org/abs/2004.10934)
* [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
* [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
* [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567v3.pdf)
* [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
* [Path Aggregation Network for Instance Segmentation](https://arxiv.org/pdf/1803.01534.pdf)
* [DC-SPP-YOLO: Dense Connection and Spatial Pyramid
Pooling Based YOLO for Object Detection](https://arxiv.org/abs/1903.08589)
* [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)
* [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206v1.pdf)
