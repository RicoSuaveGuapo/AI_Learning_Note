# Summary of One Stage Model Loss

<center>
    <img
    src=https://i.imgur.com/AYQMMSH.png
    width=500>
    <br>
    Photo from Annie Spratt
    <br></br>
</center>


Table of Content
[toc]


## Introduction

In the second part of one stage model summary (check the first part: [Summary of One Stage Model Structure](https://hackmd.io/LYDus2SFSLu2UjXTQJWctg?view)), we will begin to discuss detail of encoding ground truth for loss computation. Before we begin, there are several terms are easily to be mixed up, like grid cells, prior boxes, default boxes, and anchors. In short:

* A grid cell refers to the cell located in the final layer of convolution feature map (head part). Its depth (i.e. channel dimension) has different assigned meaning, like location offsets, objectness or class probabilities.
* Prior boxes, default boxes, or most often anchors are all refered to same concept, that is, the pre-define bounding boxes are taken as reference "points" for a model to give predictions or ground truth to be encoded relative to them.
* One grid cell usually have mutiple anchors assigned to it. For example, a grid cell in YOLOv4 head has three anchors assigned, each of anchor has their own pre-define heigh-to-wdith ratio (which might or might not depend on data)

<center>
    <img
    src=https://i.imgur.com/ls7wknv.png
    width=700>
    <br></br>
</center>


Naively, we can  directly predict the location and  dimension of a bounding box, but in practice, that leads to unstable gradients during training. Instead, most of the modern object detectors use the concept of anchor and perform log-space transformations. There are another approach though, like famous anchor-free methods [FCOS](https://hackmd.io/IX9MuFDwQXSYPDaeciwO8w). Recently, several work has showed the capbablity of the anchor-free is catching up the anchor-based.

:::info
Note that the first paper brought out the idea of anchor is  Faster R-CNN. Not YOLO serise.
:::

In the below, I will first give some overall idea of anchors. Then we will begin to check the loss functions in YOLOv3/v4. One can also jump to [Historical Review](https://hackmd.io/GQeRNWGhTQ6e9pUbvDSIhA?view) part for SSD, v1, v2, and RetinaNet loss functions.

## YOLO Logic

The features learned by the convolutional layers are passed to a classifier/regressor which makes the detection prediction (coordinates of the bounding boxes, the class label.. etc). **In YOLO, the prediction is done by using a convolutional layer which uses $1 \times 1$ convolutions. That is, YOLO prediction output is a "map"**. Note that since $1 \times 1$ kernal is used, the size of the output is exactly the same size as the input, and the channel dimensions are matched to desired number (see below).

:::success
Noitce that the "map" here has a specialized meaning. With the designed loss function, they have the meaning of offset to a anchor box, objectness and class probability. It is different from what we usually see in backbone or neck which have location or semantic information.
:::


Since YOLOv3, we have $B \times (5 + C)$ entries in the channel dimension per grid cell. $B$ represents the number of anchors each cell can predict (YOLOv1 use 2, YOLOv2 use 5, and YOLOv3/v4 use 3 per scale). For a certain scale and a grid cell, each of these $B$ anchors has unique height-to-width ratio.

>YOLOv1 has some subtilties, check [the first part](https://hackmd.io/LYDus2SFSLu2UjXTQJWctg?view#You-Only-Look-Once-v1-YOLOv1) of this serise


Each of the anchor has $5 + C$ attributes, which includes
* the center coordinates ($t_x, t_y$),
* the dimensions ($t_w, t_h$),
* the objectness score ($p_{obj}$) and
* class confidences ($p_{i}$).

You will expect that each cell of the map predicts an object through one of it's anchors **if the center of the object falls in the receptive field of that cell.**

:::info
This has to do with how YOLO is trained. In YOLO logic, one anchor is only responsible for detecting one object. While one object can be assigned to many anchors (One-to-many).
:::

First, we must ascertain which cell(s) this object belongs to. To do that, we divide a input image into a grid of map whose dimensions are matched to the "map". To be more specified, let's consider an example. An input image is $416 \times 416$, and the stride of the network is 32 (which means one grid cell can cover a square of $32 \times 32$ pixels in original image). The we have the dimensions of the grid of map $13 \times 13$.

<center>
    <img
    src=https://i.imgur.com/u3rXhm1.png
    width=500>
    <br>
    Taken from Ayoosh Kathuria's blog.
    <br></br>
</center>


Then, a cell (7, 7) marked red containing the center of the ground truth is chosen to be the one responsible for predicting... naively lol.

However, this is not the full story. In order to have better convergence during training. Researches and engineers usually employ greedy matching for encoding a ground truth to many possible anchors as long as their satisfied some criteria. 


### Encoding Ground Truth

The framework of anchor-based object detection requires ground truth to be **encoded** into target anchors (and yes, I will call them in this way :3). **Since only in this formalism, a model can "understand" the target**. This preprocessing is so crucial that if one mis-encodes target anchor, then a model will never get to learn correctly.

> As a side note, the number of target is defined to be usually greater than ground truth count, in this way, the learning process will be easier for the model.
> However, this could be tricky in implementation. Since for example, one target could have several related target anchors. One might need to set a threshold to gate/limit the number of target anchors. This threshold could be IoU or limit how many times of anchor width or height to account for ground truth.

Those target anchors encode the information of ground truth; an IoU of a target anchor and a ground truth is encoded as objectness of the target anchor. A offset between a target anchor and a ground truth is also encoded into target anchor.

### Bounding Box Regression

Let's take more about the coordinates encoding, since it is a little bit tricky. In YOLOv2 and hereafter, they all use the similar formula to
* encode a ground truth to a target anchor and
* decode a model output to a understandable result:

\begin{align}
b_x &= \sigma(t_x) + c_x, \\
b_y &= \sigma(t_y) + c_y, \\
b_w &= p_w e^{t_w}, \\
b_h &= p_h e^{t_h}. \\
\end{align}

:::info
In YOLOv4, in order to deal with the grid sensitivity, above equations are modified to 
\begin{align}
b_x &= \sigma(t_x) \times 1.1 - 0.05 + c_x, \\
b_y &= \sigma(t_y) \times 1.1 - 0.05 + c_y, \\
b_w &= p_w e^{t_w}, \\
b_h &= p_h e^{t_h}. \\
\end{align}
 In Scale-YOLOv4,
\begin{align}
b_x &= \sigma(t_x) \times 2 - 0.5 + c_x, \\
b_y &= \sigma(t_y) \times 2 - 0.5 + c_y, \\
b_w &= p_w (\sigma(t_w)\times 2)^2, \\
b_h &= p_h (\sigma(t_h)\times 2)^2. \\
\end{align}
:::

* $b_x, b_y, b_w, b_h$ are our final prediction (or ground truth). $x,y$ represent center coordinates, and $b_w, b_h$ are bounding box's width and height. These values are normalized by the original image width and height.
    * If $b_x, b_y = (0.5, 0.5)$ means that the bounding box center is in the center of the image.
        > one needs to normalize this by dividing by the grid size, e.g. for grid size $13 \times 13$, so the true $b_x$ would be 0.5/13.
* $t_x, t_y, t_w, t_h$ are the network outputs.
    * If ($\sigma(t_x), \sigma(t_y)) = (0.5, 0.5)$, it means in the center of grid cell (not the image). The so called "within the grid cell" is because $0 \leq \sigma \leq 1$.
* $c_x$ and $c_y$ are the absoulate top-left coordinates of the grid cell.
    * If $(c_x, c_y) = (2, 1)$, means the 3rd column and 2nd row in a grid cell of $N \times N$ (e.g. $13 \times 13$)
* $p_w$ and $p_h$ are anchors' width and height, which have specified height-width ratio. Note that the exponential is to account the negative value of $t_{x, y}$, since width and height cannot be negative valules.

<center>
    <img
    src=https://i.imgur.com/tlDhutn.png
    width=700>
    <br>
    I drew it :目
    <br></br>
</center>

### Selection on Prediction

For an image of size $416 \times 416$, YOLOv3 is capable of predicting $(52 \times 52 + 26 \times 26 + 13 \times 13) \times 3 = 10647$ bounding boxes. If we only has one object in the image. How do we reduce numerous during inference?

1. One will notice that many grid cells can provide a prediction to gt without good indication. It might have limit IoU with gt. That is why model should learn to give those anchors' objectness. With objectness score, we can apply a threshold to role out these predictions.
2. It is often the case that planety of grid cells around a ground truth (gt) can give predictions, and all of them have IoU above the threshold. This is when non-maximum-suppression (NMS) is applied to get the best bounding box.


### Class Confidences/Probabilities

They represent probabilities of detected objects belonging to a particular class (dog, cat, etc). **Before v3, YOLOs use softmax for outputing classes' scores.**

However, ***that design choice is dsicarded in v3, and authors have opted for using sigmoid instead.*** The reason is that softmaxing class scores assume that the classes are mutually exclusive. In simple words, if an object belongs to one class, then it's guaranteed it cannot belong to another class. This assumption is usually not suitable for real world problems.

### Scales

***Before YOLOv3, there is only 1 scale prediction. YOLOv3 makes prediction across 3 different scales.*** The head part consists of three different scales having strides 32, 16, and 8 respectively. This means, with an input of $416 \times 416$, we make detections on grid sizes $13 \times 13$, $26 \times 26$ and $52 \times 52$. Different input image resolution will have different detection scales.

<center>
    <img src = https://i.imgur.com/nqnaI0b.jpg
     width = 300>
     <br>
     Take from Ayoosh Kathuria blog
     <br></br>
</center>

After having a rough impression what are going to learn, let's dive deeper into their loss functions.

## YOLOv3 Loss
\begin{align}
\mathcal{L} = &\lambda_{\text{coor}} \sum I^{\text{obj}}_{ij} \text{MSE}\left((t_x, t_y), (t'_x, t'_y)\right)+
\lambda_{\text{coor}} \sum I^{\text{obj}}_{ij} \text{MSE}\left((t_w, t_h), (t'_w, t'_h)\right) + \\
&\sum I^{\text{obj}}_{ij} \text{BCE}(\text{obj, obj}') + 
\lambda_{\text{Noobj}} \sum I^{\text{Noobj}}_{ij} \text{BCE}(\text{obj, obj}') \times \text{ignore_mask} + \\
& \sum I^{\text{obj}}_{ij} \text{BCE} (\text{cls, cls}')
\end{align}

Anything with superscript $'$ are ground truth. Note that summation is shorthand of $\sum^{Scale}_{s=0}\sum^{S^2}_{i=0} \sum^{B}_{j=0}$ for first four terms, which sum over scales, grid cells and anchors with various height-width ratios. The last summation is to sum over different classes. $I^{\text{obj}}_{ij}$, which is like a mask, is 1 when there ground gruth in this cell, else 0; $I^{\text{Noobj}}_{ij} = 1 - I^{\text{obj}}_{ij}$, vice versa.

Therefore, ground truths come from annotation files need to be encoded like we discussed.

* For objectness loss, unlike YOLOv2, we will use binary cross-entropy instead of mean square error here.
* In the ground truth, objectness is always 1 for the cell that contains an object, and 0 for the cell that doesn’t contain any object.

By measuring the 3rd term, we can gradually teach the network to detect a region of interest. In the meantime, we don’t want the network to cheat by proposing objects everywhere. Hence, we need 4th term to penalize those false positive proposals. We get false positives by masking prediciton with $I^{\text{Noobj}}_{ij}$

* The $\text{ignore_mask}$ is used to make sure we only penalize when the current box doesn’t have much overlap with the ground truth box. If there is, we tend to be softer because it’s actually quite close to the answer.
* Since there are way too many $\text{noobj}$ than $\text{obj}$ in our ground truth, we also need  $\lambda_{\text{Noobj}} = 0.5$ to make sure the network won’t be dominated by cells that don’t have objects.

YOLOv3 predicts 3 bounding boxes for every cell (YOLOv4 as well). In v3, IoU $< 0.3$ is negative and $>0.7$ is positive target anchors. Those between 0.3 and 0.7 are not considered in the loss computation.

## YOLOv4 Loss
Compared with YOLOv3, YOLOv4 only made innovations in bounding box regression, replacing MSE with CIoU. No substantial changes in the class loss and objectness loss.

### CIoU loss

A loss function gives us signals on how to adjust weights to reduce cost. So in situations where we make wrong predictions, we expect it to give us direction on where to move. ***However, when IoU is used, this is not happening when a ground truth and a prediction does not overlap at all***. IoU loss is defined as following:

\begin{align}
\text{IoU} &= \frac{B \cap B^{gt}}{B \cup B^{gt}}, \\
\mathcal{L}_{\text{IoU}} &= 1 - \text{IoU},
\end{align}

where $B$ is the bounding boxand $gt$ is the ground truth. Generalized IoU (GIoU) fixes this by refining the loss as:

\begin{equation}
\mathcal{L}_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C - B \cup B^{gt}|}{|C|},
\end{equation}

where $C$ is the smallest box covering $B$ and $B^{gt}$.

When the $B$ is far away from $B^{gt}$, the last term will punish this situation. However, this loss function tends to expand $B$ first until it is overlapped with $B^{gt}$. Then $B$ shrinks to increase IoU. This process requires more iterations than theoretically needed.

To solve this issue, first, distance-IoU Loss (DIoU) is introduced as:

\begin{equation}
\mathcal{R}_{\text{DIoU}} = \frac{\rho (b, b^{gt})}{c^2},
\end{equation}

where $b$ and $b^{gt}$ denote the central points of $B$ and $B^{gt}$, $\rho()$ is the Euclidean distance, and $c$ is the diagonal length of $C$.

\begin{equation}
\mathcal{L}_{\text{DIoU}} = 1 - \text{IoU} + \mathcal{R}_{\text{DIoU}}
\end{equation}


<center>
    <img src =https://i.imgur.com/ZQxH9GK.png
     width= 300>
    <br>
    <br>
</center>

It introduces a new goal to reduce the central points separation between the two boxes.

Finally, Complete IoU Loss (CIoU) is introduced to:

* increase the overlapping area of the ground truth box and the predicted box,
* minimize their central point distance, and
* maintain the consistency of the boxes’ aspect ratio.

This is the final definition:

<center>
    <img src =https://i.imgur.com/bjkHInK.png
     width=600>
    <br>
    Take from Zhaohui Zheng et al. 2019
    <br><br>
</center>

$\alpha$ is a weighting and $\nu$ is used as measure the similarity of height and width ratio. The benefit of this particular design can be understood by inspecting their gradients:

\begin{align}
    \frac{\partial \nu}{\partial w}
    &= \frac{8}{\pi^2}
        \left(
            \arctan \frac{w^{gt}}{h^{gt}} -
            \arctan\frac{w}{h}
        \right) \times
        \frac{-h}{h^2 + w^2}
    \\
    \frac{\partial \nu}{\partial h}
    &= \frac{8}{\pi^2}
        \left(
            \arctan \frac{w^{gt}}{h^{gt}} -
            \arctan\frac{w}{h}
        \right) \times
        \frac{w}{h^2 + w^2}
\end{align}

One can see that $\arctan$ functions, which stands for the height and width ratio, still remain after the differentiation. Therefore, the ratio difference still have its effect in the gradients due to the fact of power 2 in the loss. ***The reason of chosing $\arctan$ function is to surpress those with very large ratio cases in to the range of $\pi/2$ in order to avoid divergent problem.***

After finishing YOLOv3/v4 loss, let's check the evolution of loss function in various one stage model before them.

## Historical Review

Note that here I only give the part of loss function is unique and inspiring, not the whole loss function.

### SSD Loss
Check [OHEM](https://hackmd.io/tGIMAyl0RqatLA3WBGoDlA?view).

### YOLOv1 Loss

The following is the head part in YOLOv1 (for 30 classes)

<center>
    <img src =https://i.imgur.com/yqOau2o.png
     width=500>
    <br>
    Take from Carol Hsin blog
    <br><br>
</center>

In YOLOv1, $S = 7$, $B = 2$ and $C = 20$. The shape of this output layer is $S^2 \times (5B + C)$. Do notice that, before the head part, there is fully-connect (FC) layers before it. Therefore, there is no so called anchor idea in v1, since FC will break the spatial information.

YOLOv1 loss design is:

<center>
    <img src =https://i.imgur.com/31ROxqD.png
     width=500>
    <br>
    Take from Joseph Redmon 2016
    <br><br>
</center>

where $𝟙^{obj}_{ij}$ is equal to $1$ if the $j$-th bounding box of the $i$-th grid cell is responsible for object, otherwise $0$. $S^2$ is the grid cell. Each term in the above formula is enumerated as below:

1. The bounding box $x$ and $y$ coordinates is parametrized to be offsets of a particular grid cell location so they are also bounded between 0 and 1. And the sum of square error (SSE) is estimated only when there is object. $\lambda_{\text{coord}} = 5$ is for increasing the loss for bounding box prediction.

2. The bounding box width and height are normalized by the image width and height so that they fall between 0 and 1. SSE is estimated only when there is object. Since small deviations in large boxes matter less than in small boxes. square root of the bounding box width $w$ and height $h$ instead of the width and height directly to partially address this problem.

3. The third one penalizes the objectness score (i.e. the probability of whether there is an object multiple the IoU between the predicted box and any ground truth box) prediction for bounding boxes responsible for predicting objects (the scores for these should ideally be 1).

4. The fourth one for bounding boxes having no objects, (the scores should ideally be zero). In every image, many grid cells do not contain any object. This pushes the “confidence” scores of those cells towards zero. This procedure often overpowers the gradient from cells that do contain objects, and makes the model unstable. Thus, the loss from confidence predictions for boxes that don’t contain objects, is decreased, i.e. $\lambda_{\text{noobj}} = 0.5$.

5. the last one penalises the class prediction for the bounding box which predicts the objects.


### YOLOv2 Loss

Overall, v2 loss is similar to v1, let me show it again:
<center>
    <img src =https://i.imgur.com/31ROxqD.png
     width=500>
    <br>
    Take from Joseph Redmon 2016
    <br><br>
</center>

However, notice that v2 decouples the class prediction mechanism from the spatial location. Instead, it predicts class and objectness for every anchor. In the other words, the entities in the channel dimension of a grid cell is changed from $5B + C$ to $B \times (5 + C)$. Furthermore, v2 adopts the idea of anchor into the framework from RPN of Faster-RCNN.

The v2 loss changes $B$ from 2 to 5 (or 9). $S$ changed from 7 to 13. The anchor count change from $7\times 7 \times 2 = 98$ to $13 \times 13 \times 5 (9) = 845(1521)$. All $\lambda$'s are the same to YOLOv1. In v2, target anchors are defined when their IoU above 0.3 with ground truth.


### RetinaNet Loss

:::info
Just a side note that for anchors $< 0.4$ IoU is negative and 0.5 is positive. Those with $0.4 \sim 0.5$ are considered as ignore class.
:::

The focal loss is defined as: 

\begin{align}
    \mathrm{FL}(y, p) = - \alpha_t (1 - p_t)^\gamma \log p_t,
\end{align}

where $\alpha_t$ is a class weight for positive and negative class, and $p_t$ is the probability of positive class $y=1$.

\begin{align}
    \alpha_t =
    \begin{cases}
        \alpha,
        \mathrm{\,\,\,\,\,\,\,\,\,\,\,\,\,\, if \,\,\,} y = 1 \\
        1 - \alpha,
        \mathrm{\,\,\, otherwise}
    \end{cases}
    \,\,\,\,\,\,\,\,\,\,
    p_t =
    \begin{cases}
        p,
        \mathrm{\,\,\,\,\,\,\,\,\,\,\,\,\,\, if \,\,\,} y = 1 \\
        1 - p,
        \mathrm{\,\,\, otherwise}
    \end{cases}
\end{align}

The hyperparameters are chose to be $\alpha=0.25$ and $\gamma=2$. When $\gamma = 0$, the loss becomes the normal cross entropy. The factor $- \alpha_t (1 - p_t)^\gamma$ give less attension (smaller values) to those well classified case (easy example, $p_t > 0.6$).

<center>
    <img src=https://i.imgur.com/rvGRh5g.png
     width=400>
     <br>
     Taken from Tsung-Yi Lin et al. 2017
</center>

## Backlog

Well, these notes take longer than imagine. Hope you guys enjoy the content so far. Stay tuned for the new content in the future :i_love_you_hand_sign:


## Reference

##### Related Note
* [Summary of One Stage Model Structure](https://hackmd.io/LYDus2SFSLu2UjXTQJWctg?view)
* [OHEM](https://hackmd.io/tGIMAyl0RqatLA3WBGoDlA?view)
* [FCOS](https://hackmd.io/IX9MuFDwQXSYPDaeciwO8w)


##### Blog
* [Review: YOLOv1 — You Only Look Once (Object Detection)](https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89)
* [YOLOv2--論文學習筆記（算法詳解）](https://www.twblogs.net/a/5bafd96b2b7177781a0f6394)
* [Focal Loss(RetinaNet) 與 OHEM](https://www.itread01.com/content/1543549147.html)
* [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 1](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
* [Dive Really Deep into YOLO v3: A Beginner’s Guide](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)
* [睿智的目标检测30——Pytorch搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106214657#1Backbone_47)
* [Yolo Object Detectors: Final Layers and Loss Functions](https://medium.com/oracledevs/final-layers-and-loss-functions-of-single-stage-detectors-part-1-4abbfa9aa71c)

##### Paper
* [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/pdf/1911.08287.pdf)
