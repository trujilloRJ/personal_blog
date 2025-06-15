+++
title = "Road segmentation with U-Net from monocular camera"
date = "2025-05-10T15:06:40+02:00"
author = "Javier Trujillo Rodriguez"
authorTwitter = "" #do not include @
cover = ""
tags = ["computer-vision", "opencv", "deep-learning"]
keywords = ["computer-vision", "automotive-perception", "unet", "deep learning"]
description = "This post explores road segmentation using a U-Net architecture trained on monocular images from the KITTI dataset. The task focuses on identifying drivable areas from RGB inputs, a challenge due to lighting, shadows, and variable road conditions. I detail the model architecture, training strategies, and key hyperparameters such as learning rate selection and loss functions."
showFullContent = false
readingTime = false
hideComments = false
math = true
+++

Understanding where a vehicle can safely drive is essential for building autonomous driving systems. This involves segmenting lane markings, road edges, and drivable areas using inputs from sensors like cameras, LiDAR, or radar.

In this post, I explore how to tackle this using just a monocular camera and a neural network based on the U-Net architecture, originally developed for medical image segmentation. Detecting drivable road surfaces from raw RGB images is challenging as the model needs to be robust to lighting variations, shadows, and road conditions. To improve the model’s ability to generalize, I used data augmentation techniques that help it handle a wider variety of scenes.

## Source code

The source code for this post is available at: https://github.com/trujilloRJ/lane-detection

## Table of Contents

1. [Dataset](#dataset)
2. [Model architecture](#model-architecture)
3. [Training details](#training-details)
4. [Results and discussion](#results-and-discussion)

# Dataset

We're working with data from the  [KITTI road detection challenge](https://www.cvlibs.net/datasets/kitti/eval_road.php), a benchmark dataset designed for evaluating road scene understanding in autonomous driving. It features real-world driving scenarios captured in both urban and highway environments.

Each image in the dataset comes with a corresponding binary mask that labels which pixels belong to the drivable road surface. In the example below, the road mask is overlaid in green. Notice how the mask highlights only the road intended for the ego-vehicle while excluding the opposite lane. This distinction is important: we’re not just detecting roads in general, we’re identifying where our vehicle can actually drive.

{{< rawhtml >}}
<image src="/posts/images/road_example.png" alt="Example lane" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The model outputs a pixel-wise classification map—basically, an image-sized grid where each pixel is labeled as either “road” or “not road.” In computer vision, this kind of task is called **semantic segmentation**. It’s a common approach not only in autonomous driving but also in fields like medical imaging, where it’s used to identify things like tumors in scans.

One architecture that has consistently shown strong performance in segmentation tasks is the [U-Net](https://arxiv.org/abs/1505.04597). Originally developed for biomedical image segmentation, U-Net’s encoder-decoder structure with skip connections makes it well-suited for capturing both high-level context and fine-grained details. In this post, we’ll see how this architecture can be effectively applied to automotive scenes as well.

# Model architecture

{{< rawhtml >}}
<image src="/posts/diagrams/UNet.png" alt="Example lane" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The figure shows the baseline architecture used for our prediction model, which is a simplified version of the U-Net. While the original U-Net includes four downsampling and upsampling blocks with a larger number of channels to handle multi-class segmentation, our version is lighter, using only three blocks and fewer channels. This reduction is sufficient for our task, as we only need to segment a single class—drivable road—rather than multiple classes.

A key and innovative feature of the U-Net architecture is the use of skip connections, illustrated in the figure with dashed lines. These connections link the output of each downsampling block directly to the corresponding upsampling block in the expanding path by concatenating their feature maps. This design helps preserve spatial information that is lost during downsampling.

# Training details

Before jumping into the results, I want to briefly go through the training process and methodology. Often, these details are overlooked in articles, and, in my opinion, they are critical to ensure good performance.

#### Loss function

To train the model effectively, we need a loss function that captures how much our predicted segmentation map differs from the ground truth. A natural starting point is the well-known Binary Cross-Entropy (BCE) loss, which works well for binary classification tasks.

However, in road segmentation, we run into a common problem: class imbalance. The drivable road typically takes up a much smaller portion of the image compared to the background. As a result, BCE can produce deceptively good results just by favoring the majority class, the background.

A better alternative for segmentation tasks is the Dice loss, which directly measures the overlap between the predicted road pixels and the ground truth. Mathematically, it’s a version of the Dice coefficient:

$$Dice = \frac{|X\cap Y|}{X + Y}$$
It’s particularly effective in imbalanced scenarios because it focuses on correctly predicting the minority class—in our case, the road pixels.

Finally, the loss used for training is a combination of BCE and Dice, known in the literature as **Combo** loss:

$$L = BCE + Dice$$

#### Batch size

Next, we need to select an appropriate mini-batch size. Smaller batches work well with limited GPU memory but can produce noisier gradient updates and slower convergence. However, noisy gradients aren't always a drawback—they can improve generalization by preventing the model from overfitting, which is especially beneficial when training on a small dataset, as in our case. Larger batch sizes offer more stable gradients and faster training but require more memory and can overfit if not carefully tuned. Given our limited dataset and GPU resources, we opted for a very small batch size of `2`.

#### Learning rate

Once the batch size is set, the next step is selecting the learning rate, arguably [the most important hyperparameter](https://www.deeplearningbook.org/). A common starting point is `3e-4`, known as "Karpathy’s constant," popularized by Andrej Karpathy in one of his [tweets](https://x.com/karpathy/status/933763912045965312).

However, we could use a more informed approach: the learning rate range test. Proposed by researcher Leslie Smith in his well-known [paper](https://arxiv.org/abs/1506.01186), this method involves starting with a very low learning rate and gradually increasing it after each batch iteration. By plotting the training loss against the learning rate, we can observe where the loss begins to decrease sharply and where it starts to rise again. The optimal learning rate typically lies just before the loss begins to increase.

{{< rawhtml >}}
<image src="/posts/images/lr_range_test.png" alt="Example lane" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The figure shows the result of the LR range test for our dataset. As observed, the training loss decreases steadily and then begins to rise sharply after a certain value around `0.1`. Based on this curve, we selected a starting learning rate of `1e-4` (very close to the empirical Karpathy's constant 😊), which lies just before the loss starts to diverge. This choice tries to maximize learning efficiency while maintaining training stability.

#### Optimizer

For optimization, I choose **Adam**. While Stochastic Gradient Descent (SGD) with momentum can outperform Adam when finely tuned, Adam is more robust to suboptimal hyperparameters and generally performs well out of the box. Since I’m not running extensive hyperparameter searches, Adam is the practical choice for this experiment.

#### Scheduler

Finally, let’s talk about learning rate schedulers. For this experiment, I’m using the **One-Cycle** policy, introduced in the [super convergence paper](https://arxiv.org/abs/1708.07120) by Leslie Smith. The idea is to start training with a gradually increasing learning rate (a “warm-up”), followed by a steady decrease to very low values.

The intuition is that high learning rates early on help the optimizer escape sharp, narrow minima, encouraging convergence toward wider, flatter ones—which are known to generalize better. By finishing with a low learning rate, the model can fine-tune within that broad minimum, improving its robustness to unseen data. The learning rate range test give us insights to select the maximum learning rate for the One-Cycle policy. I choose `1e-3` as a sensible value before the loss starts increasing.

The following table summarizes the training parameter selection:
| Parameter | Value 
| ----- | ---- 
| Loss  | Combo (BCE + Dice) 
| Batch size  | 2 
| Initial learning rate | 1e-4 
| Optimizer | AdamW 
| Scheduler | One-Cycle policy 

# Results and discussion

#### Baseline model

Starting with a simple baseline model helps us understand the problem better and provides a foundation to improve upon. For this, we used just 2 downsampling and upsampling blocks with feature channels `(32, 64, 128)` in the contracting and expanding paths.

{{< rawhtml >}}
<image src="/posts/images/loss_fn_2d.png" alt="Loss" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The figure shows training and validation loss over epochs. Despite some noisy validation loss early on—likely due to the small batch size and one-cycle scheduler—the model converges well. The best validation loss occurs at epoch `65`, and the model parameters from this point were saved for evaluation.

Loss doesn’t fully reflect model performance, we need a more interpretable metric, so we also consider precision and recall. The ROC curve below for the best model iteration shows promising results for our baseline model—with both precision and recall exceeding `90%`.

{{< rawhtml >}}
<image src="/posts/images/auc_2d.png" alt="Loss" position="center" style="border-radius: 8px; width: 400px" >
{{< /rawhtml >}}

Let's continue with the evaluation by looking at two examples.

{{< rawhtml >}}
<image src="/posts/images/good_2d.png" alt="ex00" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}
{{< rawhtml >}}
<image src="/posts/images/bad_2d.png" alt="ex02" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The first example illustrates a scene where the network accurately predicts the drivable road. In contrast, the second example features two roads, but only one is drivable. Here, the network correctly identifies most of the drivable road but mistakenly classifies the opposite road as drivable and misses some pixels on the actual road.

#### Extending our model

Let’s take it a step further by increasing the model depth—adding one more downsampling and upsampling block. This expands the feature channels to `(32, 64, 128, 256)`. The goal is that a deeper U-Net with a larger receptive field will deliver better predictions than the baseline model.

The plot below compares the loss of the new deeper model against the baseline shallow one. It’s clear that the deeper model converges faster and reaches a significantly lower validation loss by epoch `59`.

{{< rawhtml >}}
<image src="/posts/images/compare_loss.png" alt="Loss" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

Performance-wise, the improvements are clear in the same example as before. The deeper model delivers a much more accurate road prediction and avoids confusing the opposite, non-drivable lane. 

{{< rawhtml >}}
<image src="/posts/images/good_3d.png" alt="ex02" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

These results are also valid to the whole validation dataset with a higher ROC Area Under the Curve score:

{{< rawhtml >}}
<image src="/posts/images/compare_roc.png" alt="compare_roc" position="center" style="border-radius: 8px; width: 400px" >
{{< /rawhtml >}}

#### Data augmentation

While reviewing some challenging cases, I noticed the model struggled with variable lighting conditions that altered the road’s color from the usual appearance. This suggests the model was relying too heavily on color cues to detect roads. A good way to address this is to expand the training data with varied lighting scenarios, encouraging the model to focus more on shape than color. We can simulate this by applying data augmentation techniques.

For this, I used the library [**albumentations**](https://albumentations.ai/) as it offers a wide range of transformations like scaling, rotation, color shifting and many more that comes handy for computer visions tasks. In adittion, it integrates easily with Pytorch. Specifically, applying random color shifting transformations to our data can help the model become more robust to varying lighting conditions and improve its generalization.

{{< rawhtml >}}
<image src="/posts/images/compare_loss_aug.png" alt="compare_aug" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The figure shows the loss evolution for the deeper model with and without data augmentation. Including augmentation leads to a slightly lower validation loss. Interestingly, the training loss at the point of lowest validation loss is higher with augmentation, indicating less overfitting and better generalization—exactly what we want from data augmentation.

Here’s one of the toughest examples from the dataset. The top image shows predictions from the deeper model trained without data augmentation, while the bottom shows the same model trained with augmentation.

Without augmentation, the model struggles with low recall—missing much of the drivable road, likely due to heavy shadows cast by nearby buildings. It even mistakes parts of the building for road, since the color and shape features are similar.

With data augmentation, recall improves significantly, showing better robustness to lighting changes. However, the model still confuses the building for drivable road in some areas.

This challenge remains open for now, but potential solutions include (i) experimenting with more diverse data augmentations and longer training, and (ii) increasing model depth to capture more complex features.

{{< rawhtml >}}
<image src="/posts/images/bad_3d.png" alt="ex00" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}
{{< rawhtml >}}
<image src="/posts/images/good_aug.png" alt="ex02" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

#### Model size

Model size and memory requirements are crucial factors, especially when deploying to platforms with limited resources or when real-time inference is needed. Our baseline "shallow" U-Net, using feature channels `(32, 64, 128)`, contains approximately 260K trainable parameters. In contrast, the deeper model—with an added downsampling/upsampling block and expanded channels `(32, 64, 128, 256)`—has around 1 million parameters, roughly **4× larger**.

This difference in size has practical implications. While the deeper model offers better performance, it demands significantly more memory and computation. On edge devices or embedded systems where latency, power consumption, or memory constraints are critical, the smaller model may be preferred—even if it means sacrificing some segmentation accuracy. Choosing the right model becomes a trade-off between performance and efficiency, depending on the target deployment scenario.

Here’s a final touch for some flair — a GIF showcasing the model in action on various road scenes:

{{< rawhtml >}}
<image src="/posts/images/road_detection_examples.gif" alt="examples" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

