---
layout: page
title: Best practices
subtitle: Practical advice for deep learning
use-site-title: true
---

## From the experts
Sometimes the deep learning models can fail and do not live up to their expectations leaving behind many disappointed ML practitioners. There might be several reasons of failure, but most of them can be avoided by carefully designing the network and the training procedure based on the insights of the experts who tackled these problems very early on. Here are some extremely useful resources:

- You should definitely start by reading Andrej Karpathy's blog [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) on the matter.

- In their blog post [Practical Advice for Building Deep Neural Networks](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/), Matt H and Daniel R provides very useful tips and tricks.

- Here is a [cheat sheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks) from Stanford, possibly an organized version of Karpathy's blog.

- In this [video](https://www.youtube.com/watch?v=5ygO8FxNB8c) from Full Stack Deep Learning, Xavier Amatriain discusses practical side of deep learning systems. There are other insightful talks at Full Stack Deep Learning's youtube channel. Do check them out.

## From the 'novice'
There were some concepts that I realized (or at least I felt like) very important in my professional work. Some of them are already discussed in the above resources, but I would like to emphasize them for the interested reader.

- In Convolutional Networks, downsampling is achieved by either **strided convolution** or **pooling layer**. However, their influence are not necessarily the same. For example, max pooling "introduces invariance to small spatial shifts" such as translation and rotation in the image making it more robust to these kinds of variations in the input images (source: Justin Johnson's lectures on Michigan Online).

- **Batch Normalization** is proved to be a very effective addition to CNNs, but it also causes the model to behave differently during training and inference. Because, during training the model uses batch statistics for normalization which is absent during inference. This violates the important requirement of any trained model: Ideally, the model should behave the same both during training and inference. There are other normalization methods which exhibit the same behavior during training and inference, such as Instance Normalization and Layer Normalization, and sometimes they are more effective.
