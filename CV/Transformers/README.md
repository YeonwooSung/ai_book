# Transformers for CV

## Table of Contents (Transformers for CV)

1. [Axial-DeepLab](#axial-deepLab)
2. [DETR](#detr)
3. [Vision Transformers](#vision-transformers)

## Axial-DeepLab

Convolutional Neural Networks have dominated image processing for the last decade, but transformers are quickly replacing traditional models. [Wang et al. [1]](https://arxiv.org/abs/2003.07853) proposes a fully attentional model for images by combining learned Positional Embeddings with Axial Attention. This new model can compete with CNNs on image classification and achieve state-of-the-art in various image segmentation tasks.

The key architiecture in this model is "Axial-Attention" block, which uses the "position sensitive attention". Wang et al. stated that the position sensitive attention  captures long range interactions with precise positional information at a reasonable computation overhead.

![Axial-Deeplab 1](./imgs/axial-deeplab1.png)

![Axial-Deeplab 2](./imgs/axial-deeplab2.png)

![Axial-Deeplab 3](./imgs/axial-deeplab3.png)

Surprisingly, the Axial-DeepLab worked well on not only for image classification, but also as a backbone of panoptic segmentation. For the image classification, it actually got similar accuracy score with other SOTA models that use ConvNet layers. Furthermore, it achieved SOTA for various segmentation tasks.

## DETR

On 2019, FAIR (Facebook AI Research) proposed a novel model called DETR, which uses the Transfomers model for the object detection.

Please see more information about the DETR in [here](../ObjectDetection/DETR)

## Vision Transformers

ViT (Vision Transformers) is an architecture that is proposed by Google, which is a Transformer-based model for CV tasks.
The most interesting point is that this model is not the ConvNet-based model. As you know, most CV models are CNN based until now.
However, [Dosovitskiy et. al. [2]](https://arxiv.org/abs/2010.11929) showed that it is possible to use pure Transformer models for CV tasks.

## References

[1] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, Liang-Chieh Chen. [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/abs/2003.07853)

[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
