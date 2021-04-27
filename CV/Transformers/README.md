# Transformers for CV

## Table of Contents (Transformers for CV)

- [Axial-DeepLab](#axial-deepLab)
- [DETR](#detr)
- [Vision Transformers](#vision-transformers)

## Transformer based models for CV tasks

Below is a summary of key designs adopted in different variants of transformers for a representive set of CV applications.

![A summary of key designs adopted in different variants of transformers for a representive set of CV applications](./imgs/models_tasks.png)

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
The most interesting point is that this model is not the ConvNet-based model. As you know, almost all CV models are CNN based until now.
However, [Dosovitskiy et. al. [2]](https://arxiv.org/abs/2010.11929) showed that it is possible to use the Transformer-based models for the image classification.

![ViT architecture](./imgs/ViT_architecture.png)

What they did is 1) divide the input image into several patches, 2) input those patches to the CNN (ResNet) and get the feature maps, 3) flatten the feature map, 4) input the flattened feature maps to the Transformer encoder, and 5) use the classifier as a last layer.

Since they divide the input image into patches, they convert the "H x W x C" to "N * (P x P x C)", where H = height, W = width, C = channel, N = num of patches, and P is the size of the patch.

As you could imagine, when dividing the image into patches, we might loss the positional information in the original image. To overcome this issue, they adopted the positional encoding as below.

![ViT's positional encoding](./imgs/vit_positional_encoding.png)

As well as other Transformer based models, ViT also pretrains the model with huge dataset, and fine-tuning the model with small downstream task. It is well known that it is better to use the high resolution images for the fine-tuning, since it helps us to improve the accuracy.

## References

[1] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, Liang-Chieh Chen. [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/abs/2003.07853)

[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
