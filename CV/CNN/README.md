# Convolutional Neural Networks

The CNN, which stands for Convolutional Neural Network, is a class of deep neural networks, most commonly applied to analyzing visual imagery.

## Table of Contents

1. [Origin of CNN](#origin-of-cnn)
2. [Using Conv Net for other categories](#using-conv-net-for-other-categories)
3. [Understanding limits of CNN](#understanding-limits-of-cnn)
4. [Implementation](#implementation)
5. [References](#references)

## Origin of CNN

In the 1950s and 1960s, Hubel and Wiesel showed that cat and monkey visual cortexes contain neurons that individually respond to small regions of the visual field. Provided the eyes are not moving, the region of visual space within which visual stimuli affect the firing of a single neuron is known as its receptive field. Neighboring cells have similar and overlapping receptive fields.

The [neocognitron [1]](https://link.springer.com/chapter/10.1007/978-3-642-46466-9_18), which was published few years before Yann LeCun et. al. published the first paper of CNN, is a hierarchical, multilayered artificial neural network proposed by Kunihiko Fukushima in 1979. This was inspired by the above-mentioned work of Hubel and Wiesel. The neocognitron introduced the two basic types of layers in CNNs: convolutional layers, and downsampling layers. A convolutional layer contains units whose receptive fields cover a patch of the previous layer. The weight vector (the set of adaptive parameters) of such a unit is often called a filter. Units can share filters. Downsampling layers contain units whose receptive fields cover patches of previous convolutional layers. Such a unit typically computes the average of the activations of the units in its patch. This downsampling helps to correctly classify objects in visual scenes even when the objects are shifted.

In 1989, a system to recognize hand-written ZIP code numbers involved convolutions in which the kernel coefficients had been laboriously hand designed was implemented. [Yann LeCun et. al. [2]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) used back-propagation to learn the convolution kernel coefficients directly from images of hand-written numbers. Learning was thus fully automatic, performed better than manual coefficient design, and was suited to a broader range of image recognition problems and image types. There is no doubt that this approach became a foundation of modern Computer Vision.

## Using Conv Net for other categories

Some people might think that the CNN is only for the Computer Vision. However, that is not true. For example, in 2014, [Y. Kim et. al. [3]](https://arxiv.org/abs/1408.5882) used the Conv Net for sentence classification. In 2020, there are many NLP papers that use Convolutional Neural Networks to solve natural language problems. Furthermore, networks like [WaveNet [4]](https://arxiv.org/abs/1408.5882) or [MidiNet [5]](https://arxiv.org/abs/1703.10847) use the Convolutional Layers for music generation models.

## Understanding limits of CNN

In AAAI 2020 conference, Geofferey Hinton mentioned that "CNNs are designed to cope with translations" [6](https://bdtechtalks.com/2020/03/02/geoffrey-hinton-convnets-cnn-limits/). This means that a well-trained convnet can identify an object regardless of where it appears in an image. But they’re not so good at dealing with other effects of changing viewpoints such as rotation and scaling. One approach to solving this problem, according to Hinton, is to use 4D or 6D maps to train the AI and later perform object detection. “But that just gets hopelessly expensive,” he added.

## Implementation

I implemented various CNN architectures with PyTorch. You could find all codes from [my GitHub repository](https://github.com/YeonwooSung/PyTorch_CNN_Architectures).

## References

[1] Kunihiko Fukushima, Sei Miyake. [Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Visual Pattern Recognition](https://link.springer.com/chapter/10.1007/978-3-642-46466-9_18)

[2] Yann LeCun, Leon Bottou, Yoshua Bengio, Pattrick Haffner. [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

[3] Yoon Kim. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[4] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu. [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

[5] Li-Chia Yang, Szu-Yu Chou, Yi-Hsuan Yang. [MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation](https://arxiv.org/abs/1703.10847)

[6] Ben Dickson (TechTalks). ["Understanding the limits of CNNs, one of AI’s greatest achievements"](https://bdtechtalks.com/2020/03/02/geoffrey-hinton-convnets-cnn-limits/)
