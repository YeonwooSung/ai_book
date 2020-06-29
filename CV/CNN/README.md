# Convolutional Neural Networks

The CNN, which stands for Convolutional Neural Network, is a class of deep neural networks, most commonly applied to analyzing visual imagery.

## Origin of CNN

In the 1950s and 1960s, Hubel and Wiesel showed that cat and monkey visual cortexes contain neurons that individually respond to small regions of the visual field. Provided the eyes are not moving, the region of visual space within which visual stimuli affect the firing of a single neuron is known as its receptive field. Neighboring cells have similar and overlapping receptive fields.

The [neocognitron [1]](https://link.springer.com/chapter/10.1007/978-3-642-46466-9_18), which was published few years before Yann LeCun et. al. published the first paper of CNN, is a hierarchical, multilayered artificial neural network proposed by Kunihiko Fukushima in 1979. This was inspired by the above-mentioned work of Hubel and Wiesel. The neocognitron introduced the two basic types of layers in CNNs: convolutional layers, and downsampling layers. A convolutional layer contains units whose receptive fields cover a patch of the previous layer. The weight vector (the set of adaptive parameters) of such a unit is often called a filter. Units can share filters. Downsampling layers contain units whose receptive fields cover patches of previous convolutional layers. Such a unit typically computes the average of the activations of the units in its patch. This downsampling helps to correctly classify objects in visual scenes even when the objects are shifted.

In 1989, a system to recognize hand-written ZIP code numbers involved convolutions in which the kernel coefficients had been laboriously hand designed was implemented. [Yann LeCun et. al. [2]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) used back-propagation to learn the convolution kernel coefficients directly from images of hand-written numbers. Learning was thus fully automatic, performed better than manual coefficient design, and was suited to a broader range of image recognition problems and image types. There is no doubt that this approach became a foundation of modern Computer Vision.

## References

[1] Kunihiko Fukushima, Sei Miyake. [Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Visual Pattern Recognition](https://link.springer.com/chapter/10.1007/978-3-642-46466-9_18)

[2] Yann LeCun ; Leon Bottou, Yoshua Bengio, Pattrick Haffner. [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
