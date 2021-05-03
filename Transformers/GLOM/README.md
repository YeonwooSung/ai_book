# GLOM

## Object Recongition as parse tree

Let's assume that there is an image of a car. A car has cabin, motor, and wheels. If we represent this as a parse tree, then the root node will be a car, and it has 3 children nodes: nodes for 1) cabin, 2) motor, and 3) wheels. A cabin includes windows and door area, so the cabin node will have 2 children nodes: window node and door area node.

![Car - Object Recongition as parse tree](car_image_for_object_recognition_as_parse_tree.png)

So, the main object of the GLOM is to build a parse tree when the image is given, and the neural net will fully understand the content of the image if it could build the parse tree correctly. If the AI could build a suitable parse tree for the input image, then the AI could recognise the vision data just like a human recognise the world.

## Summary

Geoffrey Hinton describes [GLOM [1]](https://arxiv.org/abs/2102.12627), a Computer Vision model that combines transformers, neural fields, contrastive learning, capsule networks, denoising autoencoders and RNNs. GLOM decomposes an image into a parse tree of objects and their parts. However, unlike previous systems, the parse tree is constructed dynamically and differently for each input, without changing the underlying neural network. This is done by a multi-step consensus algorithm that runs over different levels of abstraction at each location of an image simultaneously. GLOM is just an idea for now but suggests a radically new approach to AI visual scene understanding.

## References

[1] Geoffrey Hinton. [How to represent part-whole hierarchies in a neural network](https://arxiv.org/abs/2102.12627)
