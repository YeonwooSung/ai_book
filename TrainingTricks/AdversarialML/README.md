# Adversarial Machine Learning

Adversarial machine learning is the study of the attacks on machine learning algorithms, and of the defenses against such attacks. A recent survey exposes the fact that practitioners report a dire need for better protecting machine learning systems in industrial applications.

## Adversarial attacks

Depending on what you as the attacker know about the model you wish to fool, there are several categories of attacks. The two most popular are the white box attack and black box attack. 

Both of these attack categories have the general goal of fooling a neural network into making wrong predictions. They do it by adding hard-to-notice noise to the input of the network. 

What makes these two types of attacks different is your ability to gain access to the entire architecture of the model. With white box attacks, you have complete access to the architecture (weights), and the input and output of the model. 

Control over the model is lower with black box attacks, as you only have access to the input and output of the model. 

There are some goals that you might have in mind when performing either of these attacks: 

- misclassification, where you only want to drive the model into making wrong predictions without worrying about the class of the prediction,

- source / target misclassification, where your intention is to add noise to the image to push the model to predict a specific class.

## FGM (Fast Gradient Method)

The Fast Gradient Method (FGM) is a technique for performing adversarial attacks. It combines a white box approach with a misclassification goal. It tricks a neural network model into making wrong predictions.

The name makes it seem like a difficult thing to understand, but the FGSM attack is incredibly simple. It involves three steps in this order:

1. Calculate the loss after forward propagation,
2. Calculate the gradient with respect to the pixels of the image,
3. Nudge the pixels of the image ever so slightly in the direction of the calculated gradients that maximize the loss calculated above.

The first step, calculating the loss after forward propagation, is very common in machine learning projects. We use a negative likelihood loss function to estimate how close the prediction of our model is to the actual class. 

What is not common, is the calculation of the gradients with respect to the pixels of the image. When it comes to training neural networks, gradients are how you determine the direction in which to nudge your weights to reduce the loss value. 

Instead of doing that, here we adjust the input image pixels in the direction of the gradients to maximize the loss value.

When training neural networks, the most popular way of determining the direction in which to adjust a particular weight deep in the network (that is, the gradient of the loss function with respect to that particular weight) is by back-propagating the gradients from the start (output part) to the weight. 

The same concept applies here. We back-propagate the gradients from the output layer to the input image. 

In neural network training, in order to nudge the weights to decrease the loss value, we use this simple equation:

```
new_weights = old_weights – learning_rate * gradients
```

Again, we apply the same concept for FGSM, but we want to maximize the loss, so we nudge the pixel values of the image according to the equation below:

```
new_pixels = old_pixels + epsilon * gradients
```

### FGM for training better language models

Adversarial-training let us train networks with significantly improved resistance to adversarial attacks, thus improving robustness of neural networks.

FGM implementation with PyTorch:

```python
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}
```

Add FGM in training code:

```python
fgm = FGM(model)
for batch_input, batch_label in data:
    loss = model(batch_input, batch_label)
    loss.backward()  

    # adversarial training
    fgm.attack() 
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() 
    fgm.restore()  

    optimizer.step()
    model.zero_grad()
```
