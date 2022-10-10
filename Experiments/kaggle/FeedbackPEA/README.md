# Feedback Prize - Predicting Effective Arguments

The goal of this competition is to classify argumentative elements in student writing as "effective," "adequate," or "ineffective."

[Helpful repo](https://github.com/Zumo09/Feedback-Prize)

## Data

### Preprocessing

One of the most interesting strategiesf was the preprocessing strategy that is used by [Raja Biswas](https://www.kaggle.com/conjuring92). Basically, what he did was pre-processed each essay to insert newly added span start and span end tokens for each discourse types. To provide more context to our models, he added a topic segment in this format [TOPIC] <prompt_text> [TOPIC END] <essay>. As a minor detail, we inserted [SOE] and [EOE] tokens to indicate essay start and end. As an illustration here is an example:

```
[TOPIC] Should computers read the emotional expressions of students in a classroom? [TOPIC_END] [SOE] Lead [LEAD] Imagine being told about a new new technology that reads emotions. Now imagine this, imagine being able to tell how someone truly feels about their emotional expressions because they don't really know how they really feel. [LEAD_END]  In this essay I am going to tell you about my opinions on why  Position [POSITION] I believe that the value of using technology to read student's emotional expressions is over the top. [POSITION_END] 

...

Concluding Statement [CONCLUDING_STATEMENT] In conclusion I think this is not a very good idea, but not in the way students should be watched. The way teachers are able to read students' facial expressions tells them how they feel. I don't believe it's important to look at students faces when they can fake their emotions. But, if the teacher is watching you then they're gonna get angry. This is how I feel on this topic. [CONCLUDING_STATEMENT_END] [EOE]
```

### Data Augmentation

Some used T5 to augment the data.

i.e.
```
Input:
    Generate text for <discourse effectiveness> <discourse type> | Prompt: <prompt text> | Left Context: <left context> | Right Context: <right context>

Output:
    <discourse text>


Example Input:

Generate text for Ineffective Evidence | Prompt: Should the electoral college be abolished in favor of popular vote? | Left Context:Dear, State Senator\n\nWith the elctoral college vote most of people are not getting what they prefer. For the the electoral college vote, voters vote fro not the president, but for a not slate of electors. <....> majority of the people would rather have the popular vote.| Right Context: the electoral college election consist of the 538 electors. All you have to do is win most of those votes and you win the election. <.....> The electoral college is unfair, outdated, and irrational to all the people

Example Output:

It does not make any sense to me how the president with the votes does not win the presidential election. The electoral college is unfair for all voters that vote. When you think of a vote and how it works you would think by the most one with the most votes wins cause that pretty much means that most of the people would rather have the most wins over the win that had less votes but more electoral votes.
```

## Model

Deberta worked the best since (IMO) it supports unlimited input length and uses disentangled attention with relative positional embedding. Some of the high scoring competitors added GRU/LSTM with a pooling layer as a head model to extract features from the Deberta output.

### Pretraining

Some of the high scoring competitors mentioned that they pretrained the Deberta model on the training data. Basically, they trained the Deberta model with MLM with essay dataset. One of the main reasons that the high scorers pretrained the Deberta model is that they added some special tokens (such as [LEAD], [POSITION]). Also, by pretraining the Deberta model with the training dataset, they could make their model to fully understand the vocabularies that are used in the training dataset. During pretraining, mostly used 800-ish as the max length.

### Max Length

The length of the longest text data in the training dataset was around 1800, which is way longer than the default max length of BERT variant models. Some people just used 1800 as the max length, and some used 1280 since 99% of the data in training set are shorter than 1280.

### Fine-tuning

Below are the strategies that the high scorers used for fine-tuning.

- Adversarial Training with AWP
- Multi-sample dropouts
- Multi-head attention over span representations
- LSTM over transformer encoder representations
- Loss = CE Loss + Focal loss
- Cosine scheduler
- Layer wise learning rate decay
- Mean pooling to extract span representations as opposed to [CLS] token based ones

#### Multi-sample dropouts

[Multi-sample dropout](https://arxiv.org/abs/1905.09788) is a technique that creates multiple dropout samples. The loss is calculated for each sample, and then the sample losses are averaged to obtain the final loss. This technique can be easily implemented by duplicating a part of the network after the dropout layer while sharing the weights among the duplicated fully connected layers.

[Notebook for training transformer model with Multi-sample dropout](./src/Feedback%20training.ipynb).
