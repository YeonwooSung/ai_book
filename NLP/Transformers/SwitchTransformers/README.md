# Switch Transformers

## Mixture of Experts (MoE)

The key idea is to make each expert focus on predicting the right answer for the cases where it is already doing better than the other experts

  - If we always average all the predictors, each model is trying to compensate for the combined error made by all the other models

## Switch Transformers

Scale is the next frontier for AI. Google Brain uses sparsity and hard routing to massively increase a model's parameters, while keeping the FLOPs per forward pass constant. 
The [Switch Transformer [1]](https://arxiv.org/abs/2101.03961) compares favorably to its dense counterparts in terms of speed and sample efficiency and breaks the next magic number: One Trillion Parameters.

[codebase](https://github.com/google-research/text-to-text-transfer-transformer)

### Abstract

In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) defies this and instead selects different parameters for each incoming example. The result is a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.

## References

[1] William Fedus, Barret Zoph, Noam Shazeer. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
