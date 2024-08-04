# Llama

[LLaMA](https://arxiv.org/abs/2302.13971) is a foundation language model from Meta.
It is fully open-sourced.

## Llama 3

- 8B and 70B models are available, and 400B model is coming soon (24.06.16)
    - [Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) is published already, and 405B model is out (24.08.01)
- The tokenizer of LLaMA 3 trained with 128K tokens, where LLaMA 2 tokenizer was trained with 32K tokens
- Context window is 8192 tokens, where LLaMA 2 is 4096 tokens and LLaMA 1 is 2048 tokens
- Uses grouped query attention, which is more efficient than the standard multi-head attention

### Llama 3.1

- 8B, 70B, and 405B models are available
- Llama 3.1 405B model actually as good as GPT4o and Claude 3.5 Sonnet
- Llama 3.1 70B outperforms GPT3.5-turbo and Mixtral-8x22B
- Llama 3.1 8B outperforms Gemma 2 9B and Mistral 7B instruct

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
- [Introducing Llama 3.1: Our most capable models to date](https://ai.meta.com/blog/meta-llama-3-1/)
