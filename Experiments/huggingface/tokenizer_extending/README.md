# Extending existing AutoTokenizer with new tokens

Extending existing tokenizer with new tokens from custom dataset does not always work.
Also, once you extend the tokenizer, you really should fine-tune the model on the new dataset with the new tokens.

Basically, extending the tokenizer increases the size of the embedding matrix, which will add newly initialized vectors at the end.
Using these new embeddings untrained might already be useful, but usually at least some steps of fine-tuning are required.

## Codes

1. [Extending tokenizer with new tokens](./extend_tokenizer_with_new_words.py)
2. [Adding emojis to tokenizer](./add_emojis_to_tokenizer.py) : Basically, print emojis as unicode characters to file, and then read them back in as tokens.
3. [Adding multi-word expressions to tokenizer](./add_multiword_expressions_to_tokenizer.py)
4. [Extending existing AutoTokenizer with new bpe-tokenized tokens](./extend_tokenizer_with_new_bpe_tokens.py)
5. [Adding vocabs from one tokenizer to another](./adding_vocabs_from_tokenizer_to_another.py)

## References

* [How to add new tokens to huggingface transformers vocabulary](https://www.depends-on-the-definition.com/how-to-add-new-tokens-to-huggingface-transformers/)
* [Extending existing AutoTokenizer with new tokens](https://stackoverflow.com/a/76198053/9012940)
