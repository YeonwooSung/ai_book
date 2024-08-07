# Stable Diffusion - Image to Prompts

[Kaggle Competition](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/overview)

The goal of this competition is to reverse the typical direction of a generative text-to-image model: instead of generating an image from a text prompt, can you create a model which can predict the text prompt given a generated image? You will make predictions on a dataset containing a wide variety of (prompt, image) pairs generated by Stable Diffusion 2.0, in order to understand how reversible the latent relationship is.

## Things that work for me

- [CLIP Iterrogator](https://github.com/pharmapsychotic/clip-interrogator)
- [OFATransformers](https://github.com/OFA-Sys/OFA/blob/main/transformers.md)
- [Finetuning ViT](https://www.kaggle.com/code/neos960518/stable-diffusion-vit-baseline-train)
- [KNN-based zero shot inference](https://www.kaggle.com/code/neos960518/sdip-clip-knnregression-zeroshot-method)

[my solution](https://www.kaggle.com/code/neos960518/ensemble-clipinterrogator-ofa-vit?scriptVersionId=129516485)

### Future works

- [Generate various prompts with OpenAI ChatGPT](./src/chatgpt-generated-prompts.ipynb)

## Winning Solutions

- [1st Place Solution](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/411237)
- [2nd Place Solution](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/410606)
- [3rd Place Solution](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/410686)
