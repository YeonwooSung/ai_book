from transformers import pipeline

generator = pipeline("text-generation")
generated_texts = generator("In this course, we will teach you how to")

print(generated_texts)

generator = pipeline("text-generation", model="distilgpt2")
generated_texts_distill_gpt = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(generated_texts_distill_gpt)
