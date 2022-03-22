from transformers import pipeline

generator = pipeline("text-generation")
generated_texts = generator("In this course, we will teach you how to")

print(generated_texts)