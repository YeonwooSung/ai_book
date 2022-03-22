from transformers import pipeline

classifier = pipeline("sentiment-analysis")
output = classifier("I've been waiting for a HuggingFace course my whole life.")

print(output)

outputs = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

print(outputs)
