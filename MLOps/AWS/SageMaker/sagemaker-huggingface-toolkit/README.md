# SageMaker Hugging Face Inference Toolkit

SageMaker Hugging Face Inference Toolkit is an open-source library for serving HuggingFace Transformers models on Amazon SageMaker. This library provides default pre-processing, predict and postprocessing for certain HuggingFace Transformers models and tasks.

For more information, please visit the official [SageMaker Hugging Face Inference Toolkit repo](https://github.com/aws/sagemaker-huggingface-inference-toolkit).

## Getting Started

### Installation

```sh
$ pip install sagemaker --upgrade
```

### Create a Amazon SageMaker endpoint with a trained model.

Before running this example, make sure that you have trained a model and uploaded it to Amazon S3.

```python
from sagemaker.huggingface import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.6',
    pytorch_version='1.7',
    py_version='py36',
    model_data='s3://my-trained-model/artifacts/model.tar.gz',
    role=role,
)
# deploy model to SageMaker Inference
huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")
```

### Create a Amazon SageMaker endpoint with a model from the HuggingFace Hub.

```python
from sagemaker.huggingface import HuggingFaceModel
# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad',
  'HF_TASK':'question-answering'
}
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.6',
    pytorch_version='1.7',
    py_version='py36',
    env=hub,
    role=role,
)
# deploy model to SageMaker Inference
huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")
```
