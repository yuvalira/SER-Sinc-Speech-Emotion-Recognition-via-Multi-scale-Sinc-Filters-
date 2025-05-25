# Dataset: SER-RAVDESS-Augmented

This project uses the dataset hosted on the Hugging Face Hub:

https://huggingface.co/datasets/yuvalira/SER-RAVDESS-Augmented

## How to Use

To load the dataset in your Python code using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("yuvalira/SER-RAVDESS-Augmented")
```

If you need to access a specific file (for example, `train/train.pt`) directly:

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="yuvalira/SER-RAVDESS-Augmented",
    filename="train/train.pt"
)
```

## Requirements

Make sure to install the required packages:

```bash
pip install datasets huggingface_hub
```

If the dataset is private, you will need to authenticate:

```bash
huggingface-cli login
```
