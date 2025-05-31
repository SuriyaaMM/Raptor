---
tags:
- financial
- transformer
library_name: pytorch
---
# Suriyaa-MM/raptor-fpredictor

This is a custom Financial Transformer model trained on historical stock data.
More details about the model architecture and training process can be found in the associated GitHub repository.
## Usage
```python
import torch
from huggingface_hub import hf_hub_download
from your_module import ftransformer # Assuming your ftransformer class is available

model_path = hf_hub_download(repo_id='{repo_id}', filename='pytorch_model.pth')
model = ftransformer(input_dim=<YOUR_INPUT_DIM>, num_classes=3) # Replace <YOUR_INPUT_DIM>
model.load_state_dict(torch.load(model_path))
model.eval()
print('Model loaded successfully!')
```