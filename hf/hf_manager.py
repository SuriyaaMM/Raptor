from huggingface_hub import HfApi
import foundation

api = HfApi()
repo_id = "Suriyaa-MM/raptor-fpredictor"
model_save_path = foundation.env.model_transformer_path

api.create_repo(repo_id=repo_id, private=False, exist_ok=True, token = foundation.env.HF_TOKEN)
foundation.log(f"repository {repo_id} created or already exists.", "hf_manager")

# Upload the .pth file
api.upload_file(
    path_or_fileobj=model_save_path,
    path_in_repo="pytorch_model.pth",
    repo_id=repo_id,
    repo_type="model",
    token = foundation.env.HF_TOKEN
)
foundation.log(f"uploaded {model_save_path} to {repo_id}/pytorch_model.pth", "hf_manager")

# readme
with open("hf/README.md", "w") as f:
    f.write("---\n")
    f.write(f"tags:\n- financial\n- transformer\nlibrary_name: pytorch\n")
    f.write("---\n")
    f.write(f"# {repo_id}\n\n")
    f.write("This is a custom Financial Transformer model trained on historical stock data.\n")
    f.write("More details about the model architecture and training process can be found in the associated GitHub repository.\n")
    f.write("## Usage\n")
    f.write("```python\n")
    f.write("import torch\n")
    f.write("from huggingface_hub import hf_hub_download\n")
    f.write("from your_module import ftransformer # Assuming your ftransformer class is available\n\n")
    f.write("model_path = hf_hub_download(repo_id='{repo_id}', filename='pytorch_model.pth')\n")
    f.write("model = ftransformer(input_dim=<YOUR_INPUT_DIM>, num_classes=3) # Replace <YOUR_INPUT_DIM>\n")
    f.write("model.load_state_dict(torch.load(model_path))\n")
    f.write("model.eval()\n")
    f.write("print('Model loaded successfully!')\n")
    f.write("```\n")

api.upload_file(
    path_or_fileobj="hf/README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    token = foundation.env.HF_TOKEN
)
foundation.log(f"uploaded hf/README.md", "hf_manager")