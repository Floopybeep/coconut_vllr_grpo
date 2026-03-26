#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from coconut_grpo_baseline_manual import Coconut
from utils import Config, set_seed

import yaml
import argparse
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
python -m infer_model_visualize --checkpoint checkpoints
/gsm-coconut-grpo-baseline/260307_031254/ref_models/eval_best_model_1_1600_0.25.pt ./args/gsm_vllr_grpo_coconut_baseline.yaml 
"""

plot_idx = 0

def visualize_latent_pca(all_latent_embeddings, title="Latent Token Embeddings (PCA)", colors=None, output_dir="./plots"):
    """
    Apply PCA to reduce to 3D and visualize, save to file.
    
    Args:
        all_latent_embeddings: (total_latent_tokens, hidden_dim) array from multiple prompts
        title: Title for the plot
        colors: (total_latent_tokens,) array or list of colors for each point
        output_dir: Directory to save the plot
    
    Returns:
        output_path: Path to the saved plot file
        pca: The fitted PCA model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fit PCA to 3 dimensions
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_latent_embeddings)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        colors = 'blue'
    
    # Draw lines connecting sequential latent tokens (thicker lines)
    for i in range(len(embeddings_3d) - 1):
        ax.plot(embeddings_3d[i:i+2, 0], embeddings_3d[i:i+2, 1], embeddings_3d[i:i+2, 2], 
                'gray', alpha=0.3, linewidth=3)
    
    # Plot all intermediate points in blue
    ax.scatter(embeddings_3d[1:-1, 0], embeddings_3d[1:-1, 1], embeddings_3d[1:-1, 2], 
               c='blue', s=50, alpha=0.6)
    
    # Mark first point in red
    ax.scatter(embeddings_3d[0, 0], embeddings_3d[0, 1], embeddings_3d[0, 2], 
               c='red', s=100, alpha=0.8, marker='o', label='Start')
    
    # Mark final point in green
    ax.scatter(embeddings_3d[-1, 0], embeddings_3d[-1, 1], embeddings_3d[-1, 2], 
               c='green', s=100, alpha=0.8, marker='o', label='End')
    
    ax.legend()
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Save plot to file
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"Plot saved to: {output_path}")
    plt.close(fig)  # Close the figure to free memory

    return output_path, pca


def load_model_and_tokenizer(config_file, checkpoint_path, device="cuda"):
    """
    Load model, tokenizer, and weights from a checkpoint.
    
    Args:
        config_file: Path to YAML config file
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: The loaded model
        tokenizer: The tokenizer
    """
    
    # Load configuration
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
    
    configs = Config(config_dict)
    set_seed(configs.seed)
    
    print(f"Loading model from {configs.model_id}...")
    
    # Load base model
    if configs.sdpa_attention:
        if configs.bf16:
            base_model = AutoModelForCausalLM.from_pretrained(
                configs.model_id, 
                attn_implementation="sdpa", 
                torch_dtype=torch.bfloat16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                configs.model_id, 
                attn_implementation="sdpa"
            )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # Resize embeddings for new tokens
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        base_model.resize_token_embeddings(len(tokenizer))
        embeddings = base_model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = base_model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    # Wrap model with Coconut if needed
    if configs.coconut:
        model = Coconut(
            base_model, 
            latent_id, 
            start_id, 
            end_id, 
            tokenizer.eos_token_id, 
            configs.termination_gamma
        )
    else:
        model = base_model
    
    # Move to device
    model = model.to(device)
    
    # Load checkpoint weights
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            saved_weights = checkpoint["model_state_dict"]
        else:
            saved_weights = checkpoint
        
        model.load_state_dict(saved_weights, strict=False)
        print("Checkpoint loaded successfully!")
    
    # Ensure model is in correct dtype after loading
    if configs.bf16:
        model = model.to(torch.bfloat16)
    
    model.eval()
    
    return model, tokenizer, configs


def generate_response(model, tokenizer, prompt, latent_id, max_new_tokens=128, device="cuda"):
    """
    Generate a response for the given prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Generated text (string)
    """
    prompt += "<|start-latent|>"
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_len = len(input_ids[0])
    
    print(f"\nInput prompt: {prompt}")
    print(f"Generating response (max_new_tokens={max_new_tokens})...\n")
    
    # Generate
    with torch.no_grad():

        output_ids, hidden_states, _ = model.generate_batched(
            input_ids,
            max_new_tokens=max_new_tokens,
            output_embedding=True
        )

        # if hasattr(model, 'generate_batched'):
        #     # Use Coconut's batched generation
        #     output_ids = model.generate_batched(
        #         input_ids,
        #         max_new_tokens=max_new_tokens
        #     )
        # else:
        #     # Use standard transformers generate
        #     output_ids = model.generate(
        #         input_ids,
        #         max_new_tokens=max_new_tokens,
        #         eos_token_id=tokenizer.eos_token_id,
        #         pad_token_id=tokenizer.pad_token_id
        #     )
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0, input_len:])
    print(f"Generated response:\n{generated_text}")
    print("-" * 80)

    # Generate PCA 3D representations
    latent_mask = output_ids == latent_id  # (batch_size, seq_len) boolean mask
    latent_hidden_states = hidden_states[latent_mask]  # (num_latent_tokens, hidden_dim)
    
    if latent_hidden_states.numel() == 0:
        print("Warning: No latent tokens found in output")
        return generated_text
    
    latent_hidden_states = latent_hidden_states.to(float).cpu().numpy()  # (num_latent_tokens, hidden_dim)
    print(f"Latent hidden states shape: {latent_hidden_states.shape}")

    plot_idx = len(os.listdir("./plots"))
    
    if latent_hidden_states.size > 0:
        output_path, pca = visualize_latent_pca(latent_hidden_states, title=f"Single Prompt Latent Tokens #{plot_idx}")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Infer with Coconut model")
    parser.add_argument("config_file", help="Path to config YAML file")
    parser.add_argument(
        "--checkpoint", 
        default=None,
        help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--prompt", 
        default=None,
        help="Input prompt (if not provided, will enter interactive mode)"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, configs = load_model_and_tokenizer(
        args.config_file,
        args.checkpoint,
        device=args.device
    )

    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    
    print(f"\nModel loaded successfully!")
    print(f"Coconut: {configs.coconut}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    if args.prompt:
        # Single prompt inference
        generated_text = generate_response(
            model, 
            tokenizer, 
            args.prompt,
            latent_id, 
            max_new_tokens=args.max_new_tokens,
            device=args.device
        )
        print(f"Generated response:\n{generated_text}")
        print("=" * 80)
    
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            prompt = input("Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Exiting...")
                break
            
            if not prompt:
                print("Empty prompt. Please try again.\n")
                continue
            
            # if configs.bfloat16:
            # prompt.to(torch.bfloat16)

            generated_text = generate_response(
                model, 
                tokenizer, 
                prompt, 
                latent_id,
                max_new_tokens=args.max_new_tokens,
                device=args.device
            )


if __name__ == "__main__":
    main()
