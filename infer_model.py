#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut_grpo_baseline_manual import Coconut
from utils import Config, set_seed

import yaml
import argparse


"""
python -m infer_model --checkpoint checkpoints/gsm-coconut-grpo-baseline/260307_031254/ref_models/eval_best_model_1_1600_0.25.pt ./args/gsm
_vllr_grpo_coconut_baseline.yaml 
"""


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


def generate_response(model, tokenizer, prompt, max_new_tokens=128, device="cuda"):
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
    
    print(f"\nInput prompt: {prompt}")
    print(f"Generating response (max_new_tokens={max_new_tokens})...\n")
    
    # Generate
    with torch.no_grad():

        output_ids = model.generate_batched(
            input_ids,
            max_new_tokens=max_new_tokens
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
    generated_text = tokenizer.decode(output_ids[0])
    
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
                max_new_tokens=args.max_new_tokens,
                device=args.device
            )
            print(f"Generated response:\n{generated_text}")
            print("-" * 80)


if __name__ == "__main__":
    main()
