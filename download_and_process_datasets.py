"""
Script to download and process SVAMP and GSM-Hard datasets from Hugging Face.
Outputs JSON files formatted to match gsm_test.json structure.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset


def process_svamp():
    """
    Download and process SVAMP dataset.
    Combines 'body' and 'question' columns for the question field.
    """
    print("Loading SVAMP dataset...")
    dataset = load_dataset("ChilleD/SVAMP")
    
    output_data = {}
    
    for split in ["train", "test"]:
        print(f"Processing SVAMP {split} split...")
        split_data = dataset[split]
        
        processed_items = []
        for item in split_data:
            # Combine body and question
            combined_question = f"{item['Body']} {item['Question']}"
            
            processed_item = {
                "question": combined_question,
                "steps": ["."],  # Placeholder for GRPO compatibility
                "answer": str(item["Answer"])
            }
            processed_items.append(processed_item)
        
        output_data[split] = processed_items
        print(f"  Processed {len(processed_items)} items")
    
    # Save to JSON files
    for split, items in output_data.items():
        output_path = f"data/svamp_{split}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(items, f, indent=4)
        
        print(f"Saved {output_path} with {len(items)} items")
    
    return output_data


def process_gsm_hard():
    """
    Download and process GSM-Hard dataset.
    Uses 'input' column for question and 'target' column for answer.
    """
    print("Loading GSM-Hard dataset...")
    dataset = load_dataset("reasoning-machines/gsm-hard")
    
    output_data = {}
    
    for split in ["train"]:
        print(f"Processing GSM-Hard {split} split...")
        split_data = dataset[split]
        
        processed_items = []
        for item in split_data:
            processed_item = {
                "question": item["input"],
                "steps": ["."],  # Placeholder for GRPO compatibility
                "answer": str(item["target"])
            }
            processed_items.append(processed_item)
        
        output_data[split] = processed_items
        print(f"  Processed {len(processed_items)} items")
    
    # Save to JSON files
    for split, items in output_data.items():
        output_path = f"data/gsm_hard_{split}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(items, f, indent=4)
        
        print(f"Saved {output_path} with {len(items)} items")
    
    return output_data


def main():
    """Download and process both datasets."""
    print("=" * 60)
    print("Dataset Download and Processing Script")
    print("=" * 60)
    
    try:
        print("\n--- Processing SVAMP Dataset ---")
        svamp_data = process_svamp()
        
        print("\n--- Processing GSM-Hard Dataset ---")
        gsm_hard_data = process_gsm_hard()
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - data/svamp_train.json")
        print("  - data/svamp_test.json")
        print("  - data/gsm_hard_train.json")
        print("  - data/gsm_hard_test.json")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise


if __name__ == "__main__":
    main()
