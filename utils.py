# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_latent_pca(all_latent_embeddings, title="Latent Token Embeddings (PCA)", colors=None, output_dir="./plots", question=None, answer=None, predicted=None, reward=None, idx=0, verbose=False):
    """
    Apply PCA to reduce to 3D and visualize, save to file.
    
    Args:
        all_latent_embeddings: (total_latent_tokens, hidden_dim) array from multiple prompts
        title: Title for the plot
        colors: (total_latent_tokens,) array or list of colors for each point
        output_dir: Directory to save the plot
        question: Question string to display below plot
        answer: Answer string to display below plot
        predicted: Predicted string to display below plot
        reward: Reward float to display below plot
    
    Returns:
        output_path: Path to the saved plot file
        pca: The fitted PCA model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fit PCA to 3 dimensions
    if len(all_latent_embeddings) == 0:
        print(f"Dimension of input is {all_latent_embeddings.shape}")
        return None, 1

    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_latent_embeddings)
    
    if verbose:
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Create 3D scatter plot with space for text below
    fig = plt.figure(figsize=(10, 10))
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
    
    # Add text below the plot for question, answer, predicted, and reward
    text_str = ""
    if question is not None:
        text_str += f"Question: {question}\n"
    if answer is not None:
        text_str += f"Answer: {answer}\n"
    if predicted is not None:
        text_str += f"Predicted: {predicted}\n"
    if reward is not None:
        text_str += f"Reward: {reward:.3f}"
    
    if text_str:
        text_str = text_str.replace('$', r'\$')
        fig.text(0.05, -0.08, text_str, fontsize=10, verticalalignment='top',
                 wrap=True, family='monospace', clip_on=False)
    
    plt.tight_layout()
    
    # Save plot to file
    output_path = os.path.join(output_dir, f"Question_{idx:04d}_PCA_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    if verbose:
        print(f"Plot saved to: {output_path}")
    plt.close(fig)  # Close the figure to free memory

    return pca, sum(pca.explained_variance_ratio_)


def save_latent_histogram(n_latents_all, n_latents_correct, n_latents_incorrect, n_latents_noformat, output_dir):
    """
    Save four latent-token-count histograms: total, correct, incorrect, wrong-format.
    """
    os.makedirs(output_dir, exist_ok=True)

    subsets = [
        ("all",       n_latents_all,       "steelblue",  "All Questions"),
        ("correct",   n_latents_correct,   "seagreen",   "Correct"),
        ("incorrect", n_latents_incorrect, "tomato",     "Incorrect"),
        ("noformat",  n_latents_noformat,  "goldenrod",  "Wrong Format"),
    ]

    for tag, data, color, label in subsets:
        fig, ax = plt.subplots(figsize=(8, 5))
        if data:
            max_val = max(data)
            bins = range(0, max_val + 2)
            ax.hist(data, bins=bins, align='left', rwidth=0.8, color=color, edgecolor='black')
            mean_val = sum(data) / len(data)
            tick_step = max(1, (max_val + 1) // 20)
            ax.set_xticks(range(0, max_val + 1, tick_step))
        else:
            mean_val = 0.0
            ax.text(0.5, 0.5, "No samples", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("Number of Latent Tokens")
        ax.set_ylabel("Number of Questions")
        ax.set_title(f"Latent Token Distribution — {label} (n={len(data)}, mean={mean_val:.2f})")
        fig.tight_layout()
        path = os.path.join(output_dir, f"latent_histogram_{tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Histogram saved: {path}")