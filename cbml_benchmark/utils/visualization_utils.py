import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


def plot_scalar_trends(log_path, save_dir=None):
    """
    Plot trends of scalar statistics (theta, distances, entropy, etc.)
    Assumes `log_path` is a CSV or text file with columns:
    iteration, theta, mean_intra_dist, mean_inter_dist, mean_displacement, mean_entropy, mean_max_weight
    """
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 6))
    for col in ['theta', 'mean_intra_dist', 'mean_inter_dist', 'mean_displacement']:
        plt.plot(df['iteration'], df[col], label=col)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Scalar Statistics Over Training')
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'scalar_trends.png'))
    plt.show()

def plot_tsne(feats, labels, prototypes, save_dir=None):
    """Visualize embeddings and prototypes with t-SNE."""
    protos_flat = prototypes.reshape(-1, prototypes.shape[-1])
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X_embedded = tsne.fit_transform(np.vstack([feats, protos_flat]))

    n_feats = feats.shape[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(X_embedded[:n_feats, 0], X_embedded[:n_feats, 1],
                c=labels, cmap='tab20', s=5, alpha=0.5, label='samples')
    plt.scatter(X_embedded[n_feats:, 0], X_embedded[n_feats:, 1],
                c='red', marker='X', s=80, label='prototypes')
    plt.legend()
    plt.title('t-SNE projection of embeddings and prototypes')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'tsne_prototypes.png'))
    plt.show()

# def plot_tsne(feats, labels, prototypes, save_dir=None, max_points=2000):
#     """
#     Visualize embeddings and prototypes with t-SNE.
#     Each class has a unique color; prototypes share their class color but use a distinct marker.
#     """
#     # --- Prepare data ---
#     n_classes = prototypes.shape[0]
#     proto_dim = prototypes.shape[-1]
#     protos_flat = prototypes.reshape(-1, proto_dim)

#     # Optionally subsample features for faster t-SNE
#     if len(feats) > max_points:
#         idx = np.random.choice(len(feats), max_points, replace=False)
#         feats = feats[idx]
#         labels = labels[idx]

#     # Stack features + prototypes
#     all_data = np.vstack([feats, protos_flat])
#     tsne = TSNE(n_components=2, init='pca', random_state=42)
#     X_embedded = tsne.fit_transform(all_data)

#     n_feats = feats.shape[0]
#     feat_embeds = X_embedded[:n_feats]
#     proto_embeds = X_embedded[n_feats:]

#     # --- Plot ---
#     plt.figure(figsize=(10, 10))
#     unique_labels = np.unique(labels)
#     cmap = plt.get_cmap('tab20', len(unique_labels))

#     # plot samples per class
#     for i, cls in enumerate(unique_labels):
#         cls_mask = labels == cls
#         plt.scatter(
#             feat_embeds[cls_mask, 0], feat_embeds[cls_mask, 1],
#             color=cmap(i), s=6, alpha=0.5
#         )

#     # plot prototypes (use same class color, distinct marker)
#     for i in range(n_classes):
#         proto_coords = proto_embeds[i * (prototypes.shape[1]): (i + 1) * (prototypes.shape[1])]
#         plt.scatter(
#             proto_coords[:, 0], proto_coords[:, 1],
#             color=cmap(i), edgecolors='black', marker='X', s=120, linewidth=0.5
#         )

#     plt.title('t-SNE Projection of Embeddings and Prototypes')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')

#     # Add simplified legend (only a few sample classes)
#     sample_classes = unique_labels[:min(10, len(unique_labels))]
#     handles = [plt.Line2D([0], [0], marker='o', color='w',
#                           label=f'Class {int(c)}',
#                           markerfacecolor=cmap(i), markersize=6)
#                for i, c in enumerate(sample_classes)]
#     handles.append(plt.Line2D([0], [0], marker='X', color='black',
#                               label='Prototypes', markerfacecolor='gray', markersize=10))
#     plt.legend(handles=handles, loc='best', fontsize=8, frameon=True)

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'tsne_prototypes.png'), dpi=300, bbox_inches='tight')
#     plt.close()


def plot_prototype_displacement(model, save_dir=None):
    """Boxplot of prototype displacement relative to initial positions."""
    displacements = torch.norm(model.prototypes - model.initial_prototypes, dim=-1).cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.boxplot(displacements.flatten())
    plt.title('Distribution of Prototype Displacements')
    plt.ylabel('L2 Distance from Initialization')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prototype_displacement.png'))
    plt.show()

def plot_entropy_histogram(model, save_dir=None):
    """Histogram of per-class prototype entropies."""
    with torch.no_grad():
        weights = F.softmax(model.weights, dim=1)
        entropy_per_class = -torch.sum(weights * (weights.clamp_min(1e-9)).log(), dim=1).cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(entropy_per_class, bins=30, color='orange', edgecolor='k')
    plt.xlabel('Entropy per class')
    plt.ylabel('Count')
    plt.title('Distribution of Prototype Weight Entropies')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'entropy_histogram.png'))
    plt.show()

def plot_prototype_similarity(model, save_dir=None):
    """Heatmap of prototype-prototype cosine similarity."""
    with torch.no_grad():
        protos_norm = F.normalize(model.prototypes.view(-1, model.embed_dim), p=2, dim=1)
        sim_matrix = torch.matmul(protos_norm, protos_norm.T).cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, cmap='coolwarm', center=0)
    plt.title('Prototype-Prototype Cosine Similarity')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prototype_similarity.png'))
    plt.show()
