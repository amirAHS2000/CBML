
import torch
import numpy as np


def feat_extractor(model, data_loader, logger=None):
    model.eval()
    feats = list()

    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()

        with torch.no_grad():
            out = model(imgs).data.cpu().numpy()
            feats.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f'Extract Features: [{i + 1}/{len(data_loader)}]')
        del out
    feats = np.vstack(feats)
    return feats

def compute_similarity_stats(model, criterion, loader, device):
    """
    Computes the dominant positive and dominant negative similarities 
    for the entire dataset provided by 'loader'.
    """
    model.eval()
    all_pos_sims = []
    all_neg_sims = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # 1. Get Normalized Embeddings
            embeddings = model(images)
            z = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # 2. Get Prototypes from your Criterion (Loss module)
            # Shape: [Num_Classes, K_Prototypes, Dim]
            protos = criterion.prototypes 
            C, K, D = protos.shape
            B = z.shape[0]
            
            # 3. Compute Raw Similarities [B, C, K]
            # Flatten protos to [C*K, D] for matmul
            flat_protos = protos.view(C*K, D).t()
            sims = torch.matmul(z, flat_protos).view(B, C, K)
            
            # 4. Extract Dominant Positive (Max sim to target class)
            # Select the [K] similarities for the correct class for each batch item
            pos_class_sims = sims[torch.arange(B), targets] # [B, K]
            best_pos, _ = pos_class_sims.max(dim=1) # [B]
            
            # 5. Extract Dominant Negative (Max sim to ANY non-target class)
            # Create a mask for the target class
            mask = torch.ones_like(sims, dtype=torch.bool)
            mask[torch.arange(B), targets] = False
            
            # Mask out the positive class (set to -inf so it's not selected as max)
            # Note: We reshape to [B, C*K] to find global max negative easily
            sims_flat = sims.view(B, -1)
            mask_flat = mask.view(B, -1)
            
            # Fill positives with -10.0 (sims are usually -1 to 1)
            neg_sims_masked = torch.where(mask_flat, sims_flat, torch.tensor(-10.0, device=device))
            
            # Get max over all negative prototypes
            best_neg, _ = neg_sims_masked.max(dim=1) # [B]
            
            all_pos_sims.extend(best_pos.cpu().numpy())
            all_neg_sims.extend(best_neg.cpu().numpy())
            
    return np.array(all_pos_sims), np.array(all_neg_sims)