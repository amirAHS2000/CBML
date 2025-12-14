import torch
import numpy as np


def feat_extractor(model, data_loader, logger=None, extract_layer=None):
    """
    Extract features from the model.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model with backbone and head
    data_loader : DataLoader
        Data loader for extraction
    logger : Logger, optional
        Logger for debugging
    extract_layer : str, optional
        Which layer to extract from: 'layer1', 'layer2', 'layer3', 'layer4', or None
        If None (default): uses model's output (full forward pass)
        If specified: extracts from that specific backbone layer
    
    Returns:
    --------
    feats : np.ndarray
        Extracted features [num_samples, feature_dim]
    """
    model.eval()
    feats = list()

    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()

        with torch.no_grad():
            if extract_layer is None:
                # Standard extraction: full forward pass through model
                out = model(imgs).data.cpu().numpy()
            else:
                # Extract from intermediate layer of backbone
                backbone = model.backbone
                
                # Get features from the specified layer
                x = imgs
                x = backbone.model.conv1(x)
                x = backbone.model.bn1(x)
                x = backbone.model.relu(x)
                x = backbone.model.maxpool(x)
                
                x = backbone.model.layer1(x)
                if extract_layer == 'layer1':
                    x = backbone.model.avgpool(x)
                    out = x.view(x.size(0), -1).data.cpu().numpy()
                else:
                    x = backbone.model.layer2(x)
                    if extract_layer == 'layer2':
                        x = backbone.model.avgpool(x)
                        out = x.view(x.size(0), -1).data.cpu().numpy()
                    else:
                        x = backbone.model.layer3(x)
                        if extract_layer == 'layer3':
                            x = backbone.model.avgpool(x)
                            out = x.view(x.size(0), -1).data.cpu().numpy()
                        else:
                            x = backbone.model.layer4(x)
                            x = backbone.model.avgpool(x)
                            out = x.view(x.size(0), -1).data.cpu().numpy()
            
            feats.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f'Extract Features: [{i + 1}/{len(data_loader)}]')
        del out
    
    feats = np.vstack(feats)
    return feats


def compute_similarity_stats(model, criterion, loader, device):
    """
    Computes the dominant positive and dominant negative similarities 
    for the entire dataset using the multi-prototype formulation.
    
    This matches your MP-CBML loss computation:
    - Positive: max similarity to any prototype of the true class
    - Negative: max similarity to any prototype of any other class
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
            z = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # [B, D]
            
            # 2. Get Prototypes [C, K, D]
            protos = criterion.prototypes 
            C, K, D = protos.shape
            B = z.shape[0]
            
            # 3. Compute similarities [B, C, K]
            flat_protos = protos.view(C*K, D).t()  # [D, C*K]
            sims = torch.matmul(z, flat_protos).view(B, C, K)  # [B, C, K]
            
            # 4. Dominant Positive: max similarity to target class prototypes
            pos_class_sims = sims[torch.arange(B, device=device), targets]  # [B, K]
            best_pos, _ = pos_class_sims.max(dim=1)  # [B]
            
            # 5. Dominant Negative: max similarity to non-target class prototypes
            # Create mask for negative classes
            neg_mask = torch.ones(B, C, dtype=torch.bool, device=device)
            neg_mask[torch.arange(B, device=device), targets] = False  # [B, C]
            neg_mask = neg_mask.unsqueeze(-1).expand(B, C, K)  # [B, C, K]
            
            # Extract negative similarities and find max
            neg_sims = sims[neg_mask].view(B, (C-1)*K)  # [B, (C-1)*K]
            best_neg, _ = neg_sims.max(dim=1)  # [B]
            
            all_pos_sims.extend(best_pos.cpu().numpy())
            all_neg_sims.extend(best_neg.cpu().numpy())
            
    return np.array(all_pos_sims), np.array(all_neg_sims)