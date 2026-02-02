import numpy as np

# Define IOU 
def calculate_iou(pred_mask, true_mask):
    ious = []
    # Loop through each channel (0=OM, 1=IM)
    for c in range(pred_mask.shape[0]):
        p = pred_mask[c]
        t = true_mask[c]
        
        # --- Your original logic applied to one channel ---
        intersection = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        iou = intersection / (union + 1e-6)        
        ious.append(iou)
    return np.mean(ious)  # Return the average across both membranes

# Define DICE
def calculate_dice(pred_mask, true_mask):
    dices = []
    # Loop through each channel (0=OM, 1=IM)
    for c in range(pred_mask.shape[0]):
        p = pred_mask[c]
        t = true_mask[c]
        
        # --- Your original logic applied to one channel ---
        intersection = np.logical_and(t, p).sum()
        dice = (2 * intersection) / (p.sum() + t.sum() + 1e-6)        
        dices.append(dice)
        
    return np.mean(dices) 

#Define Precision_Recall_F1 

def calculate_precision_recall_f1(pred_mask, true_mask):
    """
    Args:
        pred_mask (np.array): shape [2, H, W] (0 or 1)
        true_mask (np.array): shape [2, H, W] (0 or 1)
    """
    precisions, recalls, f1s = [], [], []
    
    for c in range(pred_mask.shape[0]):
        p = pred_mask[c].flatten()
        t = true_mask[c].flatten()
        
        # True Positive: Both are 1
        tp = np.sum((p == 1) & (t == 1))
        # False Positive: Predicted 1, but actually 0
        fp = np.sum((p == 1) & (t == 0))
        # False Negative: Predicted 0, but actually 1
        fn = np.sum((p == 0) & (t == 1))
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)