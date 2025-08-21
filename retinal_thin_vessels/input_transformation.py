import torch
import numpy as np
from PIL import Image

def _to_numpy(arr):
    """
    Converts torch.Tensor or PIL.Image 
    to numpy array if needed.
    """
    if isinstance(arr, Image.Image):
        arr = np.array(arr, copy=True)
    elif isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif not isinstance(arr, np.ndarray):
        raise TypeError(f"Unsupported input type: {type(arr)}")
    return arr

def _verify_input_validity(seg_mask, input_type="y_true"):
    """
    Inputs can be NumPy arrays, 
    torch.Tensors, or PIL Images.

    (1, H, W) or (N,1,H,W) or (H,W)
    Inputs must have shape: (1, H, W), with values in {0, 1}, {0, 255}, 
    or [0.0, 1.0] for probability maps.

    Thin vessels are defined as those with radius less than or
    equal to ceil.
    """

    seg_mask = _to_numpy(seg_mask)

    # Ensures inputs have valid shape
    if (int(seg_mask.ndim) not in [2,3,4]) or (seg_mask.ndim == 3 and seg_mask.shape[0] != 1) or (seg_mask.ndim == 4 and seg_mask.shape[1] != 1):
        raise ValueError(f"Accepted shapes: (1, H, W) or (N,1,H,W) or (H,W). Got seg_mask: {seg_mask.shape} for {input_type}")

    # Verifies the validity of the values in the passed array
    input_values = np.unique(seg_mask)
    
    if len(input_values) != 2:
        raise ValueError(f"Expected binary input for {input_type}. Got input values: {input_values}")
    
    # Binarize input if necessary
    if (input_type == "y_true" and int(input_values[0]) != 0) or (input_type == "y_pred" and int(input_values[0]) != 0):
            seg_mask = (seg_mask - seg_mask.min())/(seg_mask.max()-seg_mask.min())

    return seg_mask

def prepare_ground_truth(y_true):

    # Verifies validity of y_true
    y_true = _verify_input_validity(y_true, "y_true")

    # # Removes the channel dimension for input of shape [1,H,W]
    # if y_true.ndim == 3:
    #     y_true = y_true[0] #[1,H,W] ---> [H,W]

    # Ensures the values belong to {0,1}
    y_true = (y_true > 0).astype(np.uint8)

    return y_true

def prepare_prediction(y_pred):

    # Verifies validity of y_true
    y_pred = _verify_input_validity(y_pred, "y_pred")

    # # Removes the channel dimension for input of shape [1,H,W]
    # if y_pred.ndim == 3:
    #     y_pred = y_pred[0] #[1,H,W] ---> [H,W]
    
    # Ensures the values belong to {0,1}
    y_pred = (y_pred > 0.5).astype(np.uint8)

    return y_pred

def prepare_input(y_true, y_pred):
    
    # Prepare both inputs
    y_true, y_pred = prepare_ground_truth(y_true), prepare_prediction(y_pred)

    # Ensures inputs have the same dimensions (Height and Width)
    # if (y_true.shape[-1] != y_pred.shape[-1] or y_true.shape[-2] != y_pred.shape[-2]):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Expected both inputs to have the same dimensions. Got  y_true: {y_true.shape}, y_pred: {y_pred.shape}")


    return y_true, y_pred