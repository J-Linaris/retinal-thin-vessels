import torch
import numpy as np
from PIL import Image
from core import get_thin_vessels_mask
from input_transformation import prepare_input
import time

def __recall_thin_vessels_single_image(y_true, y_pred, ceil=1.0):
    """
    Given inputs with the predicted and true vessel masks, returns 
    the recall score on thin vessels. Inputs can be NumPy arrays, 
    torch.Tensors, or PIL Images.

    Inputs must have shape: (1, H, W), with values in {0, 1}, {0, 255}, 
    or [0.0, 1.0] for probability maps.

    Thin vessels are defined as those with radius less than or
    equal to ceil.
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Input preparation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    true_copy, pred_copy = prepare_input(y_true, y_pred)

    new_seg_true = get_thin_vessels_mask(true_copy, ceil)

    # Calculates Recall
    tp = np.sum((new_seg_true > 0) & (pred_copy > 0))
    fn = np.sum((new_seg_true > 0) & (pred_copy == 0))

    return tp/(tp+fn)

def __precision_thin_vessels_single_image(y_true, y_pred, ceil=1.0):
    """
    Given inputs with the predicted and true vessel masks, returns 
    the precision score on thin vessels. Inputs can be NumPy arrays, 
    torch.Tensors, or PIL Images.

    Inputs must have shape: (1, H, W), with values in {0, 1}, {0, 255}, 
    or [0.0, 1.0] for probability maps.

    Thin vessels are defined as those with radius less than or
    equal to ceil.
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Input preparation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    true_copy, pred_copy = prepare_input(y_true, y_pred)

    new_seg_pred = get_thin_vessels_mask(pred_copy, ceil)

    # Calculates Precision
    tp = np.sum((new_seg_pred > 0) & (true_copy > 0))
    fp = np.sum((new_seg_pred > 0) & (true_copy == 0))

    return tp/(tp+fp)

def __f1_thin_vessels_single_image(y_true, y_pred, ceil=1.0):

    r = __recall_thin_vessels_single_image(y_true, y_pred, ceil)
    p = __precision_thin_vessels_single_image(y_true, y_pred, ceil)
    f1 = 2*p*r/(p+r)

    return f1

def recall_thin_vessels(y_true, y_pred, ceil=1.0):

    # Prepares the input

    y_true, y_pred = prepare_input(y_true, y_pred)
    
    # Computes the metric
    inputs_dimension = y_true.ndim
    num_imgs = len(y_true)

    if inputs_dimension == 4:
        recall = 0
        for i in range(num_imgs):
            recall+=__recall_thin_vessels_single_image(y_true[i][0], y_pred[i][0], ceil)
        recall/=num_imgs
    
    else:
        recall = __recall_thin_vessels_single_image(y_true, y_pred, ceil)
    
    return recall

def precision_thin_vessels(y_true, y_pred, ceil=1.0):

    # Prepares the input
    
    y_true, y_pred = prepare_input(y_true, y_pred)

    # Computes the metric
    inputs_dimension = y_true.ndim
    num_imgs = len(y_true)

    if inputs_dimension == 4:
        precision = 0
        for i in range(num_imgs):
            precision+=__precision_thin_vessels_single_image(y_true[i][0], y_pred[i][0], ceil)
        precision/=num_imgs
    
    else:
        precision = __precision_thin_vessels_single_image(y_true, y_pred, ceil)
    
    return precision


def f1_thin_vessels(y_true, y_pred, ceil):

    # Prepares the input
    y_true, y_pred = prepare_input(y_true, y_pred)

    # Computes the metric
    inputs_dimension = y_true.ndim
    num_imgs = len(y_true)

    if inputs_dimension == 4:
        f1 = 0
        for i in range(num_imgs):
            f1 += __f1_thin_vessels_single_image(y_true[i][0], y_pred[i][0], ceil)

        f1 /= num_imgs
    
    else:
        f1 = __f1_thin_vessels_single_image(y_true, y_pred, ceil)
    
    return f1


def main():

    example_components_path = "../tests/imgs/"   
    img = Image.open(f"{example_components_path}img_example.png")
    seg = Image.open(f"{example_components_path}seg_example.png")
    pred = Image.open(f"{example_components_path}pred_example.png")
    
    ti = time.time()
    print(precision_thin_vessels(seg.resize(pred.size, Image.NEAREST), pred))
    tf = time.time()
    delta = tf-ti
    print(delta)

    ti = time.time()
    print(recall_thin_vessels(seg.resize(pred.size, Image.NEAREST), pred))
    tf = time.time()
    delta = tf-ti
    print(delta)
    exit(0)


if __name__ == "__main__":
    main()
