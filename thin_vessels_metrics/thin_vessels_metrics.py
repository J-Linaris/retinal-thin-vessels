import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
from torchvision.transforms import ToPILImage
from skimage.morphology import medial_axis, area_closing
from PIL import Image
import sys
import os

sys.path.append("external/DSE-skeleton-pruning")
from dsepruning import skel_pruning_DSE
# os.chdir("/home/jplinaris/UNET_DRIVE/")

def get_shift_tuples(value):
    
    # Sets the radius
    radius = int(np.ceil(value))
    
    # Creates all combinations of shifts based on the raidus
    x_shifts, y_shifts = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    
    # Stacks all shifts together on a list
    shifts = np.column_stack((x_shifts.ravel(), y_shifts.ravel()))  
    # shifts = [(dx, dy) for dx, dy in shifts if (dx, dy) != (0, 0)]  # Returns without (0,0)
    
    return  shifts


def recall_thin_vessels(y_true, y_pred, ceil=1.0):
    """
    Given a prediction of the vessels mask
    and the actual vessels mask, returns the
    recall score on thin vessels. It is also
    expected that the input values are numpy
    arrays belonging to {0,255} or {0,1}.

    Expected input shape:
    y_pred: (1,H,W) (greyscale image)
    y_true: (1,H,W) (greyscale image)

    We consider thin vessels the ones whose
    radius is less than or equal to 'ceil'.
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Input preparation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pred_copy = y_pred.copy()
    true_copy = y_true.copy()

    pred_copy = pred_copy.astype(np.uint8)
    true_copy = true_copy.astype(np.uint8)

    pred_copy = pred_copy[0] #[1,512,512] --> [512,512]
    true_copy = true_copy[0] #[1,512,512] --> [512,512]

    # Application of closing on the segmentation mask
    closed_true_copy = area_closing(true_copy)

    # Obtaining the skeleton
    skeleton_medial_axis, distances = medial_axis(closed_true_copy, return_distance=True)

    # Skeleton prunning
    skeleton_medial_axis = skel_pruning_DSE(skeleton_medial_axis, distances, np.ceil(distances.max()))

    # Compute the skeleton with the values of the distances
    dist_skel = np.where(skeleton_medial_axis>0, distances, 0) 

    # Get unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:] 

    all_values_dist_skel = values_dist_skel.copy()

    # Filtering values less than or equal to the ceiling considered (Only thin vessels remain)
    values_dist_skel = values_dist_skel[values_dist_skel <= ceil]

    #~~~~~~~~~~~~~~~~~~~~~~~Segmentation mask recriation with thin vessels only~~~~~~~~~~~~~~~~~~
    new_seg_true = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    for value in values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if abs(dist_skel[i][j] - value) <= 0.1:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            new_seg_true[i+dx][j+dy] = 255

    # Filtering to get exactly the shape of the vessels, and not something rounded
    new_seg_true = np.where((true_copy>0) & (new_seg_true>0), 255, 0).astype(np.uint8)

    #~~~~~~~Thin vessels segmentation mask addition of lost vessels in prunning/closing process~~~~~~~~
    reconstructed_seg_mask = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    
    for value in all_values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            reconstructed_seg_mask[i+dx][j+dy] = 255
    
    # Filtering to get exactly the shape of the vessels, and not something rounded
    reconstructed_seg_mask = np.where((true_copy>0) & (reconstructed_seg_mask>0), 255, 0).astype(np.uint8)

    # Gets exactly the excluded vessels
    excluded_vessels = np.where((true_copy>0) & (reconstructed_seg_mask==0), 255, 0).astype(np.uint8)

    # Concatenation of excluded_vessels seg mask with the thin vessels mask (we garantee they are small due to
    # their exclusion in the prunning/closing process)
    new_seg_true = np.where((new_seg_true>0) | (excluded_vessels>0), 255, 0).astype(np.uint8)


    # Calculates Recall
    tp = 0.0
    fn = 0.0
    for i in range(len(new_seg_true)):
        for j in range(len(new_seg_true[i])):
            if new_seg_true[i][j]:     
                if pred_copy[i][j]:
                    tp +=1
                else:
                    fn+=1

    return tp/(tp+fn)

def precision_thin_vessels(y_true, y_pred, ceil=1.0):
        """
    Given a prediction of the vessels mask
    and the actual vessels mask, returns the
    recall score on thin vessels. It is also
    expected that the input values are numpy
    arrays belonging to {0,255} or {0,1}.

    Expected input shape:
    y_pred: (1,H,W) (greyscale image)
    y_true: (1,H,W) (greyscale image)

    We consider thin vessels the ones whose
    radius is less than or equal to 'ceil'.
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Input preparation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    pred_copy = y_pred.copy()
    true_copy = y_true.copy()

    pred_copy = pred_copy.astype(np.uint8)
    true_copy = true_copy.astype(np.uint8)

    pred_copy = pred_copy[0] #[1,512,512] --> [512,512]
    true_copy = true_copy[0] #[1,512,512] --> [512,512]

    # Application of closing on the segmentation mask
    closed_pred_copy = area_closing(pred_copy)

    # Obtaining the skeleton
    skeleton_medial_axis, distances = medial_axis(closed_pred_copy, return_distance=True)

    # Skeleton prunning
    skeleton_medial_axis = skel_pruning_DSE(skeleton_medial_axis, distances, np.ceil(distances.max()))
    
    # Compute the skeleton with the values of the distances
    dist_skel = np.where(skeleton_medial_axis>0, distances, 0) # Esqueleto com as distâncias

    # Get unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:] # Valores únicos de dist_skel (raios dos vasos) (tirando o 0)
    
    all_values_dist_skel = values_dist_skel.copy()

    # Filtering values less than or equal to the ceiling considered (leaves thin vessels only)
    values_dist_skel = values_dist_skel[values_dist_skel <= ceil]

    #~~~~~~~~~~~~~~~~~~~~~~~Segmentation mask recriation with thin vessels only~~~~~~~~~~~~~~~~~~  
    new_seg_pred = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    
    for value in values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            new_seg_pred[i+dx][j+dy] = 255
    
    # Filtering to get exactly the shape of the vessels, and not something rounded
    new_seg_pred = np.where((pred_copy>0) & (new_seg_pred>0), 255, 0).astype(np.uint8)

    #~~~~~~~Thin vessels segmentation mask addition of lost vessels in prunning/closing process~~~~~~~~
    reconstructed_pred_seg_mask = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    
    for value in all_values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            reconstructed_pred_seg_mask[i+dx][j+dy] = 255
    
    # Filtering to get exactly the shape of the vessels, and not something rounded
    reconstructed_pred_seg_mask = np.where((pred_copy>0) & (reconstructed_pred_seg_mask>0), 255, 0).astype(np.uint8)

    # Gets exactly the excluded vessels
    excluded_vessels = np.where((pred_copy>0) & (reconstructed_pred_seg_mask==0), 255, 0).astype(np.uint8)

    # Concatenation of excluded_vessels seg mask with the thin vessels mask (we garantee they are small due to
    # their exclusion in the prunning/closing process)
    new_seg_pred = np.where((new_seg_pred>0) | (excluded_vessels>0), 255, 0).astype(np.uint8)

    # Calculates Precision
    tp = 0.0
    fp = 0.0
    for i in range(len(new_seg_pred)):
        for j in range(len(new_seg_pred[i])):
            if new_seg_pred[i][j]:     
                if true_copy[i][j]:
                    tp +=1
                else:
                    fp+=1

    return tp/(tp+fp)


def main():

    test_components_path = "test_components/"   
    img = Image.open(f"{test_components_path}img_example.png", "RGB")
    seg = Image.open(f"{test_components_path}seg_example.png", "L")
    pred = Image.open(f"{test_components_path}pred_example.png")
    
    

    exit(0)


if __name__ == "__main__":
    main()
