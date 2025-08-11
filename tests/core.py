import numpy as np
from skimage.morphology import medial_axis, area_closing
from PIL import Image
import sys
from input_transformation import prepare_ground_truth, prepare_prediction

sys.path.append("../external/DSE-skeleton-pruning")
from dsepruning import skel_pruning_DSE

def __get_shift_tuples(value):
    
    # Sets the radius
    radius = int(np.ceil(value))
    
    # Creates all combinations of shifts based on the raidus
    x_shifts, y_shifts = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    
    # Stacks all shifts together on a list
    shifts = np.column_stack((x_shifts.ravel(), y_shifts.ravel()))  
    # shifts = [(dx, dy) for dx, dy in shifts if (dx, dy) != (0, 0)]  # Returns without (0,0)
    
    return  shifts

def get_thin_vessels_mask(seg_mask, ceil=1.0, mask_type="ground_truth"):

    """
    Given a segmentation mask, returns a 
    numpy array containing the mask with
    only the considered thin vessels.

    Expected input shape:
        seg_mask: (H,W)
    Exprected input values:
        seg_mask in {0,1} or {0,255}
        mask_type: {"ground_truth", "prediction"}


    We consider thin vessels the ones whose
    radius is less than or equal to 'ceil'.
    """

    # Input value verification
    accepted_mask_type_values = ["ground_truth", "prediction"]

    if mask_type not in accepted_mask_type_values:
        raise ValueError(f"Expected valid mask type. Accepted values: {accepted_mask_type_values}. Got: {mask_type}")
    
    # Input preparation
    if mask_type == "ground_truth":
        seg_mask = prepare_ground_truth(seg_mask)
    else:
        seg_mask = prepare_prediction(seg_mask)
    
    # Application of closing on the segmentation mask
    closed_seg_mask = area_closing(seg_mask)

    # Obtaining the skeleton
    skeleton_medial_axis, distances = medial_axis(closed_seg_mask, return_distance=True)

    # Skeleton prunning
    skeleton_medial_axis = skel_pruning_DSE(skeleton_medial_axis, distances, np.ceil(distances.max()))

    # Compute the skeleton with the values of the distances
    dist_skel = np.where(skeleton_medial_axis>0, distances, 0) 

    # Get unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:] 

    #~~~~~~~~~~~~~~~~~~~~~~~Segmentation mask recriation with thin vessels only~~~~~~~~~~~~~~~~~~

    # Initializes the two necessary masks
    filtered_seg_mask = np.zeros(dist_skel.shape)
    reconstructed_seg_mask = np.zeros(dist_skel.shape)

    height = len(dist_skel)
    width = len(dist_skel[0])

    # Reconstructs each mask using a sphere of varying radius
    for value in values_dist_skel:
        shifts = __get_shift_tuples(value)

        for i in range(height):
            for j in range(width):
                
                if dist_skel[i][j] == value :
                    
                    if value <= ceil: 
                        for dx, dy in shifts:
                            if 0 <= i+dx < height and 0 <= j+dy < width:
                                filtered_seg_mask[i+dx][j+dy] = 255
                    
                    for dx, dy in shifts:
                        if 0 <= i+dx < height and 0 <= j+dy < width:
                            reconstructed_seg_mask[i+dx][j+dy] = 255

    # Filtering to get exactly the shape of the vessels intead of something rounded
    filtered_seg_mask = np.where((seg_mask>0) & (filtered_seg_mask>0), 255, 0).astype(np.uint8)
    reconstructed_seg_mask = np.where((seg_mask>0) & (reconstructed_seg_mask>0), 255, 0).astype(np.uint8)

    # Gets exactly the excluded vessels
    excluded_vessels = np.where((seg_mask>0) & (reconstructed_seg_mask==0), 255, 0).astype(np.uint8)

    # Concatenation of excluded_vessels seg mask with the thin vessels mask (we garantee they are small due to
    # their exclusion in the prunning/closing process)
    filtered_seg_mask = np.where((filtered_seg_mask>0) | (excluded_vessels>0), 255, 0).astype(np.uint8)
    
    return filtered_seg_mask

def main():

    example_components_path = "imgs/DRIVE_"   
    seg = Image.open(f"{example_components_path}seg_example.png")
    
    # Gets the filtered mask with only thin vessels
    thin_vessels_seg = get_thin_vessels_mask(seg)
    
    print("Showing the filtered segmentation mask with thin vessels only.")
    img = Image.fromarray(thin_vessels_seg)
    img.show()

    exit(0)


if __name__ == "__main__":
    main()