import numpy as np
from skimage.morphology import medial_axis, area_closing
from scipy.ndimage import distance_transform_edt
from PIL import Image
from retinal_thin_vessels.input_transformation import prepare_ground_truth, prepare_prediction
from retinal_thin_vessels.external.DSE_skeleton_pruning.dsepruning.dsepruning import skel_pruning_DSE

def __get_shift_tuples(value):
    
    # Sets the radius
    radius = int(np.ceil(value))
    
    # Creates all combinations of shifts based on the raidus
    x_shifts, y_shifts = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    
    # Stacks all shifts together on a list
    shifts = np.column_stack((x_shifts.ravel(), y_shifts.ravel()))  
    # shifts = [(dx, dy) for dx, dy in shifts if (dx, dy) != (0, 0)]  # Returns without (0,0)
    
    return  shifts

def __get_filtered_mask(seg_mask, ceil=1.0):
    """
    Returns a [H,W] shaped numpy array containing the thin-vessels only
    mask.
    """

    # Application of closing on the segmentation mask
    closed_seg_mask = area_closing(seg_mask)

    # Obtaining the skeleton
    skeleton_medial_axis = medial_axis(closed_seg_mask, return_distance=False)

    # Get the distance of each pixel to the background
    distances = distance_transform_edt(closed_seg_mask)

    # Skeleton pruning
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


def get_thin_vessels_mask(seg_mask, ceil=1.0, seg_mask_type="ground truth"):
    """
    Returns a numpy array containing the thin-vessels only mask.

    The Input is expected to be a segmentation mask, therefore, the 
    function accepts inputs with:

    -> shape: (H,W); (1,H,W); (N,1,H,W)
        
    -> values: BINARY
    
    -> mask_type: "ground truth" or "prediction" (affects how we 
                  transform the provided mask beofre computing the 
                  filtered one. If equal to "ground truth", will
                  apply a threshold of 0 for binarizing pixels in the
                  image (if necessary))


    Thin vessels are defined as those whose radius is less than or
    equal to 'ceil'.
    """

    # Input value verification
    accepted_mask_type_values = ["ground truth", "prediction"]

    if seg_mask_type not in accepted_mask_type_values:
        raise ValueError(f"Expected valid mask type. Accepted values: {accepted_mask_type_values}. Got: {seg_mask_type}")
    
    # Input preparation
    if seg_mask_type == "ground truth":
        seg_mask = prepare_ground_truth(seg_mask)
    else:
        seg_mask = prepare_prediction(seg_mask)

    input_dimension = seg_mask.ndim

    # Get the filtered mask(s)
    filtered_seg_mask = []
    if input_dimension == 4:
        
        # Obtain each mask separately
        for i in range(len(seg_mask)):
            #[H,W] ---> [1,H,W]
            unique_filtered_seg_mask = np.expand_dims(__get_filtered_mask(seg_mask[i][0], ceil), axis=0) 
            
            # Appends to the filtered_seg_mask vector (naturally, this vector will be [N,1,H,W])
            filtered_seg_mask.append(unique_filtered_seg_mask)
        
        # Converts into a numpy array
        filtered_seg_mask = np.array(filtered_seg_mask)

    elif input_dimension == 3:
        
        # Obtains the mask using the reduced dimension mask
        filtered_seg_mask = __get_filtered_mask(seg_mask[0], ceil)

        # Expands the dimension again for maintaining consistency with input shape
        filtered_seg_mask = np.expand_dims(filtered_seg_mask, axis=0) # [H,W] -> [1,H,W]
    
    else:
        # Obtains the mask
        filtered_seg_mask = __get_filtered_mask(seg_mask, ceil)

    return filtered_seg_mask

def main():

    # Loads the data
    example_components_path = "../tests/imgs/"   
    seg = Image.open(f"{example_components_path}DRIVE_seg_example.png")
    
    # Gets the filtered mask with only thin vessels

    #-----> for a [N,1,H,W] shape input
    seg_4_dims = np.expand_dims(np.expand_dims(np.array(seg), axis=0),axis=0)
    seg_4_dims = np.concatenate((seg_4_dims,seg_4_dims,seg_4_dims), axis=0)

    thin_vessels_seg = get_thin_vessels_mask(seg_4_dims)
    print(f"Test 1: original shape: {seg_4_dims.shape} ---> filtered mask shape: {thin_vessels_seg.shape}")
    
    #-----> for a [1,H,W] shape input
    seg_3_dims = np.expand_dims(np.array(seg), axis=0)
    
    thin_vessels_seg = get_thin_vessels_mask(seg_3_dims)
    print(f"Test 2: original shape: {seg_3_dims.shape} ---> filtered mask shape: {thin_vessels_seg.shape}")

    #-----> for a [H,W] shape input
    thin_vessels_seg = get_thin_vessels_mask(seg)
    print(f"Test 3: original shape: {np.array(seg).shape} ---> filtered mask shape: {thin_vessels_seg.shape}")

    # Shows the filtered segmentation mask
    print("Showing the filtered segmentation mask with thin vessels only.")
    img = Image.fromarray(thin_vessels_seg)
    img.show()

    exit(0)


if __name__ == "__main__":
    main()