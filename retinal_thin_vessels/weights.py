import numpy as np
from skimage.morphology import medial_axis, area_closing
from scipy.ndimage import distance_transform_edt
from PIL import Image
from retinal_thin_vessels.input_transformation import prepare_ground_truth, prepare_prediction
from retinal_thin_vessels.external.DSE_skeleton_pruning.dsepruning.dsepruning import skel_pruning_DSE
from retinal_thin_vessels.core import __get_shift_tuples

def __get_weight_mask(seg_mask, weights_function=0):
    """
    Returns a [H,W] shaped numpy array containing the weights
    for the specified seg_mask according to the wanted weights
    function. For a pixel i:

        if weights_function == 0:
            W_i = 2 / (D_esq ** 2)
        
        if weights_function == 1:
            W_i = (D_i + 1) / (D_esq ** 2)

    Expects a [H,W] numpy array as input.
    """

    # Application of closing on the segmentation mask
    seg_mask_copy = area_closing(seg_mask, 32)

    # Application of closing on the segmentation mask
    skel = medial_axis(seg_mask_copy, return_distance=False)

    # Gets the distance of each pixel to the background
    distances = distance_transform_edt(seg_mask_copy)

    # Skeleton pruning
    skel = skel_pruning_DSE(skel, distances, np.ceil(distances.max()))

    # Computes the skeleton with the values of the distances
    dist_skel = np.where(skel>0, distances, 0) # Esqueleto com as distâncias 

    # Gets unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:]

    # Sorts in decreasing order (this prioritizes greater weights in case of radius overlap)
    values_dist_skel = -np.sort(-values_dist_skel)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Weight mask computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialization
    W = np.zeros(dist_skel.shape, dtype = np.float32)

    # Sets the weights using the decreasing order of radius 
    for value in values_dist_skel:
        shifts = __get_shift_tuples(value)
        
        lines = len(W)
        columns = len(W[0])
        
        for i in range(lines):
            for j in range(columns):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < lines and 0 <= j+dy < columns:
                            
                            match weights_function:
                                case 0:
                                    W[i+dx][j+dy] = 2/(value**2)
                                case 1:
                                    W[i+dx][j+dy] = (distances[i+dx][j+dy]+1) / (value**2)
    
    return W

def get_weight_mask(seg_mask, weights_function=0, seg_mask_type="ground truth"):
    """
    Returns a numpy array containing the weight mask(s) according
    to the wanted weights function. For a pixel i:
    
        if weights_function == 0:
            W_i = 2 / (D_esq ** 2)
        
        if weights_function == 1:
            W_i = (D_i + 1) / (D_esq ** 2)

    The Input is expected to be a segmentation mask (numpy array, 
    PIL.Image instance or torch.tensor), therefore, the function
    accepts inputs with:

    -> shape: (H,W); (1,H,W); (N,1,H,W)
        
    -> values: BINARY
    
    -> seg_mask_type: "ground truth" or "prediction" (affects how we 
                  transform the provided mask before computing the 
                  weight mask(s). If equal to "groun truth", will
                  will apply a threshold of 0 for binarizing pixels
                  in the image (if necessary))
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

    # Gets the weight mask(s)
    W = []
    if input_dimension == 4:
        
        # Obtains each mask separately
        for i in range(len(seg_mask)):
            #[H,W] ---> [1,H,W]
            unique_weight_mask = np.expand_dims(__get_weight_mask(seg_mask[i][0], weights_function), axis=0) 
            
            # Appends to the filtered_seg_mask vector (naturally, this vector will be [N,1,H,W])
            W.append(unique_weight_mask)
        
        # Converts into a numpy array
        W = np.array(W)

    elif input_dimension == 3:
        
        # Obtains the mask using the reduced dimension mask
        W = __get_weight_mask(seg_mask[0], weights_function)

        # Expands the dimension again for maintaining consistency with input shape
        W = np.expand_dims(W, axis=0) # [H,W] -> [1,H,W]
    
    else:
        # Obtains the mask
        W = __get_weight_mask(seg_mask, weights_function)

    return W

def main():

    # Loads the data
    example_components_path = "../tests/imgs/"   
    seg = Image.open(f"{example_components_path}DRIVE_seg_example.png")
    
    # Gets the filtered mask with only thin vessels

    #-----> for a [N,1,H,W] shape input
    seg_4_dims = np.expand_dims(np.expand_dims(np.array(seg), axis=0),axis=0)
    seg_4_dims = np.concatenate((seg_4_dims,seg_4_dims,seg_4_dims), axis=0)

    weight_mask = get_weight_mask(seg_4_dims, weights_function=1)
    print(f"Test 1: original shape: {seg_4_dims.shape} ---> weight masks shape: {weight_mask.shape}")
    
    #-----> for a [1,H,W] shape input
    seg_3_dims = np.expand_dims(np.array(seg), axis=0)
    
    weight_mask = get_weight_mask(seg_3_dims, weights_function=1)
    print(f"Test 2: original shape: {seg_3_dims.shape} ---> weight masks shape: {weight_mask.shape}")

    #-----> for a [H,W] shape input
    weight_mask = get_weight_mask(seg, weights_function=1)
    print(f"Test 3: original shape: {np.array(seg).shape} ---> weight masks shape: {weight_mask.shape}")

    # Shows the filtered segmentation mask
    print("Showing the resulting greyscale weight mask")
    print(f"Weights in the weight mask produced by W1 formulation over the DRIVE segmentation mask belong to the interval [{weight_mask.min()},{weight_mask.max()}]")
    # weight_mask_greyscale = ((weight_mask-weight_mask.min())/(weight_mask.max()-weight_mask.min()))*255
    # img = Image.fromarray(weight_mask_greyscale.astype(np.uint8))
    # img.show()

if __name__ == "__main__":
    main()