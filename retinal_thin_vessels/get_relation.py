import numpy as np
from skimage.morphology import medial_axis, area_closing
from PIL import Image
from retinal_thin_vessels.external.DSE_skeleton_pruning.dsepruning.dsepruning import skel_pruning_DSE

def main():

    seg_mask = np.array(Image.open("imgs/DRIVE_seg_example.png"))
    print(seg_mask.shape)
    exit(0)
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


    exit(0)

main()