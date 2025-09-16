import numpy as np
from sklearn.metrics import recall_score, precision_score
from PIL import Image
from retinal_thin_vessels.metrics import recall_thin_vessels, precision_thin_vessels
from retinal_thin_vessels.core import get_thin_vessels_mask
from retinal_thin_vessels.weights import get_weight_mask

def main():

    example_components_path = "imgs/"   
    # DRIVE IMAGES
    seg = Image.open(f"{example_components_path}DRIVE_seg_example.png")
    pred = Image.open(f"{example_components_path}DRIVE_pred_example.png")
    
    #------------------------Thin vessels filtered mask tests---------------------
    # # Gets the filtered mask with only thin vessels
    # thin_vessels_seg = get_thin_vessels_mask(seg)
    
    # print("Showing the filtered segmentation mask with thin vessels only. DRIVE")
    # img = Image.fromarray(thin_vessels_seg)
    # # img.show()
    # img.save("DRIVE_seg_thin_example.png")

    # # CHASEDB1 IMAGES
    # seg = Image.open(f"{example_components_path}CHASEDB1_seg_example.png")
    
    # # Gets the filtered mask with only thin vessels
    # thin_vessels_seg = get_thin_vessels_mask(seg)
    
    # print("Showing the filtered segmentation mask with thin vessels only. CHASEDB1")
    # img = Image.fromarray(thin_vessels_seg)
    # # img.show()
    # img.save("CHASEDB1_seg_thin_example.png")

    # Loads the ground truth segmentation mask and a sample prediction
    pred = Image.open(f"imgs/DRIVE_pred_example.png")
    seg_DRIVE = seg.resize((pred.size), Image.NEAREST)

    # Binarizes images to a 0/1 format for scikit-learn compatibility
    seg_DRIVE = np.where(np.array(seg_DRIVE) > 0, 1, 0)
    pred = np.where(np.array(pred) > 0, 1, 0)

    # Compute and print the metrics
    print(f"Overall Recall score: {recall_score(seg_DRIVE.flatten(), pred.flatten())}")
    print(f"Recall score on thin vessels: {recall_thin_vessels(seg_DRIVE, pred)}")
    print("-" * 30)
    print(f"Overall Precision score: {precision_score(seg_DRIVE.flatten(), pred.flatten())}")
    print(f"Precision score on thin Vessels: {precision_thin_vessels(seg_DRIVE, pred)}")
    
    #--------------------------------Weight masks tests--------------------------------

    W_0 = get_weight_mask(seg_DRIVE, weights_function=0)
    print(f"Weights in the weight mask produced by W0 formulation belong to [{W_0.min()},{W_0.max()}]")
    W_1 = get_weight_mask(seg_DRIVE, weights_function=1)
    print(f"Weights in the weight mask produced by W0 formulation belong to [{W_1.min()},{W_1.max()}]")

    # print(f"Showing the computed weight mask using W0 formulation")
    # W_0_greyscale = 255*(W_0 - W_0.min())/(W_0.max()-W_0.min())
    # img_show=Image.fromarray(W_0_greyscale.astype(np.uint8))
    # img_show.show()

    # print(f"Showing the computed weight mask using W1 formulation")
    # W_1_greyscale = 255*(W_1 - W_1.min())/(W_1.max()-W_1.min())
    # img_show=Image.fromarray(W_1_greyscale.astype(np.uint8))
    # img_show.show()

    exit(0)


if __name__ == "__main__":
    main()