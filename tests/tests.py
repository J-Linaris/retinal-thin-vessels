from PIL import Image
# from retinal_thin_vessels.metrics import recall_thin_vessels, precision_thin_vessels
from core import get_thin_vessels_mask
from metrics import recall_thin_vessels, precision_thin_vessels
import numpy as np
from sklearn.metrics import recall_score, precision_score

def main():

    example_components_path = "imgs/"   
    # DRIVE IMAGES
    img = Image.open(f"{example_components_path}DRIVE_img_example.png")
    seg = Image.open(f"{example_components_path}DRIVE_seg_example.png")
    pred = Image.open(f"{example_components_path}DRIVE_pred_example.png")
    
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

    print(np.array(seg.resize(pred.size, Image.NEAREST)).shape)
    print(np.array(pred).shape)
    print(np.unique(np.array(seg.resize(pred.size, Image.NEAREST)).astype(np.uint8)))
    print(np.unique((np.array(pred)/255).astype(np.uint8)))
    # Load the ground truth segmentation mask and a sample prediction
    pred = Image.open(f"imgs/DRIVE_pred_example.png")
    seg_DRIVE = Image.open(f"imgs/DRIVE_seg_example.png").resize((pred.size), Image.NEAREST)

    # Binarize images to a 0/1 format for scikit-learn compatibility
    seg_DRIVE = np.where(np.array(seg_DRIVE) > 0, 1, 0)
    pred = np.where(np.array(pred) > 0, 1, 0)

    # Compute and print the metrics
    print(f"Overall Recall score: {recall_score(seg_DRIVE.flatten(), pred.flatten())}")
    print(f"Recall score on thin vessels: {recall_thin_vessels(seg_DRIVE, pred)}")
    print("-" * 30)
    print(f"Overall Precision score: {precision_score(seg_DRIVE.flatten(), pred.flatten())}")
    print(f"Precision score on thin Vessels: {precision_thin_vessels(seg_DRIVE, pred)}")
    
    exit(0)


if __name__ == "__main__":
    main()