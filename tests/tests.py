from PIL import Image
from recal_thin_vessels.metrics import recall_thin_vessels, precision_thin_vessels
from recal_thin_vessels.core import get_thin_vessels_mask

def main():

    example_components_path = "imgs/"   
    img = Image.open(f"{example_components_path}img_example.png")
    seg = Image.open(f"{example_components_path}seg_example.png")
    pred = Image.open(f"{example_components_path}pred_example.png")
    
    # Gets the filtered mask with only thin vessels
    thin_vessels_seg = get_thin_vessels_mask(seg)
    
    print("Showing the filtered segmentation mask with thin vessels only.")
    img = Image.fromarray(thin_vessels_seg)
    img.show()

    print(recall_thin_vessels(seg.resize(pred.size, Image.NEAREST), pred))
    # print(__precision_thin_vessels_single_image(seg.resize(pred.size, Image.NEAREST), pred))

    exit(0)


if __name__ == "__main__":
    main()