# retinal_thin_vessels

A Python package for computing the recall and precision scores specifically on thin vessels in retinal images and generating weight masks for BCE Loss to enhance models perfomance on segmenting these fine structures, as detailed in the paper "Vessel-Width-Based Metrics and Weight Masks for Retinal Blood Vessel Segmentation", published in WUW-SIBGRAPI 2025. The package also includes a function for visualizing thickness-based filtered masks, the basic structure for computing the proposed metrics.

It is worth stating that the functions for computing these metrics and the function for obtaining the weight masks accept, as input:
- A batch of segmentation images
- A single segmentation image (with or without the channels dimension)

In order to better understand this, you may find helpful to read the documentation of these functions.

## Package installation

```bash
pip install retinal_thin_vessels
```

## Usage Demonstration with DRIVE and CHASEDB1

### Recall and Precision on Thin Vessels Metrics
To ensure the metrics are reliable, it is important to visualize the specific thin-vessel mask used by the given functions in their calculations. Therefore, a core function, get_thin_vessels_mask(), is also provided. This function takes a standard segmentation mask and returns a new mask containing only the thin vessels.

The following code demonstrates how to generate this filtered mask using images from two public datasets: DRIVE and CHASEDB1.

```python
from PIL import Image
from retinal_thin_vessels.core import get_thin_vessels_mask
from retinal_thin_vessels.metrics import recall_thin_vessels, precision_thin_vessels
from sklearn.metrics import recall_score, precision_score
```

```python
# Import the original segmentation masks
seg_DRIVE = Image.open(f"tests/imgs/DRIVE_seg_example.png")
seg_CDB1 = Image.open(f"tests/imgs/CHASEDB1_seg_example.png")

# generates new masks containing only thin vessels
thin_vessels_seg_DRIVE = get_thin_vessels_mask(seg_DRIVE)
thin_vessels_seg_CDB1 = get_thin_vessels_mask(seg_CDB1)

# Display the original segmentation mask and the resulting thin-vessel-only mask for comparison
seg_DRIVE.show()
img_DRIVE = Image.fromarray(thin_vessels_seg_DRIVE)
img_DRIVE.show()

seg_CDB1.show()
img_CDB1 = Image.fromarray(thin_vessels_seg_CDB1)
img_CDB1.show()
```
<img src="tests/imgs/DRIVE_seg_example.png" alt="DRIVE_thin_vessels_example" width=450/>
<img src="tests/imgs/DRIVE_seg_thin_example.png" alt="DRIVE_thin_vessels_example" width=450/>
<img src="tests/imgs/CHASEDB1_seg_example.png" alt="CHASEDB1_thin_vessels_example" width=450/>
<img src="tests/imgs/CHASEDB1_seg_thin_example.png" alt="CHASEDB1_thin_vessels_example" width=450/>

Furthermore, to demonstrate the metric calculation functions, you can run the code below. It compares the overall metrics (calculated with scikit-learn) with the thin-vessel-specific metrics calculated by this package.

```python
# Load the ground truth segmentation mask and a sample prediction
pred = Image.open(f"tests/imgs/DRIVE_pred_example.png")
seg_DRIVE = Image.open(f"tests/imgs/DRIVE_seg_example.png").resize((pred.size), Image.NEAREST)

# Binarize images to a 0/1 format for scikit-learn compatibility
seg_DRIVE = np.where(np.array(seg_DRIVE) > 0, 1, 0)
pred = np.where(np.array(pred) > 0, 1, 0)

# Compute and print the metrics
print(f"Overall Recall score: {recall_score(seg_DRIVE.flatten(), pred.flatten())}")
print(f"Recall score on thin vessels: {recall_thin_vessels(seg_DRIVE, pred)}")
print("-" * 30)
print(f"Overall Precision score: {precision_score(seg_DRIVE.flatten(), pred.flatten())}")
print(f"Precision score on thin Vessels: {precision_thin_vessels(seg_DRIVE, pred)}")
```

If the program is running correctly with the provided sample images, the results should be similar to this:

```bash
Overall Recall score: 0.8553852359822509
Recall score on thin vessels: 0.751244555071562
------------------------------
Overall Precision score: 0.8422369623068674
Precision score on thin Vessels: 0.6527915897144481
```

### Weight masks

In the paper, it is proposed two weight masks formulations for setting the weight for a pixel $i$:
- W0 formulation: $$W_i = \frac{2}{R^2}$$
- W1 formulation: $$W_i = \frac{D_i+1}{R^2}$$

where, respectively, $R$ and $D_i$ refer to the radius of the vessel to which the pixel belongs to and the pixel's distance to the background. 

The following code demonstrates how to generate weight masks using images from two public datasets: DRIVE and CHASEDB1.

```python
from PIL import Image
from retinal_thin_vessels.weights import get_weight_mask
```

```python
# Import the original segmentation masks
seg_DRIVE = Image.open(f"tests/imgs/DRIVE_seg_example.png")
seg_CDB1 = Image.open(f"tests/imgs/CHASEDB1_seg_example.png")

# Generates the weight masks using the W1 formulation (just for example)
W_1_DRIVE = get_weight_mask(seg_DRIVE, weights_function=1)
W_1_CHASEDB1 = get_weight_mask(seg_CDB1, weights_function=1)
print(f"Weights in the weight mask produced by W1 formulation over the DRIVE segmentation mask belong to the interval [{W_1_DRIVE.min()},{W_1_DRIVE.max()}]")
print(f"Weights in the weight mask produced by W1 formulation over the CHASEDB1 segmentation mask belong to the interval [{W_1_CDB1.min()},{W_1_CDB1.max()}]")

# Displays a greyscale image for each computed weight mask followed by the segmentation mask
seg_DRIVE.show()
W_1_DRIVE_greyscale = 255*(W_1_DRIVE - W_1_DRIVE.min())/(W_1_DRIVE.max()-W_1_DRIVE.min())
img_DRIVE = Image.fromarray(W_1_DRIVE_greyscale.astype(np.uint8))
img_DRIVE.show()

seg_CDB1.show()
W_1_CDB1_greyscale = 255*(W_1_CDB1 - W_1_CDB1.min())/(W_1_CDB1.max()-W_1_CDB1.min())
img_CDB1 = Image.fromarray(W_1_CDB1_greyscale.astype(np.uint8))
img_CDB1.show()
```
If the program is running correctly with the provided sample images, the results should be similar to this:

```bash
Weights in the weight mask produced by W1 formulation over the DRIVE segmentation mask belong to the interval [0.0,3.2360680103302]
Weights in the weight mask produced by W1 formulation over the CHASEDB1 segmentation mask belong to the interval [0.0,3.0]
```

<img src="tests/imgs/DRIVE_seg_example.png" alt="CHASEDB1_thin_vessels_example.png" width=450/>
<img src="tests/imgs/DRIVE_W1_grey_example.png" alt="DRIVE_W1_greyscale_weight_mask_example.png" width=450/>
<img src="tests/imgs/CHASEDB1_seg_example.png" alt="CHASEDB1_thin_vessels_example.png" width=450/>
<img src="tests/imgs/CHASEDB1_W1_grey_example.png" alt="CHASEDB1_W1_greyscale_weight_mask_example.png" width=450/>

## Overall view

According to the study conducted in the referred paper (in which it was used the U-Net architecture and the BCE loss), the expected effect of each weight mask is:

- W0 mask: enhances models ability to preserve vessel architecture (high precision on thin vessels and lower recall on thin vessels)
- W1 mask: enhances models ability in correctly detecting thin vessels, at the expense of anatomical fidelity (high recall on thin vessels and lower precision on thin vessels)

Therefore, it was noticed a kind of opposite behavior provoked by each one of these masks. This conclusion and both statements above are supported by the results in the following table:
<p align="center">
  <img src="tests/misc/table_weight_masks.png" alt="weight_masks_table.png" width=900/>
</p>

OBS: Standard weight mask (Std) stands for sklearn's compute_class_weight function, that aims soly on balancing the impact of each class in the loss function. Therefore, it only makes white and black pixels have the same impact on the loss function (it was used as the baseline for the paper). Morevoer, "WBCE" stands for Weighted Binary Cross Entropy loss.
