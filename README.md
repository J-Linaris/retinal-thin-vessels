# retinal_thin_vessels

A Python package for computing the recall and precision scores specifically on thin vessels in retinal images, as detailed in our paper: {LINK TO YOUR PAPER}.

## Package installation

```bash
pip install retinal_thin_vessels
```

## Usage Demonstration with DRIVE and CHASEDB1

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

# Display the resulting thin-vessel-only mask for visualization
img_DRIVE = Image.fromarray(thin_vessels_seg_DRIVE)
img_DRIVE.show()
img_CDB1 = Image.fromarray(thin_vessels_seg_CDB1)
img_CDB1.show()
```

<img src="tests/imgs/DRIVE_seg_thin_example.png" alt="DRIVE_thin_vessels_example" width=400/>
<img src="tests/imgs/CHASEDB1_seg_thin_example.png" alt="CHASEDB1_thin_vessels_example" width=400/>

Furthermore, to demonstrate the metric calculation functions, you can run the code below. It compares the overall metrics (calculated with scikit-learn) to the thin-vessel-specific metrics calculated by this package.

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
