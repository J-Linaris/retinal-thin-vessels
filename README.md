# retinal_thin_vessels

A Python package for computing segmentation metrics specifically on thin vessels in retinal images, as detailed in our paper: {LINK TO YOUR PAPER}.

## How to install the package

```bash
pip install retinal_thin_vessels
```

## Functions demonstrations using DRIVE and CHASEDB1

This package offers functions for computing the recall and precision metrics on thin vessels. However, it's necessary to understand the ground truth these functions consider, in order to give these metrics reliability. Therefore, we offer, aswel, a function that returns the filtered mask when passed a segmentation mask. Below, the code for generating this masks using three diferent public datasets: DRIVE and CHASEDB1 is shown in order to exemplify the usage of this package. 

```python
from PIL import Image
from retinal_thin_vessels.core import get_thin_vessels
from retinal_thin_vessels.metrics import recall_thin_vessels, precision_thin_vessels
from sklearn.metrics import recall_score, precision_score
```

```python
# Imports the original segmentation masks
seg_DRIVE = Image.open(f"tests/imgs/DRIVE_seg_example.png")
seg_CDB1 = Image.open(f"tests/imgs/CHASEDB1_seg_example.png")

# Gets the filtered masks with only thin vessels
thin_vessels_seg_DRIVE = get_thin_vessels_mask(seg_DRIVE)
thin_vessels_seg_CDB1 = get_thin_vessels_mask(seg_CDB1)

# Displays the image
img = Image.fromarray(thin_vessels_seg_DRIVE)
img.show()
img = Image.fromarray(thin_vessels_seg_CDB1)
img.show()
```

<img src="tests/imgs/DRIVE_seg_thin_example.png" alt="DRIVE_thin_vessels_example" width=400/>
<img src="tests/imgs/CHASEDB1_seg_thin_example.png" alt="CHASEDB1_thin_vessels_example" width=400/>

Furthermore, to check if the metrics calculation is working, you can run the code below:

```python
# Imports the original segmentation masks and the prediction
pred = Image.open(f"tests/imgs/DRIVE_pred_example.png")
seg_DRIVE = Image.open(f"tests/imgs/DRIVE_seg_example.png").resize((pred.size), Image.NEAREST)

# Adequates both images (necessary for scikit recall and precision scores)
seg_DRIVE = np.where(np.array(seg_DRIVE) > 0, 1, 0)
pred = np.where(np.array(pred) > 0, 1, 0)

# Computes the metrics
print(f"Overall Recall score: {recall_score(seg_DRIVE.reshape(-1), pred.reshape(-1))}")
print(f"Recall score on thin vessels: {recall_thin_vessels(seg_DRIVE, pred)}")
print(f"Overall Precision score: {precision_score(seg_DRIVE.reshape(-1), pred.reshape(-1))}")
print(f"Precision score on thin Vessels: {precision_thin_vessels(seg_DRIVE, pred)}")
```

If the program is running correctly, the results will be something like for the provided images:

```bash
Overall Recall score: 0.8553852359822509
Recall score on thin vessels: 0.7512430080795525
Overall Precision score: 0.8422369623068674
Precision score on thin Vessels: 0.6528291157236281
```
