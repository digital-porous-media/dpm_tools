from dpm_tools.segmentation import statistical_region_merging as srm
from dpm_tools.segmentation import seeded_region_growing as srg
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(130621)
image = np.random.randint(0, 256, size=(10, 10, 10), dtype=np.uint8)
seeds = np.random.randint(0, 5, size=(10, 10, 10), dtype=np.uint8)

# segmented = srg(image[0].astype(np.uint16), seeds[0], normalize=False)
# print(segmented.dtype)
# srm_obj.segment()
segmented = srm(image[0].astype(np.uint16), Q=5.0)

# Plot the result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image[0, :, :])
ax[0].set_title("Original Image")
ax[1].imshow(segmented)  # [0, :, :])
ax[1].set_title("Segmented Image")
plt.show()
