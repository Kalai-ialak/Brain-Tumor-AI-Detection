import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from model import unet_model

# 1. Data load and preprocess
img = nib.load('training_data/sample_brain.nii.gz')
data = img.get_fdata()
slice_idx = data.shape[2] // 2
slice_2d = data[:, :, slice_idx, 0]
X_input = np.expand_dims(np.expand_dims(slice_2d, axis=0), axis=-1)
X_input = X_input / np.max(X_input)

# 2. Model prediction
model = unet_model(input_size=(X_input.shape[1], X_input.shape[2], 1))
prediction = model.predict(X_input)

# 3. Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original MRI Slice")
plt.imshow(slice_2d, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("AI Tumor Detection")
plt.imshow(prediction[0, :, :, 0], cmap='hot')

print("\nSaving result as 'result.png'...")
plt.savefig('result.png')
print("Done! Open result.png to see what the AI found.")