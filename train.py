import nibabel as nib
import numpy as np
import tensorflow as tf
from model import unet_model

# 1. Data-va load pannuvom
img = nib.load('training_data/sample_brain.nii.gz')
data = img.get_fdata()

# 2. Data-va model-ku yetha madhiri shape panna porom (Preprocessing)
# Sample data 4D-ah irukkum, adhai 2D slices-ah maathuvom
slice_2d = data[:, :, 10, 0] # Oru middle slice edukkarom
X_train = np.expand_dims(np.expand_dims(slice_2d, axis=0), axis=-1)
X_train = X_train / np.max(X_train) # Normalize to 0-1 range

# 3. Model-ah load pannuvom
model = unet_model(input_size=(X_train.shape[1], X_train.shape[2], 1))

# 4. Dummy Training (Start panni check pannuvom)
print("\nStarting Training...")
model.fit(X_train, X_train, epochs=5) 

print("\nTraining Completed Successfully!")