import nibabel as nib
import matplotlib.pyplot as plt
import os

# Download panna file path
file_path = os.path.join('data', 'sample_brain.nii.gz')

if os.path.exists(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    print("Data Loaded Successfully!")
    print(f"Image Shape: {data.shape}")

    # Display middle slice
    if len(data.shape) == 4:
        slice_to_show = data[:, :, data.shape[2]//2, 0]
    else:
        slice_to_show = data[:, :, data.shape[2]//2]

    plt.imshow(slice_to_show, cmap='gray')
    plt.title("My First Brain MRI Slice")
    plt.axis('off')
    plt.show()
else:
    print(f"File kanala! Path check pannunga: {os.path.abspath(file_path)}")