import numpy as np
import matplotlib.pyplot as plt
from utils_pipeline import ErrorMetrics, getPSNR
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Process input data file and output folder.')
parser.add_argument('--input_file', type=str, help='Path to the input data file (.npz)')
parser.add_argument('--output_folder', type=str, help='Path to the output folder')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()

print("Start")

input_file = args.input_file
output_folder = args.output_folder
verbose = args.verbose

# Load the .npz file
data = np.load(input_file)

gt = data["ground_truth"]
recon = data["reconstruction"]
masks = data["masks"]
inputs = data["inputs"]
errs = np.abs(gt - np.abs(recon))

SSIM_array = []
NMSE_array = []
PSNR_array = []


for i in range(400):
    c = ErrorMetrics(gt[i], recon[i])
    c.calc_SSIM()
    c.calc_NMSE()
    SSIM_array.append(c.SSIM)
    NMSE_array.append(c.NMSE)
    PSNR_array.append(getPSNR(np.abs(gt[i]),  np.abs(recon[i])))
    if verbose:
        print(
            f"Sample {i}: SSIM: {SSIM_array[-1]} NMSE: {NMSE_array[-1]} PSNR: {PSNR_array[-1]}"
        )


print("Statistics:")
print("NMSE")
# Convert to string to avoid floating point rounding errors.
print(
    f"{str(round(np.mean(NMSE_array), 6))[:8]} ± {str(round(np.std(NMSE_array), 6))[:8]}"
)

print("SSIM")
print(f"{round(np.mean(SSIM_array), 4)} ± {round(np.std(SSIM_array), 4)}")

print("PSNR")
print(f"{round(np.mean(PSNR_array), 4)} ± {round(np.std(PSNR_array), 4)}")

# Create directories if they don't exist
os.makedirs(os.path.join(output_folder, 'concatenated_images'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'error_images'), exist_ok=True)

# Loop over the arrays and concatenate images
for i, (recon_img, gt_img, err_img) in enumerate(zip(recon, gt, errs), start=1):
    # Rotate images by 180 degrees
    rotated_recon = np.rot90(recon_img, 2)
    rotated_gt = np.rot90(gt_img, 2)
    rotated_err = np.rot90(err_img, 2)
    
    # Concatenate the ground truth and reconstruction images horizontally
    concatenated_img = np.concatenate((rotated_gt, np.abs(rotated_recon)), axis=1)
    
    # Save concatenated image in concatenated_images folder
    plt.imsave(os.path.join(output_folder, f'concatenated_images/{i}.png'), concatenated_img, cmap='gray')

    # Save error image in error_images folder
    plt.imsave(os.path.join(output_folder, f'error_images/{i}.png'), rotated_err, cmap='gray')

    # print(f'Saved concatenated image {i}.png and error image {i}.png to {output_folder}')

