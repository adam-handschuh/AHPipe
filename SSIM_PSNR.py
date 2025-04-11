# Image Quality Metrics - SSIM and PSNR Calculator
import numpy as np
import cv2
from skimage.metrics import structural_similarity
import math
import argparse

def calculate_ssim(original_path, modified_path):
    """
    Calculate the SSIM between an original and modified image.
    
    Args:
        original_path (str): Path to the original image
        modified_path (str): Path to the modified image
        
    Returns:
        float: SSIM value
    """
    # Load images
    original = cv2.imread(original_path)
    modified = cv2.imread(modified_path)
    
    # Convert to grayscale if images are in color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        
    if len(modified.shape) == 3:
        modified_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
    else:
        modified_gray = modified
    
    # Check if images are the same size
    if original_gray.shape != modified_gray.shape:
        print("Error: Images must have the same dimensions")
        return None
    
    # Calculate SSIM
    # The function returns the SSIM index and the gradient images
    (score, diff) = structural_similarity(original_gray, modified_gray, full=True)
    
    # Print SSIM score
    print(f"SSIM: {score:.6f}")
    
    return score

def calculate_psnr(original_path, noisy_path):
    """
    Calculate PSNR between an original and a noisy/processed image.
    
    Args:
        original_path (str): Path to the original image
        noisy_path (str): Path to the noisy/processed image
        
    Returns:
        float: PSNR value in dB
    """
    # Load images
    original = cv2.imread(original_path)
    noisy = cv2.imread(noisy_path)
    
    # Convert to grayscale if images are in color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        
    if len(noisy.shape) == 3:
        noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    else:
        noisy_gray = noisy
    
    # Check if images are the same size
    if original_gray.shape != noisy_gray.shape:
        print("Error: Images must have the same dimensions")
        return None
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_gray - noisy_gray) ** 2)
    
    if mse == 0:  # Images are identical
        return float('inf')
    
    # Calculate PSNR
    # Assuming 8-bit images (max pixel value is 255)
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    # Print PSNR value
    print(f"PSNR: {psnr:.2f} dB")
    
    return psnr

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate SSIM and PSNR between images")
    parser.add_argument("--original", required=True, help="Path to the original image")
    parser.add_argument("--modified", required=True, help="Path to the modified/noisy image")
    parser.add_argument("--metric", choices=["ssim", "psnr", "both"], default="both", 
                        help="Metric to calculate: ssim, psnr, or both")
    
    args = parser.parse_args()
    
    # Calculate requested metrics
    if args.metric == "ssim" or args.metric == "both":
        print("Calculating SSIM...")
        ssim = calculate_ssim(args.original, args.modified)
    
    if args.metric == "psnr" or args.metric == "both":
        print("Calculating PSNR...")
        psnr = calculate_psnr(args.original, args.modified)

if __name__ == "__main__":
    main()