import os
import cv2
import numpy as np

def resize_and_crop_stripe(img, crop_top=2, crop_bottom=2, crop_left=5, crop_right=5, target_size=(128, 64)):
    """
    Crop edges and resize image to target size.
    Args:
        img: Original image as numpy array
        crop_top, crop_bottom, crop_left, crop_right: Pixels to crop from each side
        target_size: Final size (height, width)
    Returns:
        Processed image
    """
    h, w = img.shape[:2]
    cropped = img[crop_top:h - crop_bottom, crop_left:w - crop_right]
    resized = cv2.resize(cropped, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return resized

def main():
    input_folder = "raw"
    output_folder = "resized"
    os.makedirs(output_folder, exist_ok=True)

    # Parameters
    crop_top = 1
    crop_bottom = 1
    crop_left = 5
    crop_right = 5
    target_size = (128, 256)  # (height, width)

    # Supported image formats
    image_extensions = (".png", ".jpg", ".jpeg")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(image_extensions):
            continue
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        processed_img = resize_and_crop_stripe(
            img,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_left=crop_left,
            crop_right=crop_right,
            target_size=target_size
        )

        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, processed_img)

    print(f"✅ Done processing. Output saved to: {output_folder}")

main()
