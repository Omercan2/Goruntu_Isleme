import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

image_folder = r"C:\Users\omerc\Desktop\goruntu_isleme"

output_folder = os.path.join(image_folder, "binarized_images")
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.endswith('.tif')]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"{image_file} yüklenemedi.")
        continue
    
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    threshold_value = 70 

    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    output_path = os.path.join(output_folder, f"binary_{image_file.replace('.tif', '')}.png")
    cv2.imwrite(output_path, binary_image)
    print(f"İkilileştirilmiş görüntü kaydedildi: {output_path}")
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Orijinal Görüntü: {image_file}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f"İkilileştirilmiş Görüntü ({threshold_value})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
