import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


image_folder = r"C:\Users\omerc\Desktop\goruntu_isleme\binarized_images"   


new_folder_name = "morphological_results"
output_folder = os.path.join(r"C:\Users\omerc\Desktop\goruntu_isleme", new_folder_name)


os.makedirs(output_folder, exist_ok=True)

binary_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

kernel = np.ones((3, 3), np.uint8)

for binary_file in binary_files:

    binary_path = os.path.join(image_folder, binary_file)  
   
    binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

   
    erosion = cv2.erode(binary_image, kernel, iterations=1)
    
  
    dilation = cv2.dilate(binary_image, kernel, iterations=1)
    

    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    

    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f"Orijinal İkili Görüntü: {binary_file}")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(erosion, cmap='gray')
    plt.title("Erozyon")
    plt.axis('off')


    plt.subplot(2, 3, 3)
    plt.imshow(dilation, cmap='gray')
    plt.title("Şişirme")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(opening, cmap='gray')
    plt.title("Açma")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(closing, cmap='gray')
    plt.title("Kapatma")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    output_image_path_erosion = os.path.join(output_folder, binary_file.replace('.png', '_erosion.png'))
    output_image_path_dilation = os.path.join(output_folder, binary_file.replace('.png', '_dilation.png'))
    output_image_path_opening = os.path.join(output_folder, binary_file.replace('.png', '_opening.png'))
    output_image_path_closing = os.path.join(output_folder, binary_file.replace('.png', '_closing.png'))


    cv2.imwrite(output_image_path_erosion, erosion)
    cv2.imwrite(output_image_path_dilation, dilation)
    cv2.imwrite(output_image_path_opening, opening)
    cv2.imwrite(output_image_path_closing, closing)

    print(f"{binary_file} için morfolojik işlemler kaydedildi.")
