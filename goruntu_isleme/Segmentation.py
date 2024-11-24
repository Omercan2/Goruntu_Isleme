import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

image_folder = r"C:\Users\omerc\Desktop\goruntu_isleme"  

output_folder = os.path.join(image_folder, "segmented_images")
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.endswith('.tif')]


for image_file in image_files:

    image_path = os.path.join(image_folder, image_file)
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

    thresholds = [50, 100, 150]  

    mask1 = cv2.inRange(image, 0, thresholds[0])  
    mask2 = cv2.inRange(image, thresholds[0], thresholds[1])  
    mask3 = cv2.inRange(image, thresholds[1], thresholds[2])  
    mask4 = cv2.inRange(image, thresholds[2], 255)  
    
    color_segmented_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  


    color_segmented_image[mask1 == 255] = [0, 0, 255]  
    color_segmented_image[mask2 == 255] = [0, 255, 0]  
    color_segmented_image[mask3 == 255] = [255, 0, 0] 
    color_segmented_image[mask4 == 255] = [0, 255, 255]  

  
    plt.figure(figsize=(10, 5))

 
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Orijinal Görüntü: {image_file}")
    plt.axis('off')

   
    plt.subplot(1, 2, 2)
    plt.imshow(color_segmented_image)
    plt.title(f"Bölütlenmiş Görüntü (Renkli)")
    plt.axis('off')

  
    plt.tight_layout()
    plt.show()

 
    output_image_path = os.path.join(output_folder, image_file.replace('.tif', '_segmented.png'))

    color_segmented_image_rgb = cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_image_path, color_segmented_image_rgb)

    print(f"{image_file} için bölütlenmiş görüntü kaydedildi.")
