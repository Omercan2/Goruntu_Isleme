import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


image_folder = r"C:\Users\omerc\Desktop\goruntu_isleme" 

region_growing_output_folder = os.path.join(image_folder, "region_growing_results")
os.makedirs(region_growing_output_folder, exist_ok=True)


def region_growing(image, seed_point, threshold=20):
    """ Bölge genişletme algoritması """
    height, width = image.shape
    visited = np.zeros_like(image, dtype=bool)
    region = np.zeros_like(image, dtype=np.uint8)


    seed_value = image[seed_point]
    queue = [seed_point]
    visited[seed_point] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.pop(0)
        region[x, y] = 255  

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:

                if abs(int(image[nx, ny]) - int(seed_value)) <= threshold:
                    visited[nx, ny] = True
                    queue.append((nx, ny))

    return region

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
    
    masks = [mask1, mask2, mask3, mask4]
    colors = [
        [255, 0, 0],   
        [0, 255, 0],   
        [0, 0, 255],   
        [255, 255, 0]  
    ]
    region_growing_result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for idx, mask in enumerate(masks):

        seed_points = np.argwhere(mask == 255)
        if len(seed_points) > 0:
            seed_point = tuple(seed_points[np.random.choice(seed_points.shape[0])])
            region = region_growing(image, seed_point)


            for c in range(3):
                region_growing_result[:, :, c] += (region // 255 * colors[idx][c])

    region_growing_output_path = os.path.join(region_growing_output_folder, image_file.replace('.tif', '_region_growing.png'))
    cv2.imwrite(region_growing_output_path, cv2.cvtColor(region_growing_result, cv2.COLOR_BGR2RGB))

    print(f"{image_file} için region growing sonucu kaydedildi.")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Orijinal")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmentasyon")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(region_growing_result)
    plt.title("Region Growing")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
