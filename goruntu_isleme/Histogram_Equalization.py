import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

base_folder = r"C:\Users\omerc\Desktop\goruntu_isleme"
output_base_folder = r"C:\Users\omerc\Desktop\goruntu_isleme\histogram_esitleme_sonrasi ciktilari"
os.makedirs(output_base_folder, exist_ok=True)

os.makedirs(os.path.join(output_base_folder, "original_histogram"), exist_ok=True)
os.makedirs(os.path.join(output_base_folder, "equalized_histogram"), exist_ok=True)

tasks = {
    "2_binarization": "İkilileştirme",
    "3_segmentation": "Bölütleme",
    "4_morphological_operations": "Morfolojik_İşlemler",
    "5_region_growing": "Bölge_Genişletme"
}
for task_key, task_name in tasks.items():
    os.makedirs(os.path.join(output_base_folder, task_key), exist_ok=True)

image_files = [f for f in os.listdir(base_folder) if f.endswith('.tif')]

def plot_histogram(image, title, save_path=None):
    hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])
    plt.figure(figsize=(8, 6))
    plt.plot(hist, color='black')
    plt.title(title)
    plt.xlabel("Yoğunluk Değeri")
    plt.ylabel("Piksel Sayısı")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def perform_binarization(image, filename):
    threshold_value = 128 
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    output_path = os.path.join(output_base_folder, "2_binarization", filename.replace('.tif', '_binary.png'))
    cv2.imwrite(output_path, binary_image)
    return binary_image

def perform_segmentation(image, filename):
    thresholds = [50, 100, 150]
    mask1 = cv2.inRange(image, 0, thresholds[0])
    mask2 = cv2.inRange(image, thresholds[0], thresholds[1])
    mask3 = cv2.inRange(image, thresholds[1], thresholds[2])
    mask4 = cv2.inRange(image, thresholds[2], 255)

    segmented_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    segmented_image[mask1 == 255] = [255, 0, 0]
    segmented_image[mask2 == 255] = [0, 255, 0]
    segmented_image[mask3 == 255] = [0, 0, 255]
    segmented_image[mask4 == 255] = [255, 255, 0]

    output_path = os.path.join(output_base_folder, "3_segmentation", filename.replace('.tif', '_segmented.png'))
    cv2.imwrite(output_path, segmented_image)
    return segmented_image

def perform_morphological_operations(binary_image, filename):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    operations = {
        "erosion": cv2.erode(binary_image, kernel, iterations=1),
        "dilation": cv2.dilate(binary_image, kernel, iterations=1),
        "opening": cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    }
    for op_name, result in operations.items():
        output_path = os.path.join(output_base_folder, "4_morphological_operations", f"{filename.replace('.tif', f'_{op_name}.png')}")
        cv2.imwrite(output_path, result)

def perform_region_growing(image, segmented_image, filename):
    def region_growing(image, seed_point, threshold=20):
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

    mask = cv2.inRange(segmented_image, np.array([0, 255, 0]), np.array([0, 255, 0])) 
    seed_points = np.argwhere(mask == 255)
    if len(seed_points) > 0:
        seed_point = tuple(seed_points[np.random.choice(seed_points.shape[0])])
        region = region_growing(image, seed_point)
        region_colored = cv2.merge([region, region, region])
        output_path = os.path.join(output_base_folder, "5_region_growing", filename.replace('.tif', '_region_growing.png'))
        cv2.imwrite(output_path, region_colored)

for image_file in image_files:
    image_path = os.path.join(base_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    plot_histogram(image, f"{image_file} - Histogram Eşitleme Öncesi", 
                   save_path=os.path.join(output_base_folder, "original_histogram", f"{image_file.replace('.tif', '_original_histogram.png')}"))
    

    hist_equalized_image = cv2.equalizeHist(image)

    plot_histogram(hist_equalized_image, f"{image_file} - Histogram Eşitleme Sonrası", 
                   save_path=os.path.join(output_base_folder, "equalized_histogram", f"{image_file.replace('.tif', '_equalized_histogram.png')}"))

    binary_image = perform_binarization(hist_equalized_image, image_file)
    segmented_image = perform_segmentation(hist_equalized_image, image_file)
    perform_morphological_operations(binary_image, image_file)
    perform_region_growing(hist_equalized_image, segmented_image, image_file)

print("Tüm işlemler tamamlandı ve sonuçlar ilgili klasörlere kaydedildi.")
