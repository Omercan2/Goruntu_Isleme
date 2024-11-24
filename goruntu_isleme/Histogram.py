import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


image_folder = r"C:\Users\omerc\Desktop\goruntu_isleme"


output_folder = os.path.join(image_folder, "histogramlar")
os.makedirs(output_folder, exist_ok=True)


image_files = [f for f in os.listdir(image_folder) if f.endswith(".tif")]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)


    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    if image is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        continue

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 5))

  
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Görüntü: {image_file}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Frekans")

    plt.tight_layout()


    output_filename = f"histogram_{image_file.replace('.tif', '')}.png"
    output_path = os.path.join(output_folder, output_filename)

    plt.savefig(output_path)
    print(f"Histogram kaydedildi: {output_path}")

    plt.close()
