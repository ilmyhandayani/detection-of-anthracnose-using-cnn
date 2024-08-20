import cv2
import os

# Menentukan direktori gambar
image_dir = "D:/KULIAH/SEMESTER 6/Proyek Profesional/website/web/data/normal"

# Mendapatkan daftar file gambar di direktori
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".jpg") or file.endswith(".png")]

# Looping melalui setiap gambar dan menaikkan kecerahan
for image_file in image_files:
    # Membaca gambar menggunakan OpenCV
    image = cv2.imread(image_file)

    # Menambahkan nilai kecerahan ke gambar
    brightness_value = 50
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = cv2.add(v, brightness_value)
    final_hsv_image = cv2.merge((h, s, v))
    bright_image = cv2.cvtColor(final_hsv_image, cv2.COLOR_HSV2BGR)

    # Menyimpan gambar yang telah dinaikkan kecerahannya
    cv2.imwrite(image_file, bright_image)
print('selesai')