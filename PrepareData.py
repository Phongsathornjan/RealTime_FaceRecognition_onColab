import cv2
import os

# กำหนดพาธของโฟลเดอร์
dataset_folder = 'Rawdataset'
photodataset_folder = 'dataset'

# สร้างโฟลเดอร์สำหรับบันทึกภาพถ้ายังไม่มี
if not os.path.exists(photodataset_folder):
    os.makedirs(photodataset_folder)

# โหลด cascade classifier สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ฟังก์ชันในการตรวจจับและบันทึกใบหน้า
def detect_and_save_faces(image_path, output_folder):
    # อ่านภาพ
    image = cv2.imread(image_path)
    if image is None: # Check if image was loaded correctly
        print(f"Could not read image: {image_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # ถ้าพบใบหน้า
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # ครอปใบหน้า
            face_crop = image[y-50:(y+50)+h, x-50:(x+50)+w]
            
            # สร้างชื่อไฟล์สำหรับการบันทึก
            base_name = os.path.basename(image_path)
            file_name, ext = os.path.splitext(base_name)
            output_file = os.path.join(output_folder, f'{file_name}_face_{i}.jpg') # Use .jpg extension
            
            # บันทึกใบหน้า
            cv2.imwrite(output_file, face_crop)

# โหลดภาพจากโฟลเดอร์ dataset
for filename in os.listdir(dataset_folder):
    image_path = os.path.join(dataset_folder, filename)
    if os.path.isfile(image_path):
        detect_and_save_faces(image_path, photodataset_folder)

print("การตรวจจับและบันทึกใบหน้าสำเร็จ")