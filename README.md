# 🧠 Brain Tumor Classification (MRI)

> Phân loại ảnh MRI não thành **có u / không có u** bằng transfer learning (VGG16, ResNet50, InceptionV3, MobileNet) và CNN tự xây dựng, kèm web app Flask để tải ảnh lên và dự đoán.

Đồ án môn học – Hoàng Quang Huy, Trường Đại học Sư phạm Kỹ thuật Hưng Yên (2024). Báo cáo dạng bài báo: [`Brain_Tumor_Classification (3).pdf`](<Brain_Tumor_Classification (3).pdf>), video demo: [`2024-10-07 23-44-21.mkv`](<2024-10-07 23-44-21.mkv>).

| Hạng mục | Nội dung |
|:---|:---|
| Bài toán | Phân loại nhị phân: `yes` (có u não) / `no` (không có u não) |
| Dữ liệu | Brain MRI Images for Brain Tumor Detection (Kaggle): 155 ảnh có u, 98 ảnh không u |
| Tăng cường dữ liệu | `ImageDataGenerator` (xoay, dịch, shear, độ sáng, lật ngang/dọc) → **2.065 ảnh** (1.085 yes / 980 no) |
| Chia tập | 80 / 10 / 10 → 1.652 train, 207 validation, 206 test |
| Mô hình tốt nhất | **VGG16** (ImageNet, đóng băng backbone) + Dense 256 + Dropout 0.5 + Softmax 2 lớp |
| Kết quả tốt nhất | Test accuracy **98%**, K-fold (5 fold) trung bình **96,71% ± 0,90** |
| Triển khai | Flask + HTML/CSS/JS, model `weights/vgg16_model.h5` |

---

## 1. Pipeline

```
Ảnh MRI gốc (yes/no)
   │  đổi tên, gom thư mục
   ▼
Tăng cường dữ liệu (ImageDataGenerator)  ──►  2.065 ảnh
   │  crop vùng não (Gaussian blur → threshold → erode/dilate → contour lớn nhất)
   ▼
Resize 128×128, chia train/val/test (80/10/10), one-hot nhãn
   ▼
Transfer learning: backbone ImageNet (đóng băng) → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(2, Softmax)
   │  Adam, categorical cross-entropy, 25 epoch, batch 32
   ▼
Đánh giá: classification report, confusion matrix, ROC/AUC, K-fold cross-validation
   ▼
Lưu model VGG16 (.h5) → Flask web app
```

Việc cắt vùng não dùng OpenCV + `imutils`: tìm contour lớn nhất sau khi làm mờ, phân ngưỡng và khử nhiễu hình thái, rồi cắt theo 4 điểm cực trái/phải/trên/dưới để loại bỏ nền đen thừa.

## 2. Kết quả

### 2.1 Trên tập test (206 ảnh)

Trong notebook, lớp 0 = có u (114 ảnh test), lớp 1 = không u (92 ảnh test). Confusion matrix: hàng là nhãn thật, cột là nhãn dự đoán, theo thứ tự [có u, không u].

| Mô hình | Accuracy | Macro F1 | Confusion matrix |
|:---|:---:|:---:|:---|
| **VGG16** | **0,98** | **0,98** | `[[111, 3], [2, 90]]` |
| ResNet50 | 0,97 | 0,97 | `[[113, 1], [5, 87]]` |
| MobileNet | 0,90 | 0,90 | `[[100, 14], [6, 86]]` |
| CNN tự xây dựng | 0,85 | 0,85 | `[[102, 12], [18, 74]]` |
| InceptionV3 | 0,83 | 0,82 | `[[110, 4], [31, 61]]` |

### 2.2 K-fold cross-validation (5 fold, 10 epoch mỗi fold)

| Mô hình | Accuracy trung bình | Độ lệch chuẩn |
|:---|:---:|:---:|
| ResNet50 | 97,53% | ± 1,20 |
| **VGG16** | 96,71% | ± 0,90 |
| MobileNet | 92,11% | ± 0,59 |
| InceptionV3 | 83,39% | ± 2,22 |
| CNN tự xây dựng | 73,95% | ± 1,08 |

VGG16 được chọn để triển khai vì cho kết quả test cao nhất và ổn định giữa các fold (độ lệch thấp hơn ResNet50), đồng thời số tham số huấn luyện ít hơn (2,1 triệu so với 8,4 triệu).

## 3. Web app

`app.py` là ứng dụng Flask gồm 2 route:

| Route | Chức năng |
|:---|:---|
| `GET /` | Trang tải ảnh (`templates/index.html`) |
| `POST /predict` | Lưu ảnh vào `uploads/`, resize 128×128, chuẩn hóa theo `vgg16.preprocess_input`, dự đoán và trả về `"Có u não"` hoặc `"Không có u não"` |

Phía client (`static/js/script.js`) gửi form bằng `fetch`, hiển thị kết quả và ảnh vừa tải lên mà không cần tải lại trang.

## 4. Cấu trúc thư mục

```
braintumor-classification-/
├── Final_do_an_4_Brain_tumor_classfication.ipynb  # Notebook Kaggle: EDA, augmentation, huấn luyện, so sánh, K-fold
├── app.py                                         # Flask web app
├── weights/vgg16_model.h5                         # Model VGG16 đã huấn luyện (dùng bởi app.py)
├── vgg16_model.h5                                 # Bản sao model
├── templates/index.html                           # Giao diện
├── static/css/style.css, static/js/script.js
├── uploads/                                       # Ảnh MRI mẫu / ảnh người dùng tải lên
├── Brain_Tumor_Classification (3).pdf             # Báo cáo
└── 2024-10-07 23-44-21.mkv                        # Video demo
```

## 5. Chạy thử

```bash
pip install flask tensorflow opencv-python pillow numpy
python app.py
```

Mở `http://127.0.0.1:5000`, chọn một ảnh MRI (có sẵn trong `uploads/`, tên `Y*` là có u, `N*`/`*_no` là không u) và bấm **Tải lên và Dự đoán**.

Notebook được viết cho môi trường Kaggle (đường dẫn `/kaggle/input`, `/kaggle/working`); khi chạy nơi khác cần sửa lại đường dẫn dữ liệu.

## 6. Công nghệ

Python · TensorFlow / Keras · OpenCV · imutils · scikit-learn · NumPy / Pandas · Matplotlib / Seaborn · Flask

> ⚠️ Dự án phục vụ mục đích học tập, không dùng để chẩn đoán y khoa.
