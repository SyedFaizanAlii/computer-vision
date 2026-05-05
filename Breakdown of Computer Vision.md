# 👁️ Computer Vision — A Complete Learning Guide

> **Master the fundamentals and advanced techniques of Computer Vision using Python, OpenCV, and Deep Learning.**  
> This repository is a structured, in-depth resource for students and practitioners who want to understand how machines see and interpret the visual world.

---

## 📌 Table of Contents

1. [What is Computer Vision?](#1-what-is-computer-vision)
2. [How Machines See Images](#2-how-machines-see-images)
3. [Core Concepts & Terminology](#3-core-concepts--terminology)
4. [Computer Vision Pipeline](#4-computer-vision-pipeline)
5. [Image Processing Fundamentals](#5-image-processing-fundamentals)
   - [Reading & Displaying Images](#51-reading--displaying-images)
   - [Color Spaces](#52-color-spaces)
   - [Image Transformations](#53-image-transformations)
   - [Filtering & Edge Detection](#54-filtering--edge-detection)
6. [Feature Extraction](#6-feature-extraction)
7. [Classical CV Algorithms](#7-classical-cv-algorithms)
8. [Deep Learning for Computer Vision](#8-deep-learning-for-computer-vision)
   - [Convolutional Neural Networks (CNN)](#81-convolutional-neural-networks-cnn)
   - [Transfer Learning](#82-transfer-learning)
   - [Popular Architectures](#83-popular-architectures)
9. [Key CV Tasks](#9-key-cv-tasks)
   - [Image Classification](#91-image-classification)
   - [Object Detection](#92-object-detection)
   - [Image Segmentation](#93-image-segmentation)
   - [Face Detection & Recognition](#94-face-detection--recognition)
10. [Datasets & Benchmarks](#10-datasets--benchmarks)
11. [Full Code Examples](#11-full-code-examples)
12. [Tools & Libraries](#12-tools--libraries)
13. [Tips & Best Practices](#13-tips--best-practices)
14. [Common Mistakes to Avoid](#14-common-mistakes-to-avoid)
15. [Resources & References](#15-resources--references)

---

## 1. What is Computer Vision?

**Computer Vision (CV)** is a field of Artificial Intelligence that enables computers to **interpret, understand, and process visual information** from the world — images, videos, and real-time camera feeds.

Just as humans use their eyes and brain to understand what they see, computers use **cameras** (to capture input) and **algorithms / neural networks** (to process and understand) to perform similar tasks.

### Real-World Applications:
| Domain | Application |
|---|---|
| 🏥 Healthcare | Medical image analysis, tumor detection, X-ray interpretation |
| 🚗 Autonomous Vehicles | Lane detection, pedestrian detection, obstacle avoidance |
| 🛒 Retail | Cashier-less stores (Amazon Go), shelf monitoring |
| 🔒 Security | Facial recognition, surveillance, intrusion detection |
| 🌾 Agriculture | Crop disease detection, yield estimation |
| 📱 Social Media | Face filters, auto-tagging, content moderation |
| 🏭 Manufacturing | Defect detection, quality control |
| 📦 Logistics | Barcode scanning, package sorting |

---

## 2. How Machines See Images

To a human, an image is a scene. To a computer, an image is simply a **matrix of numbers**.

### Grayscale Image
A 2D matrix where each cell holds a pixel value between **0 (black)** and **255 (white)**.

```
Image (4×4 grayscale):
[[ 0,   50,  100, 150],
 [200,  255,  30,  80],
 [120,   45, 210, 190],
 [ 10,   90, 170, 240]]
```

### Color Image (RGB)
A **3D matrix** with shape `(Height, Width, 3)` — three channels: **Red, Green, Blue**.

```
Shape: (480, 640, 3)
 → 480 rows (height)
 → 640 columns (width)
 → 3 channels (R, G, B)
 → Total pixels: 307,200
 → Total values: 921,600
```

Each pixel is a combination of R, G, B values (0–255):
- Pure Red → (255, 0, 0)
- Pure Green → (0, 255, 0)
- White → (255, 255, 255)
- Black → (0, 0, 0)

---

## 3. Core Concepts & Terminology

| Term | Definition |
|---|---|
| **Pixel** | The smallest unit of an image — a single colored dot |
| **Resolution** | Number of pixels in an image (e.g., 1920×1080) |
| **Channel** | A single color component (R, G, or B in a color image) |
| **Kernel / Filter** | A small matrix used to apply effects (blur, edge detection) |
| **Convolution** | Sliding a kernel over an image to produce a feature map |
| **Feature Map** | Output of applying a filter to an image |
| **Bounding Box** | Rectangle defined by (x, y, width, height) to locate objects |
| **IoU** | Intersection over Union — measures overlap between two boxes |
| **Anchor Box** | Predefined reference boxes used in object detection |
| **Stride** | Step size when sliding a filter across an image |
| **Padding** | Adding zeros around image borders to preserve spatial dimensions |
| **Pooling** | Downsampling a feature map (MaxPool, AvgPool) |
| **Segmentation** | Classifying every pixel in an image |
| **Keypoint** | Specific point of interest in an image (e.g., eye corner, joint) |

---

## 4. Computer Vision Pipeline

```
📷 INPUT
   │
   ▼
🖼️  IMAGE ACQUISITION
   (Camera / File / Video Stream)
   │
   ▼
🔧  PREPROCESSING
   (Resize, Normalize, Denoise, Enhance)
   │
   ▼
🔍  FEATURE EXTRACTION
   (Edges, Corners, Gradients / CNN Features)
   │
   ▼
🧠  MODEL / ALGORITHM
   (Classical CV or Deep Learning)
   │
   ▼
📊  POST-PROCESSING
   (NMS, Thresholding, Drawing Boxes)
   │
   ▼
✅  OUTPUT
   (Classification Label / Detected Objects / Segmentation Mask)
```

---

## 5. Image Processing Fundamentals

### 5.1 Reading & Displaying Images

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
img = cv2.imread('image.jpg')          # BGR format by default
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Display with matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Image properties
print("Shape:", img.shape)         # (height, width, channels)
print("Data type:", img.dtype)     # uint8
print("Max pixel:", img.max())     # 255
print("Min pixel:", img.min())     # 0

# Resize
img_resized = cv2.resize(img, (224, 224))

# Save
cv2.imwrite('output.jpg', img_resized)
```

---

### 5.2 Color Spaces

Color spaces are different ways to represent color information. Choosing the right one matters for many CV tasks.

```python
# BGR → RGB (for matplotlib display)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR → Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR → HSV (useful for color-based segmentation)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR → LAB (perceptually uniform color space)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Visualize all channels
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Original (RGB)', 'Grayscale', 'HSV', 'LAB']
images = [img_rgb, img_gray, img_hsv, img_lab]

for ax, im, title in zip(axes, images, titles):
    ax.imshow(im if len(im.shape) == 3 else im, cmap='gray' if len(im.shape) == 2 else None)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**When to use which color space:**
| Color Space | Best For |
|---|---|
| **BGR/RGB** | General processing, display |
| **Grayscale** | Edge detection, shape analysis, reducing compute |
| **HSV** | Color-based object segmentation (e.g., detect red objects) |
| **LAB** | Color correction, perceptual uniformity |
| **YCrCb** | Skin detection, JPEG compression |

---

### 5.3 Image Transformations

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# ── Rotation ──────────────────────────────────────────
center = (w // 2, h // 2)
M_rot = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
img_rotated = cv2.warpAffine(img, M_rot, (w, h))

# ── Translation (shift) ───────────────────────────────
tx, ty = 50, 30  # shift 50px right, 30px down
M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
img_translated = cv2.warpAffine(img, M_trans, (w, h))

# ── Flip ──────────────────────────────────────────────
img_flip_h = cv2.flip(img, 1)   # Horizontal flip
img_flip_v = cv2.flip(img, 0)   # Vertical flip

# ── Crop ──────────────────────────────────────────────
img_crop = img[100:300, 150:400]  # [y1:y2, x1:x2]

# ── Perspective Transform ─────────────────────────────
pts1 = np.float32([[50,50],[200,50],[50,200],[200,200]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M_persp = cv2.getPerspectiveTransform(pts1, pts2)
img_warped = cv2.warpPerspective(img, M_persp, (300, 300))

# ── Normalization ─────────────────────────────────────
img_normalized = img.astype(np.float32) / 255.0  # Scale to [0, 1]
```

---

### 5.4 Filtering & Edge Detection

Filters (kernels) are used to enhance or extract features from images.

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── Blurring / Noise Reduction ────────────────────────
img_blur     = cv2.blur(img, (5, 5))                    # Average blur
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)         # Gaussian blur
img_median   = cv2.medianBlur(img, 5)                    # Median blur (good for salt-and-pepper noise)
img_bilateral= cv2.bilateralFilter(img, 9, 75, 75)      # Edge-preserving blur

# ── Edge Detection ────────────────────────────────────
# Sobel (gradient-based)
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# Canny (best general-purpose edge detector)
edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

# Laplacian
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

# ── Morphological Operations ──────────────────────────
kernel = np.ones((5, 5), np.uint8)
img_dilated  = cv2.dilate(edges, kernel, iterations=1)   # Expand white regions
img_eroded   = cv2.erode(edges, kernel, iterations=1)    # Shrink white regions
img_opened   = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)   # Erode then dilate
img_closed   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # Dilate then erode
```

---

## 6. Feature Extraction

Before deep learning, CV relied on **handcrafted features** — algorithms that describe keypoints and local regions of an image.

### SIFT — Scale-Invariant Feature Transform
Detects keypoints and computes descriptors that are invariant to scale and rotation.

```python
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

img_kp = cv2.drawKeypoints(img, keypoints, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(f"Keypoints found: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")  # (N, 128)
```

### ORB — Oriented FAST and Rotated BRIEF
Fast alternative to SIFT, suitable for real-time applications.

```python
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
```

### HOG — Histogram of Oriented Gradients
Used for pedestrian detection and object detection.

```python
from skimage.feature import hog
from skimage import exposure

features, hog_image = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)
```

---

## 7. Classical CV Algorithms

### Contour Detection
```python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

# Analyze each contour
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_contours, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

### Template Matching
```python
template = cv2.imread('template.jpg', 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw rectangle around match
th, tw = template.shape
top_left = max_loc
bottom_right = (top_left[0] + tw, top_left[1] + th)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
```

### Color-Based Object Segmentation (HSV)
```python
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for red color
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(img_hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask=mask)
```

---

## 8. Deep Learning for Computer Vision

### 8.1 Convolutional Neural Networks (CNN)

CNNs are the backbone of modern computer vision. They automatically learn visual features (edges → shapes → objects) through training.

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────┐
│  Conv Layer 1   │  → Learn low-level features (edges, gradients)
│  + ReLU         │
│  + MaxPooling   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Conv Layer 2   │  → Learn mid-level features (corners, textures)
│  + ReLU         │
│  + MaxPooling   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Conv Layer N   │  → Learn high-level features (eyes, wheels, faces)
│  + ReLU         │
└────────┬────────┘
         │
┌────────▼────────┐
│  Flatten        │
│  Fully Connected│  → Combine features into predictions
│  Softmax        │  → Output class probabilities
└─────────────────┘
```

**Building a CNN with Keras:**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

---

### 8.2 Transfer Learning

Instead of training from scratch, **transfer learning** reuses a model pre-trained on a massive dataset (like ImageNet with 1.2M images) and fine-tunes it for your specific task.

```
Pre-trained Model (ResNet50, VGG16, EfficientNet...)
trained on ImageNet (1000 classes)
        │
        ▼ Freeze base layers
┌───────────────────┐
│   Feature Extractor│  ← Keep these weights (already learned great features)
└────────┬──────────┘
         │ Unfreeze top layers (optional)
┌────────▼──────────┐
│   Custom Head     │  ← Replace with YOUR classification layers
│   Dense → Softmax │
└───────────────────┘
        │
        ▼
   Your Task (Dogs vs Cats, Medical Imaging, etc.)
```

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers

# Load pre-trained ResNet50 (without top classifier)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')  # Binary classification
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning: unfreeze top layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all except last 20 layers
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

### 8.3 Popular Architectures

| Architecture | Year | Key Innovation | Best For |
|---|---|---|---|
| **LeNet-5** | 1998 | First practical CNN | Digit recognition |
| **AlexNet** | 2012 | Deep CNN, ReLU, Dropout | ImageNet breakthrough |
| **VGG16/19** | 2014 | Very deep, simple 3×3 convs | Feature extraction |
| **GoogLeNet** | 2014 | Inception modules | Efficient computation |
| **ResNet** | 2015 | Residual/skip connections | Deep networks (50, 101, 152 layers) |
| **MobileNet** | 2017 | Depthwise separable convs | Mobile/edge devices |
| **EfficientNet** | 2019 | Compound scaling | SOTA accuracy/efficiency |
| **Vision Transformer (ViT)** | 2020 | Attention mechanism | Large-scale classification |
| **YOLO series** | 2016+ | Real-time detection | Object detection |

---

## 9. Key CV Tasks

### 9.1 Image Classification

Assign a single label to an entire image.

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = EfficientNetB0(weights='imagenet')

# Preprocess image
img = image.load_img('cat.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
preds = model.predict(x)
results = decode_predictions(preds, top=3)[0]

for class_id, class_name, confidence in results:
    print(f"{class_name}: {confidence:.2%}")
```

---

### 9.2 Object Detection

Detect and locate multiple objects in an image using bounding boxes.

```python
# Using YOLOv8 (ultralytics)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano model (fastest)

# Detect objects in image
results = model('image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = model.names[cls]
        print(f"{label}: {conf:.2%} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# Visualize
result_img = results[0].plot()
```

**Popular Object Detection Algorithms:**

| Algorithm | Speed | Accuracy | Use Case |
|---|---|---|---|
| **YOLO (v5/v8)** | Very Fast | High | Real-time detection |
| **SSD** | Fast | Medium | Mobile applications |
| **Faster R-CNN** | Slow | Very High | High-accuracy tasks |
| **RetinaNet** | Medium | High | Dense object detection |
| **DETR** | Medium | High | Transformer-based detection |

---

### 9.3 Image Segmentation

Classify every pixel in an image.

```
Semantic Segmentation    → All cars = same label (car)
Instance Segmentation    → Each car = unique label (car_1, car_2)
Panoptic Segmentation    → Both semantic + instance combined
```

```python
# Using torchvision DeepLabV3
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

model = deeplabv3_resnet50(pretrained=True).eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('street.jpg')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)['out'][0]

seg_mask = output.argmax(0).numpy()
print("Segmentation mask shape:", seg_mask.shape)
print("Unique classes found:", set(seg_mask.flatten()))
```

---

### 9.4 Face Detection & Recognition

```python
import cv2

# ── Face Detection with Haar Cascade ──────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

img = cv2.imread('people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Faces detected: {len(faces)}")
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]  # Face region

# ── Using Deep Learning (MTCNN) ───────────────────────
from mtcnn import MTCNN
import cv2

detector = MTCNN()
img_rgb = cv2.cvtColor(cv2.imread('people.jpg'), cv2.COLOR_BGR2RGB)
results = detector.detect_faces(img_rgb)

for face in results:
    x, y, w, h = face['box']
    confidence = face['confidence']
    keypoints = face['keypoints']  # eyes, nose, mouth, ears
    print(f"Face confidence: {confidence:.2%}")
    print(f"Keypoints: {keypoints}")
```

---

## 10. Datasets & Benchmarks

| Dataset | Size | Task | Description |
|---|---|---|---|
| **ImageNet** | 1.2M images, 1000 classes | Classification | Gold standard for classification |
| **COCO** | 330K images, 80 classes | Detection/Segmentation | Complex scenes, multiple objects |
| **CIFAR-10/100** | 60K images | Classification | Small 32×32 images, beginner-friendly |
| **Pascal VOC** | 11K images, 20 classes | Detection | Classic benchmark |
| **Open Images** | 9M images | Detection/Segmentation | Google's large-scale dataset |
| **LFW** | 13K images | Face Recognition | Labeled Faces in the Wild |
| **CelebA** | 200K face images | Face Attributes | Celebrity face attributes |
| **MNIST** | 70K images | Classification | Handwritten digits, first CV project |
| **Oxford-IIIT Pets** | 7K images | Classification/Segmentation | 37 pet categories |

---

## 11. Full Code Examples

### Complete Image Classification Pipeline

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ── 1. Data Loading & Augmentation ────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print(f"Classes: {train_gen.class_indices}")

# ── 2. Build Transfer Learning Model ──────────────────
base_model = MobileNetV2(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# ── 3. Compile & Train ────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# ── 4. Fine-tuning ─────────────────────────────────────
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# ── 5. Plot Training History ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
```

---

## 12. Tools & Libraries

| Library | Purpose | Install |
|---|---|---|
| **OpenCV** | Image processing, classical CV | `pip install opencv-python` |
| **TensorFlow/Keras** | Deep learning, CNNs | `pip install tensorflow` |
| **PyTorch** | Deep learning, research | `pip install torch torchvision` |
| **Ultralytics YOLO** | Real-time object detection | `pip install ultralytics` |
| **scikit-image** | Image analysis algorithms | `pip install scikit-image` |
| **Pillow (PIL)** | Image I/O and basic ops | `pip install Pillow` |
| **Albumentations** | Fast image augmentation | `pip install albumentations` |
| **MTCNN** | Face detection | `pip install mtcnn` |
| **Detectron2** | Facebook's detection framework | See official docs |
| **Mediapipe** | Hand/pose/face landmark detection | `pip install mediapipe` |

---

## 13. Tips & Best Practices

### ✅ Do's

- **Normalize pixel values** — always scale to [0, 1] or standardize for deep learning
- **Use data augmentation** — rotation, flip, zoom, brightness to prevent overfitting
- **Start with Transfer Learning** — almost always better than training from scratch
- **Visualize your data** — look at samples before training; catch labeling errors early
- **Monitor with validation curves** — detect overfitting before it gets bad
- **Use GPU** — CV tasks are highly parallelizable; use CUDA with PyTorch/TensorFlow
- **Check class balance** — imbalanced datasets hurt model performance
- **Use appropriate image size** — larger isn't always better; balance quality vs compute

### 📐 Image Preprocessing Checklist

```
□ Resize all images to consistent dimensions
□ Normalize pixel values (÷ 255 or standardize)
□ Handle missing/corrupted images
□ Apply augmentation to training set only
□ Convert color space if needed (BGR→RGB)
□ Ensure consistent channel ordering
```

---

## 14. Common Mistakes to Avoid

| ❌ Mistake | ✅ Correct Approach |
|---|---|
| Forgetting BGR→RGB conversion in OpenCV | Always convert before matplotlib display |
| Applying augmentation to validation/test set | Augment training data ONLY |
| Not normalizing pixel values | Divide by 255.0 or use model's preprocess function |
| Training CNN from scratch with little data | Use Transfer Learning |
| Using wrong color space for segmentation | Use HSV for color-based, not BGR |
| Evaluating on test set during tuning | Use validation set; test set is final only |
| Ignoring class imbalance | Use class weights or oversampling |
| Not shuffling training data | Always shuffle to prevent ordering bias |
| Forgetting to set model to eval mode | Use `model.eval()` in PyTorch during inference |

---

## 15. Resources & References

### 📚 Books
- *Deep Learning for Vision Systems* — Mohamed Elgendy
- *Programming Computer Vision with Python* — Jan Erik Solem
- *Computer Vision: Algorithms and Applications* — Richard Szeliski *(free PDF available)*
- *Deep Learning* — Goodfellow, Bengio, Courville *(deeplearningbook.org)*

### 🌐 Online Courses
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) — Stanford (FREE)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) — FREE
- [OpenCV Bootcamp](https://opencv.org/university/) — OpenCV official

### 🛠️ Documentation
- [OpenCV Docs](https://docs.opencv.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/images)
- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Albumentations](https://albumentations.ai/docs/)

### 📄 Landmark Papers
- LeCun et al. (1998) — *Gradient-Based Learning Applied to Document Recognition* (LeNet)
- Krizhevsky et al. (2012) — *ImageNet Classification with Deep CNNs* (AlexNet)
- He et al. (2015) — *Deep Residual Learning for Image Recognition* (ResNet)
- Redmon et al. (2016) — *You Only Look Once: Unified, Real-Time Object Detection* (YOLO)
- Dosovitskiy et al. (2020) — *An Image is Worth 16×16 Words* (Vision Transformer)

---

## 👨‍💻 Author

**Syed Faizan Ali**  
Computer Science Student | Data Science & Computer Vision Enthusiast  
📍 Rahim Yar Khan, Punjab, Pakistan

[![GitHub](https://img.shields.io/badge/GitHub-SyedFaizanAlii-black?logo=github)](https://github.com/SyedFaizanAlii)
[![Kaggle](https://img.shields.io/badge/Kaggle-syedfaizanalii-blue?logo=kaggle)](https://www.kaggle.com/syedfaizanalii)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://in/syed-faizan-ali-70b164306)

---

> ⭐ *If this repository helped you learn Computer Vision, please give it a star!*  
> 🍴 *Feel free to fork, contribute, and share with fellow learners.*

---

*Last Updated: 2026 | License: MIT*
