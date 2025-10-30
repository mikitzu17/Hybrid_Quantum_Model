#!/usr/bin/env python
# coding: utf-8

# In[17]:


pip install numpy matplotlib pillow scikit-learn tensorflow


# In[18]:


'''
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
'''


# In[2]:


'''
IMG_SIZE = 128  # or 224
DATA_DIR = r"D:\four 1\FINAL YEAR PROJECT\model cnn\Alzheimer_MRI_4_classes_dataset\Alzheimer_MRI_4_classes_dataset"

categories = os.listdir(DATA_DIR)
data = []
labels = []

for category in categories:
    path = os.path.join(DATA_DIR, category)
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            image = Image.open(img_path).convert('L')  # grayscale
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image = np.array(image) / 255.0  # normalize
            data.append(image)
            labels.append(category)
        except:
            print("Error loading:", img_path)

'''


# In[19]:


''' 
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # add channel dim
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

'''


# In[41]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# In[42]:


val_data.reset()
IMG_SIZE = 128
BATCH_SIZE = 32
DATA_PATH = r"D:\four 1\FINAL YEAR PROJECT\model cnn\Alzheimer_MRI_4_classes_dataset\Alzheimer_MRI_4_classes_dataset"  # e.g., './OASIS_Dataset'
images, labels = next(train_data)
class_names = list(train_data.class_indices.keys())


# In[43]:


datagen = ImageDataGenerator(
    rescale=1./255,         # normalizing the pixel values
    validation_split=0.2    # 20% for validation
)

train_data = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# Get predictions for the **entire validation set**
predictions = model.predict(val_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes  # Actual class labels

# Get class names in correct order
class_names = list(val_data.class_indices.keys())


# In[44]:


print(train_data.class_indices)


# In[45]:


'''
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 classes
])'''

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # Changed from 4 to 3
])



# In[46]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[47]:


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# In[48]:


plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Alzheimer's Classes")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# In[50]:


plt.figure(figsize=(8, 8))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    label_index = np.argmax(labels[i])  # Convert one-hot to index
    plt.title(class_names[label_index])
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[ ]:




