# importing neccesary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Loading the Training Dataset and preprocessing it:
def load_dataset(data_dir):
    datagen = ImageDataGenerator(
        rescale=1.0/255, 
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),  # Resize all images to 128x128 pixels
        batch_size=16,
        class_mode='sparse',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),  # Resize all images to 128x128 pixels
        batch_size=16,
        class_mode='sparse',
        subset='validation'
    )
    return train_generator, validation_generator

# Designing the CNN for Facial Recognition using Transfer Learning:
def create_transfer_learning_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Training the CNN with the Custom Dataset:
# data_dir = '/mnt/d/projects/minor_project/data_set/face_recognition_data_images'  # path to the image dataset in wsl2
data_dir = 'D:/projects/minor_project/data_set/face_recognition_data_images'  # path to the image dataset in windows

input_shape = (128, 128, 3)
num_classes = len(os.listdir(data_dir))  # number of classes i.e the no. of students

train_generator, validation_generator = load_dataset(data_dir)

model = create_transfer_learning_model(input_shape, num_classes)
model.summary()

# Output of the model summary:
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ vgg16 (Functional)                   │ (None, 4, 4, 512)           │      14,714,688 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 8192)                │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 256)                 │       2,097,408 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 256)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 10)                  │           2,570 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘


# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping and capture the history
history = model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[early_stopping])

# Saving the model
model.save('models/facial_recognition_model.keras')

# Plotting learning curves
def plot_learning_curves(history):
    epochs = range(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(14, 5))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot to a file
    plt.savefig('training_validation_metrics.png')
    print("Plot saved as training_validation_metrics.png")

plot_learning_curves(history)

# Testing and Evaluating the Model with Custom Dataset:
# data_dir_test = '/mnt/d/projects/minor_project/data_set/face_recognition_data_images'  # path to the test image dataset in wsl2
data_dir_test = 'D:/projects/minor_project/data_set/face_recognition_data_images'  # path to the test image dataset in windows

datagen = ImageDataGenerator(rescale=1.0/255, fill_mode='nearest')
test_generator = datagen.flow_from_directory(data_dir_test, target_size=(128, 128), batch_size=16, class_mode='sparse')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Output of Predicting on test data:
# Test Accuracy: 0.6896551847457886

# Predicting on test data
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_true, y_pred))



# output for the above confsion matrix and classification report:

# Confusion Matrix:

#  [[1 0 2 1 0 0 0 0 0 1]
#  [0 2 1 1 0 0 0 0 0 1]
#  [0 3 3 2 0 0 1 2 1 2]
#  [1 1 0 1 0 0 0 1 1 0]
#  [2 0 2 1 0 0 0 0 0 0]
#  [0 1 0 2 0 0 0 1 1 0]
#  [0 1 2 0 0 0 0 0 0 1]
#  [2 1 1 0 0 0 0 1 0 0]
#  [1 2 1 1 0 0 0 0 0 0]
#  [0 0 3 1 0 0 0 0 0 1]]

# Classification Report:

#                precision    recall  f1-score   support

#            0       0.14      0.20      0.17         5
#            1       0.18      0.40      0.25         5
#            2       0.20      0.21      0.21        14
#            3       0.10      0.20      0.13         5
#            4       0.00      0.00      0.00         5
#            5       0.00      0.00      0.00         5
#            6       0.00      0.00      0.00         4
#            7       0.20      0.20      0.20         5
#            8       0.00      0.00      0.00         5
#            9       0.17      0.20      0.18         5

#     accuracy                           0.16        58
#    macro avg       0.10      0.14      0.11        58
# weighted avg       0.12      0.16      0.13        58
