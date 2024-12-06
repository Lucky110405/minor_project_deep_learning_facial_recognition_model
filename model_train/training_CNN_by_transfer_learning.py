# importing neccesary libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
# Setting up the GPU memory growth:
mixed_precision.set_global_policy('mixed_float16')
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Loading the Training Dataset and preprocessing it:
def load_dataset(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),  # Increased image size. # resize all images to 224x224 pixels
        batch_size=16,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),  
        batch_size=16,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )
    return train_generator, validation_generator

# Designing the CNN for Facial Recognition using Transfer Learning:
def create_transfer_learning_model(input_shape, num_classes):
    # Use a recent model as the base model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
   
    # Fine-tune the last few layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        # layers.Dense(256, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
        
    ])

    # # Add learning rate scheduling
    # initial_learning_rate = 0.001
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=1000,
    #     decay_rate=0.9,
    #     staircase=True
    # )

    # # Better optimizer configuration
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compile with better metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Training the CNN with the Custom Dataset:
data_dir = '/mnt/d/projects/minor_project/data_set/lfw_public_dataset'  # path to the image dataset in wsl2
#data_dir = 'D:/projects/minor_project/data_set/face_recognition_data_images'  # path to the image dataset in windows

input_shape = (224, 224, 3) # (height × width × channels)
num_classes = len(os.listdir(data_dir))  # number of classes i.e the no. of students

train_generator, validation_generator = load_dataset(data_dir)

model = create_transfer_learning_model(input_shape, num_classes)
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Add more callbacks
callbacks = [
    early_stopping,
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_facial_recognition_model_transfer_learning.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train the model with early stopping and capture the history
history = model.fit(train_generator, epochs=100, validation_data=validation_generator, callbacks=callbacks)

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
data_dir_test = '/mnt/d/projects/minor_project/data_set/face_recognition_data_images'  # path to the test image dataset in wsl2
# data_dir_test = 'D:/projects/minor_project/data_set/face_recognition_data_images'  # path to the test image dataset in windows

datagen = ImageDataGenerator(rescale=1.0/255, fill_mode='nearest')
test_generator = datagen.flow_from_directory(data_dir_test, target_size=(224, 224), batch_size=16, class_mode='sparse')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Output of Predicting on test data:
# Test Accuracy: 0.6896551847457886

# Predicting on test data
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

# # Cross-Validation
# def cross_validate_model(model, data_dir, n_splits=5):
#     datagen = ImageDataGenerator(rescale=1.0/255)
#     scores = []
    
#     # Get list of all image files and labels
#     all_images = []
#     all_labels = []
#     for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
#         class_path = os.path.join(data_dir, class_name)
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 if os.path.isfile(img_path):
#                     all_images.append(img_path)
#                     all_labels.append(class_idx)
    
#     all_images = np.array(all_images)
#     all_labels = np.array(all_labels)
    
#     # Perform k-fold cross validation
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
#     for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
#         print(f"\nFold {fold + 1}/{n_splits}")
        
#         # Split data for this fold
#         train_paths = all_images[train_idx]
#         train_labels = all_labels[train_idx]
#         val_paths = all_images[val_idx]
#         val_labels = all_labels[val_idx]
        
#         # Create temporary directories for this fold
#         temp_train_dir = os.path.join('temp', f'fold_{fold}', 'train')
#         temp_val_dir = os.path.join('temp', f'fold_{fold}', 'val')
#         os.makedirs(temp_train_dir, exist_ok=True)
#         os.makedirs(temp_val_dir, exist_ok=True)
        
#         # Create symbolic links for training and validation data
#         for path, label in zip(train_paths, train_labels):
#             class_dir = os.path.join(temp_train_dir, str(label))
#             os.makedirs(class_dir, exist_ok=True)
#             os.symlink(path, os.path.join(class_dir, os.path.basename(path)))
            
#         for path, label in zip(val_paths, val_labels):
#             class_dir = os.path.join(temp_val_dir, str(label))
#             os.makedirs(class_dir, exist_ok=True)
#             os.symlink(path, os.path.join(class_dir, os.path.basename(path)))
        
#         # Create generators for this fold
#         train_generator = datagen.flow_from_directory(
#             temp_train_dir,
#             target_size=(224, 224),
#             batch_size=16,
#             class_mode='sparse'
#         )
        
#         val_generator = datagen.flow_from_directory(
#             temp_val_dir,
#             target_size=(224, 224),
#             batch_size=16,
#             class_mode='sparse'
#         )
        
#         # Train and evaluate
#         history = model.fit(
#             train_generator,
#             validation_data=val_generator,
#             epochs=50,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         # Get the best validation accuracy
#         best_val_acc = max(history.history['val_accuracy'])
#         scores.append(best_val_acc)
        
#         # Cleanup temporary directories
#         import shutil
#         shutil.rmtree('temp')
        
#     return np.mean(scores), np.std(scores)


# # Model evaluation with cross-validation:
# X = np.concatenate([x for x, y in train_generator], axis=0)
# y = np.concatenate([y for x, y in train_generator], axis=0)

# mean_score, std_score = cross_validate_model(model, (X, y))
# print(f"Cross-validation score: {mean_score:.4f} (±{std_score:.4f})")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Adding model performance visualization plots:
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# After computing confusion matrix
class_names = list(train_generator.class_indices.keys())
plot_confusion_matrix(cm, class_names)

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
