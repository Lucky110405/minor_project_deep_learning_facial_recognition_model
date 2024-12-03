import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

# Enabling mixed precision training to leverage the RTX 3050 Tensor Cores:
set_global_policy('mixed_float16')


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
        fill_mode='nearest')

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='sparse',
        subset='training')

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='sparse',
        subset='validation')

    return train_generator, validation_generator


# Designing the CNN for Facial Recognition:
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Adjust the output layer based on the number of classes

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Training the CNN with the public Dataset:
# data_dir = '/mnt/d/projects/minor_project/lfw_public_dataset' # path to the image dataset in wsl2
data_dir = 'D:/projects/minor_project/lfw_public_dataset' # path to the image dataset in windows

input_shape = (128, 128, 3)
# Dynamically determine the number of classes
num_classes = len(os.listdir(data_dir)) # number of classes i.e the no. of students

train_generator, validation_generator = load_dataset(data_dir)

model = create_cnn_model(input_shape, num_classes)
model.summary() # Seeing the model architecture

# Define early stopping callback to automatically stop training when the model's performance on the validation data stops improving. This can help you avoid overfitting and save time.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define ModelCheckpoint callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    filepath='best_facial_recognition_model.keras',  # Save path for the best model
    monitor='val_accuracy',  # Metric to monitor
    save_best_only=True,  # Save only the model with the best val_accuracy
    verbose=1  # Print message when a new best model is saved
)

# Training the model with early stopping callback and ModelCheckpoint callback:
history = model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[early_stopping, checkpoint])

# Saving the model:
model.save('facial_recognition_model.h5')

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
    plt.savefig('training_validation_plot.png')
    print("Plot saved as training_validation_plot.png")

plot_learning_curves(history)

#Testing and Evaluating the Model on custom dataset:
# data_dir_test = '/mnt/d/projects/minor_project/face_recognition_data_images' # path to the test image dataset in wsl2
data_dir_test = 'D:/projects/minor_project/face_recognition_data_images' # path to the test image dataset in windows

datagen = ImageDataGenerator(rescale=1.0/255, fill_mode='nearest')
test_generator = datagen.flow_from_directory(
    data_dir_test, 
    target_size=(128, 128),
    batch_size=16, 
    class_mode='sparse')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

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

#  [[ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0 14]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  4]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  5]
#  [ 0  0  0  0  0  0  0  0  0  0  0]]

# Classification Report:

#                precision    recall  f1-score   support

#            0       0.00      0.00      0.00       5.0
#            1       0.00      0.00      0.00       5.0
#            2       0.00      0.00      0.00      14.0
#            3       0.00      0.00      0.00       5.0
#            4       0.00      0.00      0.00       5.0
#            5       0.00      0.00      0.00       5.0
#            6       0.00      0.00      0.00       4.0
#            7       0.00      0.00      0.00       5.0
#            8       0.00      0.00      0.00       5.0
#            9       0.00      0.00      0.00       5.0
#         1871       0.00      0.00      0.00       0.0

#     accuracy                           0.00      58.0
#    macro avg       0.00      0.00      0.00      58.0
# weighted avg       0.00      0.00      0.00      58.0