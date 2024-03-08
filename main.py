from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Function to load and resize images
def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        # Resizing the images to (150x150)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img.flatten())
                        labels.append(subfolder)
                        print('Labels \n', labels)
                    else:
                        print(f"Warning: Unable to load {filename}")

    return images, labels


# Loading the testing images
def load_test_images_from_folder(folder, target_shape=None):
    test_images = []
    test_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        # Resizing the images to (150x150)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        test_images.append(img.flatten())
                        test_labels.append(subfolder)
                        print('Labels \n', labels)
                    else:
                        print(f"Warning: Unable to load {filename}")

    return test_images, test_labels


# Loading the validation images
def load_validation_images_from_folder(folder, target_shape=None):
    validation_images = []
    validation_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        # Resizing the images to (150x150)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        validation_images.append(img.flatten())
                        validation_labels.append(subfolder)
                        print('Labels \n', labels)
                    else:
                        print(f"Warning: Unable to load {filename}")

    return validation_images, validation_labels


# Load the data from the folders
data_folder = './Set_date'  # Folder with training data
test_folder = './Set_test'  # Folder with testing data
validation_folder = './Set_validare'  # Folder with validation data


images, labels = load_train_images_from_folder(data_folder, target_shape=(150, 150))
test_images, test_labels = load_test_images_from_folder(test_folder, target_shape=(150, 150))
validation_images, validation_labels = load_validation_images_from_folder(validation_folder, target_shape=(150, 150))


# Build and train the SVM model
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
model.fit(images, labels)

#  Cross Validation
folds = 5
model_accuracy = cross_val_score(model, images, labels, cv=folds, scoring='accuracy')
print(f'Average accuracy: {model_accuracy.mean() * 100:.2f}%')


# Make predictions on the validation set
predictions = model.predict(validation_images)

# Accuracy score for validation
score = accuracy_score(predictions, validation_labels)
print('{}% of validation samples were correctly classified'.format(str(score * 100)))


# Make predictions on the test set
predictions2 = model.predict(test_images)

# Accuracy score for testing
score = accuracy_score(predictions2, test_labels)
print('{}% of test samples were correctly classified'.format(str(score * 100)))


def plot_images(images, true_labels, predicted_labels):
    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i].reshape(150, 150, 3))
        title = f'True: {true_labels[i]}\nPredicted: {predicted_labels[i]}'
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Plot the test images with their true and predicted labels
plot_images(test_images, test_labels, predictions2)
plot_images(validation_images, validation_labels, predictions)

# Confusion matrix for test set
conf_matrix_test = confusion_matrix(test_labels, predictions2)

# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=np.unique(test_labels),
            yticklabels=np.unique(test_labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')
plt.show()
