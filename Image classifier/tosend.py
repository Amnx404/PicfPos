# %%
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import random_split

# Check if MPS (Apple's Metal Performance Shaders) or CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your dataset
data_dir = "./frames/dataset/"

import imgaug as ia
import imgaug.augmenters as iaa
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# Define your imgaug augmentation pipeline
seq = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.CoarseDropout(p=0.1, size_percent=0.05)),  # Random shapes
    iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),  # Blur
    iaa.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),  # Small shifts
    # Add more augmentations as needed
])

# Custom transformation function using imgaug
def imgaug_transform(image):
    image_np = np.array(image)
    image_aug = seq.augment_image(image_np)
    return Image.fromarray(image_aug)

# Integrate with torchvision transforms
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.Lambda(imgaug_transform),  # Apply imgaug augmentations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset with custom_transform
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=custom_transform)




# view class distribution
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# get count of each class
label_counts = Counter(dataset.targets)
total_count = len(dataset.targets)
print(label_counts)
print(total_count)

#PLOT
#find the class names
class_names = dataset.classes
labels = class_names
counts = label_counts.values()
plt.figure(figsize=(10, 5))
plt.bar(labels, counts)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
#make it a bit mor wide
plt.xticks(rotation=90)
plt.show()



# %%
# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
batch_size_m = 500
# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_m, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_m, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_m, shuffle=False)

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to unnormalize and convert tensor to numpy
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of training data
dataiter = iter(train_loader)
images, labels = dataiter.next()

# Show images
imshow(torchvision.utils.make_grid(images[0]))


# %%
# Load a pre-trained model (ResNet18 in this case) and modify it
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))  # Adjusting for the number of classes
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%

# Helper function to calculate metrics
def calculate_metrics(loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Training function with metrics evaluation
def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate training metrics
        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_loader)
        # Calculate validation metrics
        val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.2f}, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}, F1: {train_f1:.2f}, "
              f"Val Acc: {val_accuracy:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, F1: {val_f1:.2f}")

    print("Finished Training")



# %%
# Train the model
train_model(num_epochs=2)
# Optionally, you can save the trained model
torch.save(model.state_dict(), 'model_aaa_2.pth')
# Train the model
train_model(num_epochs=2)
# Optionally, you can save the trained model
torch.save(model.state_dict(), 'model_aaa_4.pth')
# Train the model
train_model(num_epochs=2)
# Optionally, you can save the trained model
torch.save(model.state_dict(), 'model_aaa_6.pth')



# %%
# Optionally, you can save the trained model
torch.save(model.state_dict(), 'model.pth')

# %%
from PIL import Image
import torchvision.transforms as transforms

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function for inference
def predict_image(model, image_path, class_names):
    image = preprocess_image(image_path)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted[0].item()]

    return predicted_class

# Example usage
# image_path = '/Users/amnx/Code/PICfPOS/frame_0304 copy.jpeg'
# model = torch.load('model.pth')
  # Make sure this is your trained model
class_names = dataset.classes  # The class names from your dataset

# predicted_class = predict_image(model, image_path, class_names)
# print(f"Predicted class: {predicted_class}")


# %%
#load model saved as torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('./model_aaa_4.pth'))


# %%
#start reading a video IMG_0387.MOV
import cv2
import os

# Path to your video file
video_path = r"C:\Users\aman.sa\Documents\code\PicfPos\Image classifier\input folder\IMG_0387.MOV"

# Directory to save the frames
if not os.path.exists('demo'):
    os.makedirs('demo')
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Function to preprocess image
def preprocess_image(frame):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Define your preprocessing steps here
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Rest of your code remains the same

# Start capturing the feed
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))
frame_count = 0
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    #preprocess the frame and predict the class
    image = preprocess_image(frame)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted[0].item()]
    #save the video with annotation of the class predicted
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    out.write(frame)
    
    

cap.release()
out.release()
cv2.destroyAllWindows()



# %%



