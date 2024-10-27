# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset
# import pandas as pd
# from featurizer_model import FeaturizerModel
# import os
# from PIL import Image
#
# class CustomObjectDataset(Dataset):
#     def __init__(self, data_file, image_folder, transform=None):
#         self.data = pd.read_csv(data_file, sep='\t')
#         self.image_folder = image_folder
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         image_signature = row['image_signature']
#         label = row['label']
#
#         # Load image
#         image_path = os.path.join(self.image_folder, f"{image_signature}.jpg")
#         image = Image.open(image_path).convert("RGB")
#
#         # Calculate bounding box
#         img_width, img_height = image.size
#         left = int(row['bounding_x'] * img_width)
#         top = int(row['bounding_y'] * img_height)
#         right = int((row['bounding_x'] + row['bounding_width']) * img_width)
#         bottom = int((row['bounding_y'] + row['bounding_height']) * img_height)
#
#         # Crop image to bounding box
#         image = image.crop((left, top, right, bottom))
#
#         # Apply transformations
#         if self.transform:
#             image = self.transform(image)
#
#         return image, label
#
# # Initialize the model, loss function, and optimizer
# model = FeaturizerModel()
# criterion = nn.MSELoss()  # For reconstruction loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Enable training mode
# model.train()
#
# # Load data (e.g., CIFAR10 as an example)
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])
#
# dataset = CustomObjectDataset(data_file='./datasets/raw_train.tsv', image_folder='images', transform=transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
#
# # Training loop
# num_epochs = 10
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# for epoch in range(num_epochs):
#     total_loss = 0.0
#     for images, _ in train_loader:  # ignore labels for autoencoder
#         images = images.to(device)
#
#         # Zero the gradients
#         optimizer.zero_grad()
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, images)  # Compare output with input
#
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     # Print loss for each epoch
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
#
# # Save the trained model
# torch.save(model.state_dict(), 'featurizer_model.pth')
