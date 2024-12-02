import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import json

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from featurizer_model import FeaturizerModel, FeaturizerModelPolicy


class FashionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
            img_dir (str): Directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

        # self.dataframe = self.dataframe[self.dataframe['image_signature'].apply(
        #     lambda x: os.path.exists(os.path.join(self.img_dir, x.split('\t')[0]+".jpg"))
        # )]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_signature = self.dataframe['image_signature'][idx].split('\t')[0]
        label = self.dataframe['image_signature'][idx].split('\t')[-1]

        # Load image
        for img_file in os.listdir(self.img_dir):
            if img_file.startswith(img_signature):
                img_path = os.path.join(self.img_dir, img_file)
                try:
                    image = Image.open(img_path).convert('RGB')

                    # Get bounding box coordinates
                    bounding_x = float(self.dataframe['image_signature'][idx].split('\t')[1])
                    bounding_y = float(self.dataframe['image_signature'][idx].split('\t')[2])
                    bounding_width = float(self.dataframe['image_signature'][idx].split('\t')[3])
                    bounding_height = float(self.dataframe['image_signature'][idx].split('\t')[4])

                    # Calculate actual pixel coordinates (assume the image is 1280x720, for example)
                    img_width, img_height = image.size
                    left = int(bounding_x * img_width)
                    top = int(bounding_y * img_height)
                    right = int((bounding_x + bounding_width) * img_width)
                    bottom = int((bounding_y + bounding_height) * img_height)

                    # Crop image to the bounding box
                    image = image.crop((left, top, right, bottom))

                    # Apply transformations if any
                    if self.transform:
                        image = self.transform(image)

                    return image, label
                except FileNotFoundError:
                    print(f"File {img_file} not found")
                break


# def load_image(image_signature, img_folder):
#     # Look for image matching the pattern "<image_signature>_<label>_<other_info>.jpg"
#     for img_file in os.listdir(img_folder):
#         if img_file.startswith(image_signature):
#             return Image.open(os.path.join(img_folder, img_file))
#     return None  # Return None if image not found

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to desired size
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    return tuple(zip(*batch)) if batch else ([], [])

img_folder = './index_images/'
# Load TSV data
data = pd.read_csv('./datasets/raw_train.tsv', sep=' ', engine='python')
data = data[:10000] # Using 10,000 For Training

dataset = FashionDataset(data, img_folder, transform)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_fn
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Program started on {device} ...")

# model = FeaturizerModel().to(device)
model = FeaturizerModelPolicy().to(device)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

losses = []
total_rewards = []

for epoch in range(100):
    rewards = []
    # log_probs = []
    epoch_rewards = []
    for batch in dataloader:
        images, labels = batch
        if len(images) == 0:
            continue
        images = torch.stack(images).to(device)
        labels = LabelEncoder().fit_transform(labels)
        labels = torch.tensor(labels).to(device)

        probs = model(images)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        reward = torch.tensor([1.0 if a==1 else -1.0 for a, l in zip(actions, labels)])
        rewards.append(reward)
        # log_probs.append(log_probs)
        epoch_rewards.append(reward.mean().item())

    rewards = torch.cat(rewards)
    log_probs = torch.cat(log_probs)
    loss = -(log_probs*rewards).mean()
    losses.append(loss.item())
    total_rewards.append(sum(epoch_rewards)/len(epoch_rewards))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch[{epoch+1}/100], Loss: {loss.item():.4f}, Avg Reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}')
#
# for epoch in range(100):
#     model.train()
#     running_loss = 0.0
#     # print(f"Epoch {epoch} is running ...")
#
#     for batch in dataloader:
#         images, labels = batch
#
#         if len(images) == 0:
#             continue
#
#         images = torch.stack(images).to(device)
#         labels = LabelEncoder().fit_transform(labels)
#         labels = torch.tensor(labels).to(device)
#
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     avg_loss = running_loss / len(dataloader)
#     print(f'Epoch [{epoch + 1}/{20}], Loss: {avg_loss:.4f}')
#     losses.append(avg_loss)

with open('reward_B32_0001_policy.json', 'w') as f:
    json.dump(total_rewards, f)
f.close()

with open('loss_B32_0001_policy.json', 'w') as f:
    json.dump(losses, f)
f.close()