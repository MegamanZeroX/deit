import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rawpy
import numpy as np
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer, trunc_normal_
import models
import torch.nn.functional as F
import os


pretrained_model = model = models.deit_base_distilled_patch16_224(pretrained=True)
pretrained_model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kd_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.9):
    soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                         F.softmax(teacher_logits / temperature, dim=1),
                         reduction='batchmean') * (temperature ** 2)
    
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1. - alpha) * hard_loss

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        
        x = x.unsqueeze(1)
        B = x.shape[0]
        print(x.shape)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2




class DNGDataset(Dataset):
    def __init__(self, image_paths, rgb_image_paths, labels, transform=None, RGB_transforms = None):
        self.image_paths = image_paths
        self.rgb_image_paths = rgb_image_paths
        self.labels = self.parse_label_file(labels)
        self.transform = transform
        self.RGB_transforms = RGB_transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        file = os.listdir(self.image_paths)[index]
        rgb_file = file.split(".")[0] +f'.jpg'
        file_path = os.path.join(self.image_paths, file)
        rgb_filepath = os.path.join(self.rgb_image_paths, rgb_file)
        
        with rawpy.imread(file_path) as raw:
            image = np.asarray(raw.raw_image).astype(np.float32)
            image = image.reshape(1, image.shape[0], image.shape[1])
            #image = np.array(image)

        with Image.open(rgb_filepath) as rgb_img:
            rgb_img = rgb_img.convert('RGB')

        if self.RGB_transforms:
            rgb_img = self.RGB_transforms(rgb_img)
            
        if self.transform:
            image = self.transform(image)

        return image, self.labels[f'{index+1:04d}'], rgb_img
    
    def parse_label_file(self, label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()

        label_mapping = {}
        for line in lines:
            parts = line.strip().split(' ')
            filename = parts[0]
            labels = parts[1]
            label_mapping[filename.split('.')[0]] = labels

        return label_mapping

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  
])

RGB_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

train_image_paths = '/home/yzhang63/deit-main/data/S7_ISP_Dataset/medium_dng'
train_labels = '/home/yzhang63/deit-main/predictions.txt'
rgb_image_paths = '/home/yzhang63/deit-main/data/S7_ISP_Dataset/medium_jpg'
train_dataset = DNGDataset(train_image_paths, rgb_image_paths, train_labels, transform=transform, RGB_transforms=RGB_transforms)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

model = DistilledVisionTransformer(
    img_size=256,  
    patch_size=16,  
    embed_dim=768, 
    depth=12,  
    num_heads=12,  
    num_classes=1000,
)

criterion_withpretrained = nn.CrossEntropyLoss()
criterion_withgroundtrth = nn.CrossEntropyLoss()
alpha = 0.9


optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10  

pretrained_model.to(device)

for epoch in range(num_epochs):
    for images, labels, rgb_img in train_loader:
        images= images.to(device)
        model.to(device)
        raw_output, raw_dist_output = model(images)
        
        with torch.no_grad():
            RGB_output = pretrained_model(rgb_img)
        
        loss = kd_loss(raw_output, RGB_output, labels, temperature=4.0, alpha=0.9)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

