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
import csv

pretrained_model = models.deit_base_distilled_patch16_224(pretrained=True)
pretrained_model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kd_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.9):
    
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    #L_KD
    soft_loss = -torch.sum(teacher_probs * torch.log(student_probs), dim=1).mean()
    
    '''soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                         F.softmax(teacher_logits / temperature, dim=1),
                         reduction='batchmean') * (temperature ** 2)'''
    
    hard_loss = F.cross_entropy(student_logits, labels)

    return (1. - alpha) * soft_loss + alpha * hard_loss, hard_loss, soft_loss


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
        
        if x.shape[-3] == 1:
            x = x.repeat(1, 3, 1, 1)
        B = x.shape[0]

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
            import torchvision.transforms.functional as F
            image = torch.from_numpy(image)
            image = F.resize(image, size=(256, 256))

        return image, int(self.labels[f'{index+1:04d}']), rgb_img
    
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
    
def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)

def main(output_dir = None, checkpoint = None):
    
    training_records = {
        "num_epoch": [],
        "learning_rate": [],
        "cross-entropy loss(hard loss)": [],
        "distillation loss(soft loss)": [],
        "both hard and soft loss": [],
        "accuracy": [],
    }
    record = True 
    learning_rate = 0.001
    
    max_accuracy = 0.0 
    
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = DistilledVisionTransformer(
        img_size=256,  
        patch_size=16,  
        embed_dim=768, 
        depth=12,  
        num_heads=12,  
        num_classes=1000,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 200  
    
    if checkpoint:
        model, optimizer = load_checkpoint(checkpoint, model, optimizer)

    pretrained_model.to(device)
    model.to(device)
    for epoch in range(num_epochs):
        #model.train()
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        correct = 0
        total = 0
        for images, labels, rgb_img in train_loader:
            images= images.to(device)
            raw_output, raw_output_dist = model(images)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            with torch.no_grad():
                rgb_img = rgb_img.to(device)
                RGB_output = pretrained_model(rgb_img)
            
            loss, cross_loss, dist_loss = kd_loss(raw_output, RGB_output, labels, temperature=4.0, alpha=0.9)

            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()
            
            running_loss += loss.item()
            running_hard_loss += cross_loss.item()
            running_soft_loss += dist_loss.item()
            accuracy = calculate_accuracy(raw_output, labels)
            correct += (raw_output.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_soft_loss = running_soft_loss / len(train_loader)
        epoch_hard_loss = running_hard_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print("---")
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print("---")
        
        if (epoch_accuracy > max_accuracy) and output_dir:
            print(f"Saving best model with accuracy {epoch_accuracy:.2f}%")
            max_accuracy = epoch_accuracy
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "accuracy": max_accuracy,
            }, output_dir)
            
        if record:
            training_records["num_epoch"].append(epoch + 1)
            training_records["learning_rate"].append(learning_rate)
            training_records["cross-entropy loss(hard loss)"].append(epoch_hard_loss)  # Or "Cross-Entropy"/"Distillation" based on your condition
            training_records["distillation loss(soft loss)"].append(epoch_soft_loss)
            training_records["both hard and soft loss"].append(epoch_loss)
            training_records["accuracy"].append(epoch_accuracy)
    
    if record:
        save_training_records_to_csv(training_records)
            
def save_training_records_to_csv(records, filename="training_records.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(records.keys())  # Write the header
        writer.writerows(zip(*records.values()))  # Write the rows

if __name__ == '__main__':
    save_path = "/home/yzhang63/deit-main/checkpoint.pth"
    main(save_path, None)
