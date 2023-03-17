from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((384,384)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
