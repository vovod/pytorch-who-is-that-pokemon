from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

data_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])