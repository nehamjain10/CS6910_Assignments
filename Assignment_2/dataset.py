#Team  12   ['cavallo', 'farafalla', 'elefante', 'gatto', 'gallina']


from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import torch
from PIL import Image
class AnimalDataset(Dataset):
    """Animal Dataset"""

    def __init__(self,image_files,labels, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform  = transforms
        self.image_files = image_files
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.image_files[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image,label