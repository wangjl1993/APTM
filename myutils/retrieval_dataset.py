from pathlib import Path
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Callable, Union, Dict
import json

IMAGE_SUFFIX = [".jpg", ".png", ".jpeg", ".bmp"]

# cuhk dataset preprocessing 
TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555)),
])


class RetrievalImageDataset(Dataset):
    def __init__(self, root: str, transforms_func: Callable=TEST_TRANSFORMS) -> None:
        root = Path(root)
        self.transforms_func = transforms_func
        self.img_files = []
        for img_f in root.iterdir():
            if img_f.suffix in IMAGE_SUFFIX:
                self.img_files.append(img_f)

    @property
    def image_files(self,):
        return self.img_files
    
    def __len__(self,):
        return len(self.img_files)

    def __getitem__(self, index):
        img_f = self.img_files[index]
        data = Image.open(img_f)
        if self.transforms_func is not None:
            data = self.transforms_func(data)
        return data, str(img_f)


class RetrievalTextDataset(Dataset):
    def __init__(self, img_caption: Union[str, List[Dict]]) -> None:
        """bulid text dataset for retrievaling.

        Args:
            img_caption (Union[str, List]): file or list. including "caption" ("file_path")
        """
        super().__init__()
        if isinstance(img_caption, str):
            img_caption = Path(img_caption)
            assert img_caption.exists(), f"{img_caption} does not exist."
            with open(img_caption, "r") as f:
                img_caption = json.load(f)

        self.img_caption = img_caption
    
    def __len__(self,):
        return len(self.img_caption)
    
    def __getitem__(self, index):
        img_f, caption = self.img_caption[index].get("file_path", "None"), self.img_caption[index]["caption"]
        return img_f, caption
    
    @property
    def img_caption_info(self,):
        return self.img_caption
    