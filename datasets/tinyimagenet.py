import os

from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    """
    A PyTorch Dataset for TinyImageNet

    Directory structure is assumed to be:
      root/
          wnids.txt
          words.txt  (optional)
          train/
              <wnid>/
                  images/
                      *.JPEG
          val/
              images/
                  *.JPEG
              val_annotations.txt
          test/
              images/
                  *.JPEG

    Args:
        root_dir (str): Root directory of the TinyImageNet dataset
        split (str): 'train', 'val', or 'test'
        use_words (bool): If True, load and use words.txt to map wnid to human-readable names
        transform (callable, optional): Transform to be applied on an image.
    """
    def __init__(self, root_dir, split='train', use_words=False, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.use_words = use_words
        self.transform = transform

        # load wnids from file and sort them to ensure consistency
        wnids_path = os.path.join(root_dir, 'wnids.txt')
        if not os.path.isfile(wnids_path):
            raise FileNotFoundError(f"File not found: {wnids_path}")
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f if line.strip()]
        # sort to have a deterministic order
        self.classes = sorted(wnids)
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.classes)}

        # optionally load human-readable class names from words.txt
        if self.use_words:
            words_path = os.path.join(root_dir, 'words.txt')
            if os.path.isfile(words_path):
                self.wnid_to_name = {}
                with open(words_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            wnid, names = parts[0], parts[1]
                            self.wnid_to_name[wnid] = names
                # create a mapping from class index to name (fallback to wnid if missing)
                self.idx_to_class = [self.wnid_to_name.get(wnid, wnid) for wnid in self.classes]
            else:
                self.idx_to_class = self.classes
        else:
            self.idx_to_class = self.classes

        self.img_paths = []
        self.labels = []

        if self.split == 'train':
            self._load_train_data()
        elif self.split == 'val':
            self._load_val_data()
        elif self.split == 'test':
            self._load_test_data()
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")

    def _load_train_data(self):
        train_dir = os.path.join(self.root_dir, 'train')
        for wnid, label in self.class_to_idx.items():
            class_dir = os.path.join(train_dir, wnid, 'images')
            if not os.path.isdir(class_dir):
                continue  # skip missing classes (shouldn't happen)
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):
                    self.img_paths.append(fpath)
                    self.labels.append(label)

    def _load_val_data(self):
        val_dir = os.path.join(self.root_dir, 'val')
        images_dir = os.path.join(val_dir, 'images')
        ann_file = os.path.join(val_dir, 'val_annotations.txt')
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"Validation annotations file not found: {ann_file}")
        # parse annotations: each line has img_name, wnid, and extra info
        img_to_wnid = {}
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, wnid = parts[0], parts[1]
                    img_to_wnid[img_name] = wnid

        for fname in os.listdir(images_dir):
            fpath = os.path.join(images_dir, fname)
            if os.path.isfile(fpath):
                wnid = img_to_wnid.get(fname)
                if wnid is None or wnid not in self.class_to_idx:
                    continue # skip if invalid class (shouldn't happen)
                self.img_paths.append(fpath)
                self.labels.append(self.class_to_idx[wnid])

    def _load_test_data(self):
        test_dir = os.path.join(self.root_dir, 'test', 'images')
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        for fname in os.listdir(test_dir):
            fpath = os.path.join(test_dir, fname)
            if os.path.isfile(fpath):
                self.img_paths.append(fpath)
                self.labels.append(-1)  # label placeholder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label
        