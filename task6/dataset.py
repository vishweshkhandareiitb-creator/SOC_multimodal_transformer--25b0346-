import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SimpleTokenizer:
    def __init__(self):
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2}
        self.id2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>"}
        self.vocab_size = 3

    def build_vocab(self, captions):
        for caption in captions:
            for word in caption.lower().strip().split():
                if word not in self.word2id:
                    self.word2id[word] = self.vocab_size
                    self.id2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text):
        tokens = [2]
        for word in text.lower().strip().split():
            tokens.append(self.word2id.get(word, 1))
        return tokens

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, image_size=64, max_text_len=32, split='train'):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.pairs = self._load_pairs(captions_file, split)
        
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_pairs(self, captions_file, split):
        pairs = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        unique_images = []
        for line in lines[1:]: 
            parts = line.strip().split('|')
            if len(parts) >= 3:
                img_name = parts[0].strip()
                caption = parts[2].strip()
                pairs.append((img_name, caption))
                if img_name not in unique_images:
                    unique_images.append(img_name)
        
        if split == 'train':
            split_imgs = set(unique_images[:6000])
        elif split == 'val':
            split_imgs = set(unique_images[6000:7000])
        else:
            split_imgs = set(unique_images[7000:])
            
        return [p for p in pairs if p[0] in split_imgs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        tokens = self.tokenizer.encode(caption)
        tokens = tokens[:self.max_text_len]
        mask = [1] * len(tokens) + [0] * (self.max_text_len - len(tokens))
        tokens = tokens + [0] * (self.max_text_len - len(tokens))

        return {
            'image': image,
            'tokens': torch.tensor(tokens),
            'mask': torch.tensor(mask),
            'caption': caption,
        }