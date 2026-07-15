import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Flickr8kDataset, SimpleTokenizer

def main():
    captions_path = 'data/captions.txt'
    image_dir = 'data/Images'

    with open(captions_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()[1:]

    all_captions = []
    for line in raw_lines:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            all_captions.append(parts[2].strip())

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_captions)

    train_dataset = Flickr8kDataset(
        image_dir=image_dir, 
        captions_file=captions_path, 
        tokenizer=tokenizer, 
        split='train'
    )

    sample = train_dataset[0]
    print("Image shape:", sample['image'].shape)
    print("Token shape:", sample['tokens'].shape)
    print("Caption:", sample['caption'])

    print("\n--- 10 Sample Captions ---")
    for i in range(10):
        print(train_dataset[i]['caption'])

    print("\n--- Timing One Epoch ---")
    # The num_workers=2 here is what requires the __main__ block on Windows
    loader = DataLoader(train_dataset, batch_size=128, num_workers=2)

    start_time = time.time()
    for batch in loader:
        pass
    end_time = time.time()

    print(f"Epoch load time: {end_time - start_time:.2f} seconds")

    img = sample['image'].permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()