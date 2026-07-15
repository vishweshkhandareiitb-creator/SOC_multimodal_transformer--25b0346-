import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Flickr8kDataset, SimpleTokenizer
from clip_model import CLIPStyleModel
from vit import VisionTransformer
from text_encoder import TextTransformer

@torch.no_grad()
def search_images(query, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Data and Tokenizer
    captions_path = 'data/captions.txt'
    image_dir = 'data/Images'
    
    with open(captions_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()[1:]
    all_captions = [line.strip().split('|')[2].strip() for line in raw_lines if len(line.strip().split('|')) >= 3]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_captions)
    
    # We will search through a batch of 128 unseen validation images
    val_dataset = Flickr8kDataset(image_dir, captions_path, tokenizer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    batch = next(iter(val_loader))
    images = batch['image'].to(device)

    # 2. Load Model
    vit_encoder = VisionTransformer(image_size=64, patch_size=8, embed_dim=192, depth=4, num_heads=6, dropout=0.1)
    text_encoder = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=32, embed_dim=192, depth=4, num_heads=6, dropout=0.1)
    
    model = CLIPStyleModel(vit_encoder, text_encoder, embed_dim=192, projection_dim=128).to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=True))
    model.eval()

    # 3. Encode Images and Query
    print(f"Encoding images and searching for: '{query}'...")
    image_embeds = model.encode_image(images)
    image_embeds = F.normalize(image_embeds, dim=-1)
    
    # Tokenize the search query
    query_tokens = tokenizer.encode(query)[:32]
    query_mask = [1] * len(query_tokens) + [0] * (32 - len(query_tokens))
    query_tokens = query_tokens + [0] * (32 - len(query_tokens))
    
    q_tokens_tensor = torch.tensor([query_tokens]).to(device)
    q_mask_tensor = torch.tensor([query_mask]).to(device)
    
    text_embed = model.encode_text(q_tokens_tensor, q_mask_tensor)
    text_embed = F.normalize(text_embed, dim=-1)

    # 4. Compute similarities and get top K
    similarities = (text_embed @ image_embeds.T).squeeze(0)
    top_scores, top_indices = similarities.topk(top_k)

    # 5. Display the results
    fig, axes = plt.subplots(1, top_k, figsize=(15, 4))
    fig.suptitle(f"Search Query: '{query}'", fontsize=16)
    
    for i, idx in enumerate(top_indices):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        # Denormalize the image so it looks normal
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Match {i+1}\nScore: {top_scores[i]:.3f}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('search_results.png')
    plt.show()

if __name__ == '__main__':
    # You can change this text to search for whatever you want!
    search_query = "a dog running on the grass"
    search_images(search_query)