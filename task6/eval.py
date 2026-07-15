import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Flickr8kDataset, SimpleTokenizer
from clip_model import CLIPStyleModel
from vit import VisionTransformer
from text_encoder import TextTransformer

@torch.no_grad()
def evaluate_retrieval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on: {device}")

    # 1. Setup Data and Tokenizer
    captions_path = 'data/captions.txt'
    image_dir = 'data/Images'
    
    with open(captions_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()[1:]
    all_captions = [line.strip().split('|')[2].strip() for line in raw_lines if len(line.strip().split('|')) >= 3]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_captions)
    val_dataset = Flickr8kDataset(image_dir, captions_path, tokenizer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 2. Setup Model and Load Weights
    vit_encoder = VisionTransformer(image_size=64, patch_size=8, embed_dim=192, depth=4, num_heads=6, dropout=0.1)
    text_encoder = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=32, embed_dim=192, depth=4, num_heads=6, dropout=0.1)
    
    model = CLIPStyleModel(vit_encoder, text_encoder, embed_dim=192, projection_dim=128).to(device)
    
    try:
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        print("Successfully loaded best_model.pt")
    except FileNotFoundError:
        print("Error: best_model.pt not found. Did training finish?")
        return
        
    model.eval()

    # 3. Extract all embeddings
    all_image_embeds, all_text_embeds = [], []
    print("Extracting embeddings for the validation set...")
    
    for batch in val_loader:
        img_e = model.encode_image(batch['image'].to(device))
        txt_e = model.encode_text(batch['tokens'].to(device), batch['mask'].to(device))
        
        all_image_embeds.append(F.normalize(img_e, dim=-1).cpu())
        all_text_embeds.append(F.normalize(txt_e, dim=-1).cpu())

    image_embeds = torch.cat(all_image_embeds, 0)
    text_embeds = torch.cat(all_text_embeds, 0)

    # 4. Compute Similarity and Recall
    print("Computing Recall metrics...")
    sim_matrix = image_embeds @ text_embeds.T
    N = sim_matrix.shape[0]
    labels = torch.arange(N)

    def get_recall_at_k(sim, labels, k=1):
        top_k_indices = sim.topk(k, dim=1)[1]
        correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
        return (correct.float().mean().item()) * 100

    i2t_r1 = get_recall_at_k(sim_matrix, labels, k=1)
    i2t_r5 = get_recall_at_k(sim_matrix, labels, k=5)
    i2t_r10 = get_recall_at_k(sim_matrix, labels, k=10)

    t2i_r1 = get_recall_at_k(sim_matrix.T, labels, k=1)
    t2i_r5 = get_recall_at_k(sim_matrix.T, labels, k=5)
    t2i_r10 = get_recall_at_k(sim_matrix.T, labels, k=10)

    print("\n=== Retrieval Results ===")
    print(f"Image -> Text | R@1: {i2t_r1:.1f}% | R@5: {i2t_r5:.1f}% | R@10: {i2t_r10:.1f}%")
    print(f"Text -> Image | R@1: {t2i_r1:.1f}% | R@5: {t2i_r5:.1f}% | R@10: {t2i_r10:.1f}%")
    
    # Random chance for 1000 items is ~0.1%
    print("\n(Note: Random chance is ~0.1%. If your scores are higher, your model actually learned!)")

if __name__ == '__main__':
    evaluate_retrieval()

    