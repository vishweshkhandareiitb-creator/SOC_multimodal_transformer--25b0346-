import os
import time
import csv
import torch
from torch.utils.data import DataLoader
from dataset import Flickr8kDataset, SimpleTokenizer
from clip_model import CLIPStyleModel
from vit import VisionTransformer
from text_encoder import TextTransformer

def train():
    config = {
        'batch_size': 128,
        'lr': 5e-4,
        'weight_decay': 0.05,
        'warmup_steps': 50,
        'total_steps': 500,
        'grad_clip': 1.0,
        'projection_dim': 128,
        'embed_dim': 192,
        'image_size': 64,
        'patch_size': 8,
        'max_text_len': 32,
        'vit_depth': 4,
        'text_depth': 4,
        'n_head': 6,
        'dropout': 0.1,
        'val_every': 50,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training on: {device.type.upper()} ---")
    
    captions_path = 'data/captions.txt'
    image_dir = 'data/Images'
    
    with open(captions_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()[1:]
    all_captions = [line.strip().split('|')[2].strip() for line in raw_lines if len(line.strip().split('|')) >= 3]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_captions)
    
    train_dataset = Flickr8kDataset(image_dir, captions_path, tokenizer, split='train')
    val_dataset = Flickr8kDataset(image_dir, captions_path, tokenizer, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    vit_encoder = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['vit_depth'],
        num_heads=config['n_head'],
        dropout=config['dropout']
    )
    
    text_encoder = TextTransformer(
        vocab_size=tokenizer.vocab_size, 
        max_seq_len=config['max_text_len'],
        embed_dim=config['embed_dim'],
        depth=config['text_depth'],
        num_heads=config['n_head'],
        dropout=config['dropout']
    )
    
    model = CLIPStyleModel(
        vit_encoder, 
        text_encoder, 
        embed_dim=config['embed_dim'], 
        projection_dim=config['projection_dim']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        total_steps=config['total_steps'], 
        pct_start=config['warmup_steps'] / config['total_steps']
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    with open('training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss', 'lr', 'temperature'])

    step = 0
    best_val_loss = float('inf')
    model.train()
    
    print(f"Starting training for {config['total_steps']} steps...")
    start_time = time.time()
    
    while step < config['total_steps']:
        for batch in train_loader:
            if step >= config['total_steps']:
                break
                
            optimizer.zero_grad()
            
            images = batch['image'].to(device)
            tokens = batch['tokens'].to(device)
            masks = batch['mask'].to(device)

            with torch.cuda.amp.autocast():
                loss, _, _ = model(images, tokens, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if step % config['val_every'] == 0 or step == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_images = v_batch['image'].to(device)
                        v_tokens = v_batch['tokens'].to(device)
                        v_masks = v_batch['mask'].to(device)
                        with torch.cuda.amp.autocast():
                            v_loss, _, _ = model(v_images, v_tokens, v_masks)
                        val_loss += v_loss.item()
                val_loss /= len(val_loader)
                
                current_lr = scheduler.get_last_lr()[0]
                current_temp = model.loss_fn.log_inv_tau.exp().item() ** -1
                
                elapsed = time.time() - start_time
                print(f"Step {step:05d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Temp: {current_temp:.4f} | Time: {elapsed/60:.1f}m")
                
                with open('training_log.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, loss.item(), val_loss, current_lr, current_temp])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pt')
                    
                model.train()
                
            step += 1

if __name__ == '__main__':
    train()