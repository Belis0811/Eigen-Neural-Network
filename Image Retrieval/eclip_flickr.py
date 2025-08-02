from EResNet_BP import ResNet152 as EResNet
from EResNet_BP import ENNLinear
import EViT as ViT

import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.datasets import Flickr30k 
import os
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 1000
EMBED_DIM   = 256 
IMG_SIZE = 224
class EResNetEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        backbone = EResNet(num_classes=NUM_CLASS)
        # checkpoint = torch.load('/home1/anzheche/checkpoint_r101_image_bp.pth')
        # modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
        # backbone.load_state_dict(modified_state_dict)
        # print("Loaded checkpoint_r101_bp.pth")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_dim = self.backbone(dummy).shape[1] 
        
        self.projection = ENNLinear(feat_dim,embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
    
    def forward(self,x):
        features = self.backbone(x)
        proj = self.projection(features)
        return self.ln(proj)
        
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
MAX_TOK = 77  
tokenizer.model_max_length = MAX_TOK

for p in clip_text_model.parameters():  
    p.requires_grad = False
class ECLIP(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, text_dim=clip_text_model.config.hidden_size):
        super().__init__()
        self.image_encoder = EResNetEncoder(embed_dim=embed_dim)
        self.text_encoder = clip_text_model
        
        self.text_proj = ENNLinear(text_dim, embed_dim)
        self.ln_text = nn.LayerNorm(embed_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
    
    def encode_image(self, images):
        img = self.image_encoder(images)              
        return F.normalize(img, dim=1)
    
    def encode_text(self, input_ids, attention_mask):
        txt_features = self.text_encoder(input_ids=input_ids,
                                         attention_mask=attention_mask).pooler_output
        txt = self.ln_text(self.text_proj(txt_features))
        return F.normalize(txt, dim=1)
        
    def forward(self, images, input_ids, attention_mask):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)
        logits  = img_emb @ txt_emb.t() * self.logit_scale.exp()

        targets   = torch.arange(images.size(0), device=images.device)
        loss_i2t  = F.cross_entropy(logits, targets)
        loss_t2i  = F.cross_entropy(logits.t(), targets)
        return (loss_i2t + loss_t2i) / 2, logits

import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from pathlib import Path, PurePath
from torchvision.utils import save_image
import random
import pandas as pd
from torch.utils.data import Dataset, random_split
import torch.backends.cudnn as cudnn

BATCH_SIZE       = 18
BASE_LR          = 1e-4    
NUM_EPOCHS       = 1000
WEIGHT_DECAY     = 1e-3
ORTH_LOSS_WEIGHT = 5e-5   
SAVE_GRID_EVERY = 5

CSV_PATH   = "/home1/anzheche/Flickr30k/flickr30k_images/results.csv"
IMG_FOLDER = "/home1/anzheche/Flickr30k/flickr30k_images/flickr30k_images"

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
])

class FlickrCSV(Dataset):
    COLS = ["image_name", "comment_number", "comment"]

    def __init__(self, img_root, csv_file, indices=None, transform=None):
        df = pd.read_csv(csv_file, sep="\\|", engine="python")
        if list(df.columns) != self.COLS:      
            df = pd.read_csv(csv_file, sep="\\|", header=None,
                             names=self.COLS, engine="python")

        all_imgs = set(os.listdir(img_root))
        capt_map = (df.groupby("image_name")["comment"]
                      .apply(list).to_dict())
        self.ids = [k for k in capt_map if k in all_imgs]   
        self.captions  = {k: capt_map[k] for k in self.ids}

        if indices is not None:
            self.ids = [self.ids[i] for i in indices]

        self.img_root  = img_root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(os.path.join(self.img_root, img_id)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.captions[img_id][0]       # one caption

        
full_ds = FlickrCSV(IMG_FOLDER, CSV_PATH, transform=transform)

n = len(full_ds)
train_len = int(0.8 * n)
val_len   = int(0.1 * n)
test_len  = n - train_len - val_len
trainset, valset, testset = random_split(
        full_ds, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42))
        
# trainset = FlickrOneCaption(FLICKR_ROOT, "train", transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)

# testset = FlickrOneCaption(FLICKR_ROOT,  "val",   transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True)

@torch.no_grad()
def recall_k(model, loader, K=(1, 5, 10), chunk=8192):
    model.eval(); I, T = [], []
    for imgs, caps in loader:
        imgs = imgs.to(device)
        tok = tokenizer(caps, padding='max_length',
                truncation=True, max_length=MAX_TOK,
                return_tensors='pt').to(device)
        I.append(model.encode_image(imgs))
        T.append(model.encode_text(tok.input_ids, tok.attention_mask))
    I, T = torch.cat(I), torch.cat(T)
    N = I.size(0); hits = {k: 0 for k in K}

    for i in range(0, N, chunk):
        sims = (I[i:i+chunk] @ T.T) * model.logit_scale.exp()
        for k in K:
            hits[k] += (sims.topk(k, 1)[1]
                         == torch.arange(i, min(i+chunk, N), device=sims.device)[:, None]
                       ).any(1).sum().item()
    return {k: hits[k] / N for k in K}

def save_retrieval_examples(model, loader, epoch,
                            out_dir="runs/eclip_flickr30k/retrieval", n_samples=8):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    img_embs, caps_all, imgs_all = [], [], []
    for imgs, caps in loader:
        with torch.no_grad():
            emb = model.encode_image(imgs.to(device)).cpu()
        img_embs.append(emb); imgs_all.extend(imgs); caps_all.extend(caps)
    img_embs = torch.cat(img_embs)

    text_ids = random.sample(range(len(caps_all)), n_samples)
    text_tiles = []
    for tid in text_ids:
        caption = caps_all[tid]
        tok = tokenizer(caption, padding='max_length',
                truncation=True, max_length=MAX_TOK,
                return_tensors='pt').to(device) 
        with torch.no_grad():
            txt_emb = model.encode_text(tok.input_ids, tok.attention_mask).cpu()
        top_img = imgs_all[(txt_emb @ img_embs.T).topk(1).indices.item()]
        text_tiles.append(top_img)
        (Path(out_dir) / f"t2i_{epoch:03d}_{tid}.txt").write_text(caption)

    save_image(torch.stack(text_tiles),
               f"{out_dir}/epoch_{epoch:03d}_t2i.png", nrow=4)

    img_ids = random.sample(range(len(imgs_all)), n_samples)
    for iid in img_ids:
        img = imgs_all[iid]
        img_p = img.unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img_p).cpu()
        top_cap_idx = (img_emb @ torch.cat(
            [model.encode_text(*tokenizer(caps_all[j], padding='max_length',
                truncation=True, max_length=MAX_TOK, return_tensors='pt')
                               .to(device).values()).cpu()
             for j in range(len(caps_all))]).T).topk(1).indices.item()

        save_image(img, f"{out_dir}/epoch_{epoch:03d}_i2t_{iid}.png")
        (Path(out_dir) / f"i2t_{epoch:03d}_{iid}.txt").write_text(caps_all[top_cap_idx])
    print(f"[retrieval] epoch {epoch} saved bidirectional examples → {out_dir}")

model = ECLIP()
    
try:
    checkpoint = torch.load('checkpoint_eclip_flickr.pth')
    modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(modified_state_dict)
    print("Loaded checkpoint_eclip_flickr.pth – continuing training …")
except FileNotFoundError:
    print("No existing checkpoint found – training from scratch …")
    
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

# if torch.cuda.device_count() > 1:
#     #print(torch.cuda.device_count())
#     model = nn.DataParallel(model)
    
model.to(device)


log_file = open("training_log_eclip_flickr.txt", "w")
best_train_r1 = 0.0
best_val_r1   = 0.0
best_val_mrr   = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0
    correct10 = 0.0
    seen = 0.0
    
    for i, (imgs, caps) in enumerate(tqdm(trainloader)):
        imgs = imgs.to(device)
        tokens = tokenizer(caps, padding=True, truncation=True, max_length=MAX_TOK,
                           return_tensors="pt").to(device)

        loss, logits = model(imgs, tokens.input_ids, tokens.attention_mask)

        if ORTH_LOSS_WEIGHT > 0:
            orth_loss = 0.0
            for module in model.modules():
                if hasattr(module, 'orthonormalize'):
                    orth_loss += module.orthonormalize()
            loss += ORTH_LOSS_WEIGHT * orth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        top5 = logits.topk(5, dim=1).indices          
        top1 = top5[:, 0]
        top10 = logits.topk(10, dim=1).indices
        
        labels_batch = torch.arange(imgs.size(0), device=imgs.device)
        correct1 += (top1 == labels_batch).sum().item()
        correct5 += (top5 == labels_batch.unsqueeze(1)).any(1).sum().item()
        correct10 += (top10 == labels_batch.unsqueeze(1)).any(1).sum().item()
        seen     += imgs.size(0)
        
        if (i + 1) % 10 == 0:
            torch.save(model.state_dict(), 'checkpoint_eclip_coco.pth')
            # print("Checkpoint saved → checkpoint_eclip_coco.pth")

    train_top1 = correct1 / seen
    train_top5 = correct5 / seen
    train_top10 = correct10 / seen
    avg_loss   = total_loss / len(trainloader)
    best_train_r1 = max(best_train_r1, train_top1)
    print(f"Epoch: {epoch}/{NUM_EPOCHS} | loss: {avg_loss:.4f} |  Recall@1: {train_top1:.3f} | Recall@5: {train_top5:.3f} | Recall@10: {train_top10:.3f}")
    log_file.write(f"Epoch: {epoch}/{NUM_EPOCHS} | loss: {avg_loss:.4f} |  Recall@1: {train_top1:.3f} | Recall@5: {train_top5:.3f} | Recall@10: {train_top10:.3f}\n")
    log_file.write(f"Best Train Recall@1: {best_train_r1}\n")
    
    
    model.eval()
    rec = recall_k(model, testloader)
    print(f"Test Recall@1/5/10 {rec[1]:.3f}/{rec[5]:.3f}/{rec[10]:.3f}")
    
    best_val_r1 = max(best_val_r1,rec[1])

    
    if (epoch + 1) % SAVE_GRID_EVERY == 0: 
        save_retrieval_examples(model, testloader, epoch)
    
    log_file.write(f"Test Recall@1/5/10 {rec[1]:.3f}/{rec[5]:.3f}/{rec[10]:.3f}\n")
    log_file.write(f"Best Test Recall@1: {best_val_r1}\n\n")
    log_file.flush()
    
    