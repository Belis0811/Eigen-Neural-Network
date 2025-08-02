import math, random, argparse, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import open_clip
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from torchvision.utils import save_image
from transformers import CLIPModel, CLIPTokenizer     

from EResNet_BP import ResNet152 as EResNet
from EResNet_BP import ENNLinear

MODEL_ID      = "zer0int/LongCLIP-L-Diffusers"
ALPHA         = 0.2  
EMBED_DIM        = 256
IMG_SIZE         = 224
NUM_CLASS        = 80
BATCH_SIZE       = 100
LR               = 1e-4
WD               = 1e-3
ORTH_W           = 5e-5
EPOCHS           = 1000
SAVE_GRID_EVERY  = 5
DEVICE           = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class EResNetEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        backbone = EResNet(num_classes=NUM_CLASS)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_dim = self.backbone(dummy).shape[1] 
        
        self.proj = ENNLinear(feat_dim,embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
    
    def forward(self,x, return_patches=False):
        if return_patches:                       
            feats = self.backbone.forward_features(x)  
            B,C,H,W = feats.shape
            patches = feats.flatten(2).permute(0,2,1)    
            patches = self.ln(self.proj(patches))          
            g = self.ln(self.proj(F.adaptive_avg_pool2d(feats,1).flatten(1)))
            return F.normalize(g,dim=1), F.normalize(patches,dim=-1)
        else:
            g = self.backbone(x)              
            return F.normalize(self.ln(self.proj(g)), dim=1)


tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
txt_model = (CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval())         
tokenizer.model_max_length = 248   

@torch.no_grad()
def encode_text(long_caps, short_caps):
    
    if isinstance(long_caps, torch.Tensor):  long_caps  = long_caps.tolist()
    if isinstance(short_caps, torch.Tensor): short_caps = short_caps.tolist()

    tok_long  = tokenizer(long_caps,  padding=True, truncation=True,
                          return_tensors="pt").to(DEVICE)
    tok_short = tokenizer(short_caps, padding=True, truncation=True,
                          return_tensors="pt").to(DEVICE)

    seq_long  = txt_model.text_model(**tok_long,
                                     output_hidden_states=True).last_hidden_state
    t_fine768 = seq_long[:, 0]
    
    t_coarse768 = txt_model.get_text_features(**tok_short)

    T_fine   = F.normalize(model.txt_proj(t_fine768),   dim=1)
    T_coarse = F.normalize(model.txt_proj(t_coarse768), dim=1)

    return T_fine, T_coarse


def pce(patch):                            
    patch = patch - patch.mean(1, keepdim=True)
    u,s,vh = torch.linalg.svd(patch, full_matrices=False)
    pc = vh[:, 0, :]                                   
    return F.normalize(pc, dim=1)

class ENN_LongCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = EResNetEncoder().to(DEVICE).eval()  
        hidden = txt_model.config.text_config.hidden_size
        self.txt_proj = ENNLinear(hidden, 256)  
        self.ln_txt   = nn.LayerNorm(256)
        self.t_log   = nn.Parameter(torch.ones([])*math.log(1/0.07))

    def forward(self, imgs, long_caps, short_caps):
        with torch.no_grad():
            I_fine, I_patch = self.img_enc(imgs, return_patches=True)
        T_fine, T_coarse   = encode_text(long_caps, short_caps)
        
        # T_fine   = F.normalize(self.ln_txt(self.txt_proj(T_fine)),   dim=1)
        # T_coarse = F.normalize(self.ln_txt(self.txt_proj(T_coarse)), dim=1)

        logits_f = I_fine @ T_fine.T * self.t_log.exp()
        lab      = torch.arange(len(imgs), device=DEVICE)
        loss_f   = F.cross_entropy(logits_f, lab)

        I_coarse = pce(I_patch)
        logits_c = I_coarse @ T_coarse.T * self.t_log.exp()
        loss_c   = F.cross_entropy(logits_c, lab)

        return loss_f + ALPHA*loss_c, logits_f

    def encode_image(self, x): return self.img_enc(x)
    def encode_text (self, ids, mask):
        feat = txt_model.get_text_features(input_ids=ids,
                                           attention_mask=mask)
        return F.normalize(self.ln_txt(self.txt_proj(feat)), dim=1)

def coco_loaders(root="./COCO"):
    T = transforms.Compose([
        transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466,0.4578275,0.40821073),
                             (0.26862954,0.26130258,0.27577711)),
    ])
    class OneCap(torch.utils.data.Dataset):
        def __init__(self, root, ann):
            self.ds = CocoCaptions(root, ann, transform=T)
        def __len__(self):  return len(self.ds)
        def __getitem__(self, idx):
            img, caps = self.ds[idx]
            long  = caps[0]                   
            short = " ".join(long.split()[:16])   
            return img, long, short            
    tr = OneCap(f"{root}/train2017", f"{root}/annotations/captions_train2017.json")
    va = OneCap(f"{root}/val2017",   f"{root}/annotations/captions_val2017.json")
    return (torch.utils.data.DataLoader(tr, BATCH_SIZE, True,  pin_memory=True),
            torch.utils.data.DataLoader(va, BATCH_SIZE, False, pin_memory=True))

@torch.no_grad()
def recall(model, loader, K=(1,5,10), chunk=8192):
    model.eval(); I=[]; T=[]
    for imgs,longs, _ in loader:
        tok=tokenizer(longs,padding=True,truncation=True,return_tensors="pt").to(DEVICE)
        I.append(model.encode_image(imgs.to(DEVICE)))
        T.append(model.encode_text(tok.input_ids, tok.attention_mask))
    I,T=torch.cat(I),torch.cat(T)
    N = I.size(0); hits = {k: 0 for k in K}

    for i in range(0, N, chunk):
        sims = (I[i:i+chunk] @ T.T) * model.t_log.exp()
        for k in K:
            hits[k] += (sims.topk(k, 1)[1]
                         == torch.arange(i, min(i+chunk, N), device=sims.device)[:, None]
                       ).any(1).sum().item()
    return {k: hits[k] / N for k in K}


def save_examples(model, loader, epoch,
                            out_dir="runs/longeclip_coco/retrieval", n_samples=8):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    img_embs, caps_all, imgs_all = [], [], []
    for imgs, caps, *_  in loader:
        with torch.no_grad():
            emb = model.encode_image(imgs.to(DEVICE)).cpu()
        img_embs.append(emb); imgs_all.extend(imgs); caps_all.extend(caps)
    img_embs = torch.cat(img_embs)

    text_ids = random.sample(range(len(caps_all)), n_samples)
    text_tiles = []
    for tid in text_ids:
        caption = caps_all[tid]
        tok = tokenizer(caption, padding=True, truncation=True, return_tensors='pt').to(DEVICE) 
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
        img_p = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_emb = model.encode_image(img_p).cpu()
        top_cap_idx = (img_emb @ torch.cat(
            [model.encode_text(*tokenizer(caps_all[j], return_tensors='pt')
                               .to(DEVICE).values()).cpu()
             for j in range(len(caps_all))]).T).topk(1).indices.item()

        save_image(img, f"{out_dir}/epoch_{epoch:03d}_i2t_{iid}.png")
        (Path(out_dir) / f"i2t_{epoch:03d}_{iid}.txt").write_text(caps_all[top_cap_idx])
    print(f"[retrieval] epoch {epoch} saved bidirectional examples → {out_dir}")

if __name__=="__main__":
    train,val=coco_loaders()
    model=ENN_LongCLIP().to(DEVICE)
    
    try:
        model.load_state_dict(torch.load('checkpoint_elongclip_coco.pth'))
        print("Loaded checkpoint_elongclip_coco.pth – continuing training …")
    except FileNotFoundError:
        print("No existing checkpoint found – training from scratch …")
    
    opt=torch.optim.AdamW(model.parameters(),LR,weight_decay=WD)

    log_file = open("training_log_elongclip_coco.txt", "w")
    best_train_i2t_r1 = 0.0

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for i, (imgs, longs, shorts) in enumerate(tqdm(train)):
            loss,_ = model(imgs.to(DEVICE),longs, shorts)
            if ORTH_W>0:
                loss+=ORTH_W*sum(m.orthonormalize() for m in model.modules() if hasattr(m,"orthonormalize"))
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                torch.save(model.state_dict(), 'checkpoint_elongclip_coco.pth')

        rec_i2t=recall(model,val)
  
        avg_loss   = total_loss / len(train)
        best_train_i2t_r1 = max(best_train_i2t_r1, rec_i2t[1])

        
        print(f"ep{ep:03d}  I→T R@1/5/10 {rec_i2t[1]:.3f}/{rec_i2t[5]:.3f}/{rec_i2t[10]:.3f}")
        
        log_file.write(f"epoch: {ep:03d}/{EPOCHS} | loss: {avg_loss:.4f} |  I→T R@1/5/10 {rec_i2t[1]:.3f}/{rec_i2t[5]:.3f}/{rec_i2t[10]:.3f}\n")
        log_file.write(f"Best Train I→T Recall@1: {best_train_i2t_r1:.3f}\n\n")
        log_file.flush()
        
        if (ep+1)%SAVE_GRID_EVERY==0:
            model.eval()
            save_examples(model,val,ep)
