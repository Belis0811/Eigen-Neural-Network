import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import tiny_imagenet_loader
from tqdm import tqdm
import os

from EResNet import ResNet152 as EResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=64
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNEL = 3

train_directory = './tiny-imagenet-200/train'
test_directory = './tiny-imagenet-200/'

class EarlyStopping:
    def __init__(self, patience=3, verbose=True, name="null"):
        self.patience = patience
        self.verbose = verbose
        self.name = name
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
])

trainset = torchvision.datasets.ImageFolder(
    root=train_directory, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)

testset = tiny_imagenet_loader.TinyImageNet_load(
    root=test_directory, train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True)

model = EResNet(num_classes=200).to(device)
model.assign_local_heads(num_classes=200, C=IMG_CHANNEL, H=IMG_HEIGHT, W=IMG_WIDTH)

checkpoint = 'checkpoint_r152_tiny.pth'
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    print("Weight Loaded!")
else:
    print("Train from scratch!")

criterion = nn.CrossEntropyLoss()
optim_conv1 = optim.AdamW(list(model.conv1.parameters()) + list(model.bn1.parameters()),lr=1e-4)

optim_layer1 = [optim.AdamW(block.parameters(), lr=1e-4) for block in model.layer1]
optim_layer2 = [optim.AdamW(block.parameters(), lr=1e-4) for block in model.layer2]
optim_layer3 = [optim.AdamW(block.parameters(), lr=1e-4) for block in model.layer3]
optim_layer4 = [optim.AdamW(block.parameters(), lr=1e-4) for block in model.layer4]

# optim_layer1 = [optim.AdamW(block.parameters(), lr=5e-4) for block in model.layer1]
# optim_layer2 = [optim.AdamW(block.parameters(), lr=5e-4) for block in model.layer2]
# optim_layer3 = [optim.AdamW(block.parameters(), lr=5e-4) for block in model.layer3]
# optim_layer4 = [optim.AdamW(block.parameters(), lr=5e-4) for block in model.layer4]

optim_fc = optim.AdamW(model.fc.parameters(), lr=1e-4)

MAX_LR_FCTR = 5       
num_epochs = 500
early_stopping = EarlyStopping(patience=5, verbose=True)

def make_onecycle(optimiser, base_lr):
    return optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr   = base_lr * MAX_LR_FCTR,
        total_steps = num_epochs * len(trainloader),
        pct_start   = 0.35,            # ramp-up for 35 % of schedule
        anneal_strategy='cos',
        div_factor   = 25,             # initial LR = max_lr / div_factor
        final_div_factor = 1e3         # final LR  = initial / final_div_factor
    )

# sched_conv1 = make_onecycle(optim_conv1, 5e-5)
# sched_fc    = make_onecycle(optim_fc,    8e-4)
# sched_l1    = [make_onecycle(o, 4e-4) for o in optim_layer1]
# sched_l2    = [make_onecycle(o, 6e-4) for o in optim_layer2]
# sched_l3    = [make_onecycle(o, 6e-4) for o in optim_layer3]
# sched_l4    = [make_onecycle(o, 8e-4) for o in optim_layer4]

sched_conv1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_conv1, T_max=200)
sched_fc    = torch.optim.lr_scheduler.CosineAnnealingLR(optim_fc,    T_max=200)
sched_l1    = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200) for o in optim_layer1]
sched_l2    = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200) for o in optim_layer2]
sched_l3    = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200) for o in optim_layer3]
sched_l4    = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200) for o in optim_layer4]


#orth_loss_weight=1e-5
orth_loss_weight=1e-4

from ResNet_ENN import ENNConv2d, ENNLinear
import torch.nn.functional as F

log_file = open("training_log_renn_r152_tiny.txt", "w")
best_train = 0.0
best_val = 0.0

for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i,(images, labels) in enumerate(tqdm(trainloader)):
            images, labels = images.to(device), labels.to(device)
            
            # conv1 layer local training
            out = F.relu(model.bn1(model.conv1(images)))
            out_flat = out.view(out.size(0), -1)
            logits = model.conv1_head(out_flat)
            loss = criterion(logits, labels)
            loss += orth_loss_weight * model.conv1.orthonormalize()
            optim_conv1.zero_grad()
            loss.backward()
            optim_conv1.step()
            sched_conv1.step()
            out = out.detach()
            
            # layer 1 local
            for i, block in enumerate(model.layer1):
                out = block.train_step(out, labels, criterion, optim_layer1[i], orth_loss_weight)
                sched_l1[i].step()
                
            # layer 2 local
            for i, block in enumerate(model.layer2):
                out = block.train_step(out, labels, criterion, optim_layer2[i], orth_loss_weight)
                sched_l2[i].step()

            # layer 3 local
            for i, block in enumerate(model.layer3):
                out = block.train_step(out, labels, criterion, optim_layer3[i], orth_loss_weight)
                sched_l3[i].step()

            # layer 4 local
            for i, block in enumerate(model.layer4):
                out = block.train_step(out, labels, criterion, optim_layer4[i], orth_loss_weight)
                sched_l4[i].step()
            
            # fc layer local     
            out_pool = F.avg_pool2d(out, kernel_size=out.size()[2:]) 
            out_flat = out_pool.view(out_pool.size(0), -1)
            logits = model.fc(out_flat)
            loss = criterion(logits, labels)
            loss += orth_loss_weight * model.fc.orthonormalize()
            optim_fc.zero_grad()
            loss.backward()
            optim_fc.step()
            sched_fc.step()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
               
            running_loss += loss.item()
            
      
        epoch_loss = running_loss / len(trainloader.dataset)
        t_accuracy = 100.0 * correct / total
        
        best_train = max(best_train,t_accuracy)
        
        if early_stopping(epoch_loss):
                print('Early stopping triggered')
                break
            
        if epoch % 10 == 0:
            print(f'Saving model...')
            torch.save(model.state_dict(), 'checkpoint_r152_tiny.pth')
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {t_accuracy:.2f}%')
    
        model.eval()
        val_loss   = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in tqdm(testloader, desc='val'):
                images, labels = images.to(device), labels.to(device)

                #conv1 layer
                out = F.relu(model.bn1(model.conv1(images)))
                out = out.detach()                         
                
                # layer 1-4
                for block in model.layer1: out = block(out).detach()
                for block in model.layer2: out = block(out).detach()
                for block in model.layer3: out = block(out).detach()
                for block in model.layer4: out = block(out).detach()

                # fc layer
                out_pool = F.avg_pool2d(out, kernel_size=out.size()[2:])
                logits   = model.fc(out_pool.view(out_pool.size(0), -1))
                
                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = logits.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        accuracy   = 100.0 * correct / total
        best_val = max(best_val,accuracy)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        
        log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss}, Training Accuracy: {t_accuracy:.2f}%, Test Accuracy: {accuracy:.2f}%\n")
        log_file.write(f"Best Train Accuracy: {best_train:.2f}%, Best Validation Accuracy: {best_val:.2f}%\n\n")
        log_file.flush()
