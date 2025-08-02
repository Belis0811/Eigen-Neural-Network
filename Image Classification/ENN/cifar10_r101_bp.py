import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from EResNet_BP import ResNet101 as EResNetBP


cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE       = 400
NUM_CLASSES      = 10
BASE_LR          = 1e-4   
MAX_LR_FACTOR    = 5     
NUM_EPOCHS       = 1000
WEIGHT_DECAY     = 1e-2
ORTH_LOSS_WEIGHT = 5e-5   

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = EResNetBP(num_classes=NUM_CLASSES).to(DEVICE)

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.AdamW(model.parameters(), lr=BASE_LR,
                         weight_decay=WEIGHT_DECAY)

total_steps = NUM_EPOCHS * len(trainloader)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

try:
    model.load_state_dict(torch.load('checkpoint_r101_cifar10_bp.pth'))
    print("Loaded checkpoint_r101_bp.pth – continuing training …")
except FileNotFoundError:
    print("No existing checkpoint found – training from scratch …")

log_file = open("training_log_renn_r101_bp.txt", "w")
best_train_acc = 0.0
best_val_acc   = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(trainloader, desc=f"epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss   = criterion(logits, labels)

        if ORTH_LOSS_WEIGHT > 0:
            orth_loss = 0.0
            for module in model.modules():
                if hasattr(module, 'orthonormalize'):
                    orth_loss += module.orthonormalize()
            loss += ORTH_LOSS_WEIGHT * orth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() 
        _, preds = logits.max(1)
        total     += labels.size(0)
        correct   += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    train_acc  = 100.0 * correct / total
    best_train_acc = max(best_train_acc, train_acc)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss   = criterion(logits, labels)
            val_loss += loss.item()

            _, preds = logits.max(1)
            total   += labels.size(0)
            correct += preds.eq(labels).sum().item()

    val_loss /= total
    val_acc  = 100.0 * correct / total
    best_val_acc = max(best_val_acc, val_acc)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {epoch_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), 'checkpoint_r101_cifar10_bp.pth')
        print("Checkpoint saved → checkpoint_r101_bp.pth")
    
    log_file.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss}, Training Accuracy: {train_acc:.2f}%, Test Accuracy: {val_acc:.2f}%\n")
    log_file.write(f"Best Train Accuracy: {best_train_acc:.2f}%, Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
    log_file.flush()

print(f"Best Train Acc: {best_train_acc:.2f}% | Best Val Acc: {best_val_acc:.2f}%")
