import torch, torch.nn as nn, torch.optim as optim, torch.backends.cudnn as cudnn
import torchvision, torchvision.transforms as transforms
from tqdm import tqdm
import os, math

from EResNet_BP import ResNet101 as EResNet101        
from EResNet_BP import ENNLinear, ENNConv2d           
cudnn.benchmark = True

device      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 450
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
NUM_CLASSES = 1000

train_directory = '/project/pbogdan_1210/imagenet/train'
test_directory  = '/project/pbogdan_1210/imagenet/val'

class EarlyStopping:
    def __init__(self, patience=3, verbose=True, name="null"):
        self.patience = patience
        self.verbose  = verbose
        self.name     = name
        self.counter  = 0
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
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform_train)
testset  = torchvision.datasets.ImageFolder(root=test_directory,  transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers = 2, pin_memory=True)


model = EResNet101(num_classes=NUM_CLASSES)           

checkpoint = 'checkpoint_r101_image.pth'             
if os.path.exists(checkpoint):
    state = torch.load(checkpoint)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    print("Weight Loaded!")
else:
    print("No weight available!")
    
cudnn.benchmark = True

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
model.to(device)

criterion = nn.CrossEntropyLoss()

optim_backbone = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(optim_backbone, T_max=200)

orth_loss_weight = 5e-5

num_epochs     = 500
early_stopping = EarlyStopping(patience=10, verbose=True)
best_val  = 0.0
best_train = 0.0

log_file = open("training_log_r101_image.txt", "w")


for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i,(images, labels) in enumerate(tqdm(trainloader)):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        ortho = 0.0
        for m in model.modules():
            if isinstance(m, (ENNLinear, ENNConv2d)):
                ortho += m.orthonormalize()
        loss = loss + orth_loss_weight * ortho

        optim_backbone.zero_grad()
        loss.backward()
        optim_backbone.step()
        scheduler.step()

        _, pred = logits.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
        running_loss += loss.item()
        if i % 10 == 0:
            print(f'Saving model...')
            torch.save(model.state_dict(), checkpoint)
            # print(f'Epoch {epoch+1}: Batch[{i}/{len(trainloader)}] Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')


    train_loss = running_loss / len(trainloader.dataset)
    train_acc  = 100.0 * correct / total
    best_train = max(best_train, train_acc)

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='val'):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            _, pred = logits.max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()

    val_loss /= total
    val_acc   = 100.0 * correct / total
    best_val  = max(best_val, val_acc)

    print(f'Validation Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  Best: {best_val:.2f}%')

    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss}, Training Accuracy: {train_acc:.2f}%, Test Accuracy: {val_acc:.2f}%\n")
    log_file.write(f"Best Train Accuracy: {best_train:.2f}%, Best Validation Accuracy: {best_val:.2f}%\n\n")
    log_file.flush()

log_file.close()