"""
Updated Version of ResNet with ENN integrated
The EResNet utilzed the eigenvectors to represent normal weights in CNN. i.e. in conv layer and linear layer, we use W = Q Λ P^T to 
do weight update and local loss update. 
This is the architecture that fully removed BP, meaning we do not need backward in the whole model.
Detailed ENN could be found at ENN.py, which only contains eigenvector layers and activation layers. 
With no conv and mlp, we could still achieve a compatitable result in small datasets such as MNIST and CIFAR10/

update log: 
1. local head hyper parameters added - now support different image channel, height and width
2. updated test case with new features, including initialize the local heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ENNLinear(nn.Module):
    """
    Linear layer with ENN parameterization of weights (Q Λ P^T)
    the weight of linear layer initialize with kaiming norm, then represents with W = Q Λ P^T
    The output is the W
    Each iteration compute local loss for Q and P (i.e. the differetial of I vector of P and Q with new P and Q)
    This ensures the weight update is local, no BP in this module
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ENNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        r = min(in_features, out_features)

        W_init = torch.empty(out_features, in_features)
        nn.init.kaiming_normal_(W_init, mode='fan_out') 
        U, S, Vh = torch.linalg.svd(W_init.cpu(), full_matrices=False)
        # Take the first r columns for U and V^T (economy SVD)
        U = U[:, :r]   
        V = Vh.T[:, :r]  
        S = S[:r]   
        
        self.Q = nn.Parameter(U.to(W_init.device))            
        self.Lambda = nn.Parameter(S.to(W_init.device))      
        self.P = nn.Parameter(V.to(W_init.device))            
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.matmul(self.P)             
        out = out * self.Lambda            
        out = out.matmul(self.Q.T)        
        if self.bias is not None:
            out = out + self.bias
        return out

    def orthonormalize(self):
        I_Q = torch.eye(self.Q.shape[1], device=self.Q.device)
        I_P = torch.eye(self.P.shape[1], device=self.P.device)
        # Frobenius norm squared of (Q^T Q - I) and (P^T P - I)
        loss_Q = torch.norm(self.Q.T @ self.Q - I_Q, p='fro')**2
        loss_P = torch.norm(self.P.T @ self.P - I_P, p='fro')**2
        return loss_Q + loss_P

class ENNConv2d(nn.Module):
    """
    2D Convolution layer with ENN parameterization of weights (Q Λ P^T)
    the weight of conv layer initialize with kaiming norm, then represents with W = Q Λ P^T
    The new representation of weight then pass to nn.conv2d for the weight update
    Each iteration compute local loss for Q and P (i.e. the differetial of I vector of P and Q with new P and Q)
    This ensures the weight update is local, no BP in this module
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False):
        super(ENNConv2d, self).__init__()
        if groups != 1:
            raise NotImplementedError("ENNConv2d currently only supports groups=1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        
        self.kernel_size = (kh, kw)    
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
    
        in_dim = in_channels * kh * kw
        out_dim = out_channels
        r = min(in_dim, out_dim)

        W_init = torch.empty(out_dim, in_dim)
        nn.init.kaiming_normal_(W_init, mode='fan_out')
        U, S, Vh = torch.linalg.svd(W_init.cpu(), full_matrices=False)
        U = U[:, :r]   
        V = Vh.T[:, :r] 
        S = S[:r]
        
        self.Q = nn.Parameter(U.to(W_init.device))           
        self.Lambda = nn.Parameter(S.to(W_init.device))     
        self.P = nn.Parameter(V.to(W_init.device))           
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.Q_prev = None
        
    def forward(self, x):
        weight_flat = self.Q * self.Lambda    
        weight_flat = weight_flat @ self.P.T  
        weight = weight_flat.view(self.out_channels, self.in_channels, *self.kernel_size)
        return F.conv2d(x, weight, self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


    def orthonormalize(self):
        I_Q = torch.eye(self.Q.shape[1], device=self.Q.device)
        I_P = torch.eye(self.P.shape[1], device=self.P.device)
        loss_Q = torch.norm(self.Q.T @ self.Q - I_Q, p='fro')**2
        loss_P = torch.norm(self.P.T @ self.P - I_P, p='fro')**2
        return loss_Q + loss_P

    
class BasicBlock(nn.Module):
    """
    Residual block for ResNet-18/34 
    conv2d and linear replaced with ENN style integrated
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ENNConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = ENNConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
 
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:

            self.downsample = nn.Sequential(
                ENNConv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
    # the model learns locally. the output comes from the local head, then compute the CE loss and orth loss (i.e. P loss and Q loss) for each layer. 
    # The weight updated locally with no BP.
    def train_step(self, x: torch.Tensor, target: torch.Tensor, criterion, optimizer, orth_loss_weight=1e-3) -> torch.Tensor:
        out = self.forward(x)
        out_flat = out.view(out.size(0), -1)
        logits = self.local_head(out_flat)
        loss_cls = criterion(logits, target)
        loss_ortho = 0.0
        for module in self.modules():
            if isinstance(module, (ENNConv2d, ENNLinear)):
                loss_ortho = loss_ortho + module.orthonormalize()
        loss = loss_cls + orth_loss_weight * loss_ortho
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return out.detach() 


class Bottleneck(nn.Module):
    """
    Residual block for ResNet-50/101/152 
    conv2d and linear replaced with ENN style integrated
    """
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ENNConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = ENNConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = ENNConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion * planes)
       
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ENNConv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
        self.local_head = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
    def train_step(self, x: torch.Tensor, target: torch.Tensor, criterion, optimizer, orth_loss_weight=1e-3) -> torch.Tensor:
        out = self.forward(x)
        out_flat = out.view(out.size(0), -1)
        logits = self.local_head(out_flat)
        loss_cls = criterion(logits, target)
        loss_ortho = 0.0
        for module in self.modules():
            if isinstance(module, (ENNConv2d, ENNLinear)):
                loss_ortho = loss_ortho + module.orthonormalize()
        loss = loss_cls + orth_loss_weight * loss_ortho
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return out.detach()


class ResNetENN(nn.Module):
    """
    Same as most widely used ResNet with following modifications:
    1. replace conv2d with ENNConv2d, 
    2. replace liner with ENNLiner, 
    3. add local head variables.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetENN, self).__init__()
        self.in_planes = 64

        self.conv1 = ENNConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc = ENNLinear(512 * block.expansion, num_classes)

        self.local_heads_created = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor):
        assert self.local_heads_created, "Local heads not initialized. Call assign_local_heads() first."
        local_logits = []
        out = F.relu(self.bn1(self.conv1(x)))

        out_flat = out.view(out.size(0), -1)
        local_logits.append(self.local_heads[0](out_flat))
      
        out = out.detach()

        head_index = 1  
        for block in self.layer1:
            out = block(out)
            out_flat = out.view(out.size(0), -1)
            local_logits.append(block.local_head(out_flat))
            head_index += 1
            out = out.detach()
            
        for block in self.layer2:
            out = block(out)
            out_flat = out.view(out.size(0), -1)
            local_logits.append(block.local_head(out_flat))
            head_index += 1
            out = out.detach()
            
        for block in self.layer3:
            out = block(out)
            out_flat = out.view(out.size(0), -1)
            local_logits.append(block.local_head(out_flat))
            head_index += 1
            out = out.detach()
            
        for block in self.layer4:
            out = block(out)
            out_flat = out.view(out.size(0), -1)
            local_logits.append(block.local_head(out_flat))
            head_index += 1
            out = out.detach()
            
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])  
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, local_logits
    
    def assign_local_heads(self, num_classes: int, C: int, H: int,W: int):
        device = next(self.parameters()).device  
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W, device=device)  
            out = F.relu(self.bn1(self.conv1(dummy)))
            feature_dim = out.view(1, -1).size(1)
            self.conv1_head = nn.Linear(feature_dim, num_classes).to(device)
            self.conv1_head.weight.requires_grad = False
            if self.conv1_head.bias is not None:
                self.conv1_head.bias.requires_grad = False

            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    out = block(out)  
                    feature_dim = out.view(1, -1).size(1)
                    head = nn.Linear(feature_dim, num_classes).to(device)
                    head.weight.requires_grad = False
                    if head.bias is not None:
                        head.bias.requires_grad = False
                    block.local_head = head

        self.train(was_training)
        self.local_heads_created = True
    
def ResNet18(num_classes=10):
    return ResNetENN(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNetENN(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNetENN(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNetENN(Bottleneck, [3, 4, 23, 3],num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNetENN(Bottleneck, [3, 8, 36, 3],num_classes=num_classes)


def test():
    net = ResNet18()
    net.assign_local_heads(num_classes=100, C=1, H=32, W=32)
    y,_ = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()