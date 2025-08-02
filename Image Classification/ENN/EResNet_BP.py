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
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = ENNConv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = ENNConv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ENNConv2d(in_planes, planes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = ENNConv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = ENNConv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = ENNConv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                ENNConv2d(in_planes, planes * self.expansion, 1,
                          stride, 0, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetENN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = ENNConv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.fc = ENNLinear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return self.fc(out)

def orth_loss(model):
    loss = 0.0
    for m in model.modules():
        if isinstance(m, (ENNConv2d, ENNLinear)):
            loss = loss + m.ortho_loss()
    return loss

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
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()