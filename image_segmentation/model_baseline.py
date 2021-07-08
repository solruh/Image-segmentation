import torch
import torch.nn as nn
import torch.nn.functional as F
value=0
class segnet(nn.Module):
    def __init__(self, out_channel=10):
        super(segnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, out_channel, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out
        
class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, bias =False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),            
            nn.Conv2d(features, features, 3, 1, 1, bias = False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),          
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, bias =False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True), 

            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1, bias =False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True), 
            )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x_residual = x
        x = self.conv2(x) + x_residual
        x = self.last(x)
        return x


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        self.m =False
        self.dropout =0
        if dropout:
            value=1
            self.dropout = nn.Dropout(0.5)
            self.m = True
        self.downsample = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x_residual = x
        x = self.conv2(x) + x_residual
        if(value== 1):
            self.dropout(x)
        x = self.downsample(x)
        return x

class UNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256, dropout=True)
        self.dec4 = UNetDec(256, 512,dropout =True)
        #self.dec5 = UNetDec(512,1024, dropout = True)
        self.center = nn.Sequential(
            nn.Conv2d(512,1024, 3, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        #self.enc5 =UNetEnc(2048,1024,512)
        self.enc4 = UNetEnc(1024,512,256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        #dec5 = self.dec5(dec4)
        center = self.center(dec4)
        #enc5 = self.enc5(torch.cat([
          #  center, F.upsample_bilinear(dec5, center.size()[2:])], 1))
        enc4 = self.enc4(torch.cat([
            center, F.upsample_bilinear(dec4, center.size()[2:])], 1))
        enc3 = self.enc3(torch.cat([
            enc4, F.upsample_bilinear(dec3, enc4.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = segnet()
    output = model(batch)
    print(output.size())