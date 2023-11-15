import torch
import torch.nn as nn
import torch.optim as optim
from util.import_tqdm import tqdm
from singleton_pattern import load_config
class PhysNet(torch.nn.Module):
    def __init__(self):
        super(PhysNet, self).__init__()
        config = load_config.get_config()
        data_format = config['data_format']
        self.num_epochs = config.get('num_epochs',10)
        slice_interval = data_format['slice_interval']
        self.physnet = torch.nn.Sequential(
            EncoderBlock(),
            decoder_block(),
             # torch.nn.AdaptiveMaxPool3d((slice_interval, 1, 1)),  # spatial adaptive pooling
            torch.nn.AdaptiveAvgPool3d((slice_interval, 1, 1)),  # spatial adaptive pooling
            torch.nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        return self.physnet(x).view(-1, length)
    def train_model(self,dataloader):
        print('start training...')
        self.train()

        criterion = nn.MSELoss()

        optimizer = optim.SGD(self.parameters(), lr=0.01)

        progress_bar = tqdm(range(self.num_epochs), desc="Progress")
        for epoch in progress_bar:
            loss = None
            for batch_X, batch_y in dataloader:
                outputs = self(batch_X.float())
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        self.eval()
        print('train end.')

class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        #, in_channel, out_channel, kernel_size, stride, padding
        self.encoder_block = torch.nn.Sequential(
            ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # Temporal Halve
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # Temporal Halve
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

    def forward(self, x):
        return self.encoder_block(x)

class decoder_block(torch.nn.Module):
    def __init__(self):
        super(decoder_block, self).__init__()
        self.decoder_block = torch.nn.Sequential(
            DeConvBlock3D(64, 64, [4, 1, 1], [2, 1, 1], [1, 0, 0]),
            DeConvBlock3D(64, 64, [4, 1, 1], [2, 1, 1], [1, 0, 0])
        )

    def forward(self, x):
        return self.decoder_block(x)

class DeConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        self.deconv_block_3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ELU()
        )

    def forward(self, x):
        return self.deconv_block_3d(x)

class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_3d(x)
