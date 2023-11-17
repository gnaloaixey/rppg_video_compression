import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
from util.cache import Cache
from util.import_tqdm import tqdm




class LSTCrPPG(torch.nn.Module):
    def __init__(self, frames=32):
        super(LSTCrPPG, self).__init__()
        self.encoder_block = EncoderBlock()
        self.decoder_block = DecoderBlock()

    def forward(self, x):
        e = self.encoder_block(x)
        out = self.decoder_block(e)
        return out.squeeze()
    def train_model(self,dataloader,num_epochs = None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        print('start training...')
        self.train()
        cache = Cache('model')
        criterion = LSTCrPPGLoss()

        optimizer = optim.SGD(self.parameters(), lr=0.01)

        progress_bar = tqdm(range(num_epochs), desc="Progress")
        for epoch in progress_bar:
            loss = None
            for batch_X, batch_y in dataloader:
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        self.eval()
        cache.save_model(self)
        print('train end.')
class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        #, in_channel, out_channel, kernel_size, stride, padding
        self.encoder_block1 = torch.nn.Sequential(
            ConvBlock3D(3, 16, [3,3,3], [1,1,1], [1,1,1]),
            ConvBlock3D(16, 16, [3,3,3], [1,1,1], [1,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.encoder_block2 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.encoder_block3 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.encoder_block4 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.encoder_block5 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.encoder_block6 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.encoder_block7 = torch.nn.Sequential(
            ConvBlock3D(64, 64, [5, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(64)
        )

    def forward(self, x):
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(e1)
        e3 = self.encoder_block3(e2)
        e4 = self.encoder_block4(e3)
        e5 = self.encoder_block5(e4)
        e6 = self.encoder_block6(e5)
        e7 = self.encoder_block7(e6)
        return [e7,e6,e5,e4,e3,e2,e1]

class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.decoder_block6_transpose = torch.nn.ConvTranspose3d(64,64,[5,1,1],[1,1,1])
        self.decoder_block6 = torch.nn.Sequential(
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.decoder_block5_transpose =torch.nn.ConvTranspose3d(64, 64, [4, 1, 1],[2,1,1])
        self.decoder_block5 = torch.nn.Sequential(
            ConvBlock3D(64, 32, [3, 3, 3], [1, 1, 1],[1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1],[0,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.decoder_block4_transpose = torch.nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block4 = torch.nn.Sequential(
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.decoder_block3_transpose = torch.nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block3 = torch.nn.Sequential(
            ConvBlock3D(32, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.decoder_block2_transpose = torch.nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block2 = torch.nn.Sequential(
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.decoder_block1_transpose = torch.nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block1 = torch.nn.Sequential(
            ConvBlock3D(16, 3, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(3, 3, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(3)
        )
        self.predictor = torch.nn.Conv3d(3, 1 ,[1,4,4])



    def forward(self, encoded_features):
        encoded_feature_0,encoded_feature_1,encoded_feature_2,encoded_feature_3,\
            encoded_feature_4,encoded_feature_5,encoded_feature_6 = encoded_features

        d = self.decoder_block6_transpose(encoded_feature_0)
        d6 = self.decoder_block6(self.TARM(encoded_feature_1, d))
        print(d6.shape)
        d5 = self.decoder_block5(self.TARM(encoded_feature_2,self.decoder_block5_transpose(d6)))
        d4 = self.decoder_block4(self.TARM(encoded_feature_3,self.decoder_block4_transpose(d5)))
        d3 = self.decoder_block3(self.TARM(encoded_feature_4,self.decoder_block3_transpose(d4)))
        d2 = self.decoder_block2(self.TARM(encoded_feature_5,self.decoder_block2_transpose(d3)))
        d1 = self.decoder_block1(self.TARM(encoded_feature_6,self.decoder_block1_transpose(d2)))
        predictor = self.predictor(d1)
        return predictor

    def TARM(self, e,d):
        target = d
        shape = d.shape
        e = torch.nn.functional.adaptive_avg_pool3d(e,d.shape[2:])
        e = e.view(e.shape[0],e.shape[1], shape[2],-1)
        d = d.view(d.shape[0], shape[1], shape[2], -1)
        temporal_attention_map = e @ torch.transpose(d,3,2)
        temporal_attention_map = torch.nn.functional.softmax(temporal_attention_map,dim=-1)
        refined_map = temporal_attention_map@e
        out = refined_map #( 1 + torch.reshape(refined_map,shape)) * target
        return out

class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.ELU()
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class LSTCrPPGLoss(nn.Module):
    def __init__(self):
        super(LSTCrPPGLoss, self).__init__()
        self.timeLoss = nn.MSELoss()
        self.lambda_value = 0.2
        self.alpha = 1.0
        self.beta = 0.5

    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)

        # predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
        # targets = (targets - torch.mean(targets)) / torch.std(targets)

        targets = torch.nn.functional.normalize(targets, dim=1)
        predictions = torch.nn.functional.normalize(predictions, dim=1)

        l_time = self.timeLoss(predictions, targets)
        l_frequency = self.frequencyLoss(predictions, targets)
        return self.alpha * l_time + self.beta * l_frequency

    def frequencyLoss(self, predictions, target):
        batch, n = predictions.shape
        predictions = self.calculate_rppg_psd(predictions)
        target = self.calculate_rppg_psd(target)
        di = torch.log(predictions) - torch.log(target)
        sum_di_squared = torch.sum(di ** 2, dim=-1)
        sum_di = torch.sum(di, dim=-1)

        hybrid_loss = (1 / n) * sum_di_squared - (self.lambda_value / (n ** 2)) * sum_di ** 2
        loss = torch.sum(hybrid_loss) / batch
        return loss

    def calculate_rppg_psd(self, rppg_signal):
        spectrum = fft.fft(rppg_signal)
        # 使用复乘法计算PSD
        psd = torch.abs(spectrum) ** 2

        return psd


