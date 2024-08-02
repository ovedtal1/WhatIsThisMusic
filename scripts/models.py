import torch.nn as nn
import torch
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
from mamba_ssm import Mamba
#from xlstm import (
#    xLSTMBlockStack,
#    xLSTMBlockStackConfig,
#    mLSTMBlockConfig,
#    mLSTMLayerConfig,
#    sLSTMBlockConfig,
#    sLSTMLayerConfig,
#    FeedForwardConfig,
#)
torch.manual_seed(1)


class MambaVisionModel(nn.Module):
    def __init__(self):
        super(MambaVisionModel, self).__init__()
        self.MambaVision = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)#.requires_grad_(False)
        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=20480, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
        
        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=10240, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock0 = nn.Sequential(nn.Linear(in_features=40960, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
        self.fcBlock3 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
        self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=10),
                                    nn.Softmax(dim=1))
        self.MambaSeq = Mamba(d_model=7680, d_state=64,  d_conv=4, expand=2).to("cuda") 

    def forward(self, inp):
        #print(f'inp shape: {inp.shape}')
        ## Chunk division
        #print(f'inp out: {inp.shape}')
        """
        chunk1 = inp[:,:,0:90,:]
        chunk2 = inp[:,:,70:160,:]
        chunk3 = inp[:,:,145:235,:]
        chunk4 = inp[:,:,220:310,:]
        chunk5 = inp[:,:,295:385,:]
        chunk6 = inp[:,:,363:448,:]
        """
        #print(f'chunk out: {chunk1.shape}')
        ## padd input 
        padded = torch.cat([inp,inp,inp],dim=1)
        """
        padded1 = torch.cat([chunk2,chunk1,chunk1],dim=1)
        padded2 = torch.cat([chunk2,chunk2,chunk2],dim=1)
        padded3 = torch.cat([chunk3,chunk3,chunk3],dim=1)
        padded4 = torch.cat([chunk4,chunk4,chunk4],dim=1)
        padded5 = torch.cat([chunk5,chunk5,chunk5],dim=1)
        padded6 = torch.cat([chunk6,chunk6,chunk6],dim=1)
        #print(f'padded shape: {padded.shape}')
        """
        """
        out_avg_pool1, features1 = self.MambaVision(padded1)
        out_avg_pool2, features2 = self.MambaVision(padded2)
        out_avg_pool3, features3 = self.MambaVision(padded3)
        out_avg_pool4, features4 = self.MambaVision(padded4)
        out_avg_pool5, features5 = self.MambaVision(padded5)
        out_avg_pool6, features6 = self.MambaVision(padded6)
        
        #print(f'fetures out: {features1[3].shape}')
        fetures_flat1 = features1[3].reshape(chunk1.shape[0],1,7680)
        fetures_flat2 = features2[3].reshape(chunk2.shape[0],1,7680)
        fetures_flat3 = features3[3].reshape(chunk3.shape[0],1,7680)
        fetures_flat4 = features4[3].reshape(chunk4.shape[0],1,7680)
        fetures_flat5 = features5[3].reshape(chunk5.shape[0],1,7680)
        fetures_flat6 = features6[3].reshape(chunk6.shape[0],1,7680)
        #fetures_flat3 = features[3].reshape(inp.shape[0],10240)
        
        #print(f'features[3] shape: {features[3].shape}')
        #print(f'features[2] shape: {features[2].shape}')
        #print(f'features[1] shape: {features[1].shape}')
        ## Concat 
        concat = torch.cat((fetures_flat1,fetures_flat2,fetures_flat3,fetures_flat4,fetures_flat5,fetures_flat6),dim=1)
        outMamba = self.MambaSeq(concat)
        """
        #print(f'mamba in: {concat.shape}')
        #concat = concat.permute(0,2,1)
        #print(f'mamba out: {outMamba.shape}')
        # take prediction
        out_avg_pool1, features = self.MambaVision(padded)
        #outMamba = self.MambaSeq(features1[3].reshape(chunk1.shape[0],1,7680*6))
        #outMamba = outMamba.reshape(outMamba.shape[0],7680)
        #out3 = self.fcBlock1(fetures_flat3)
        #out = torch.cat((out2,out3),dim=1)  
        #print(features[2].shape)      
        #print(features[3].shape) 
        #out0 = self.fcBlock0(features[1].reshape(padded.shape[0],40960))     
        out1 = self.fcBlock1(features[2].reshape(padded.shape[0],20480))
        out2 = self.fcBlock2(features[3].reshape(padded.shape[0],10240))
        out = torch.cat((out1,out2),dim=1) 
        out = self.fcBlock3(out)
        out = self.output(out)
        return out


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov4.weight)
        self.convBlock4 = nn.Sequential(cov4,
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        #print(inp.shape)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.convBlock4(out)

        out = out.view(out.size()[0], -1)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out


class CrnnModel(nn.Module):
    def __init__(self):
        super(CrnnModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.GruLayer = nn.GRU(input_size=2048,
                               hidden_size=256,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(2048),
                                       nn.Dropout(0.6))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=2048, out_features=512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, time, freq)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        # [16, 256, 8, 8]

        out = out.contiguous().view(out.size()[0], out.size()[2], -1)
        out, _ = self.GruLayer(out)
        out = out.contiguous().view(out.size()[0],  -1)
        # out_features=4096

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out


class CrnnLongModel(nn.Module):
    def __init__(self):
        super(CrnnLongModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov4.weight)
        self.convBlock4 = nn.Sequential(cov4,
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.GruLayer = nn.GRU(input_size=2048,
                               hidden_size=256,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(1024),
                                       nn.Dropout(0.5))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, time, freq)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.convBlock4(out)
        # [16, 256, 16, 16]

        out = out.contiguous().view(out.size()[0], out.size()[2], -1)
        out, _ = self.GruLayer(out)
        out = out.contiguous().view(out.size()[0], -1)

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)

        out = self.output(out)

        return out


class RnnModel(nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()

        self.GruLayer = nn.GRU(input_size=256,
                               hidden_size=1024,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=False)

        cov1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov4.weight)
        self.convBlock4 = nn.Sequential(cov4,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=11520, out_features=4096),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=4096, out_features=2048),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock3 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock4 = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, 1, time, freq)
        inp = inp.contiguous().view(inp.size()[0], inp.size()[2], -1)
        out, _ = self.GruLayer(inp)
        out = out.contiguous().view(out.size()[0], 1, out.size()[1], out.size()[2])
        out = self.convBlock1(out)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.convBlock4(out)
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.fcBlock3(out)
        out = self.fcBlock4(out)
        out = self.output(out)
        return out


if __name__ == '__main__':
    TestModel = CnnModel()
    from Paras import Para
    Para.batch_size = 32
    from data_loader import torch_dataset_loader
    test_loader = torch_dataset_loader(Para.LA_TEST_DATA_PATH, Para.batch_size, False, Para.kwargs)

    for index, data in enumerate(test_loader):
        spec_input, target = data['mel'], data['tag']

        TestModel.eval()
        predicted = TestModel(spec_input)
        break



