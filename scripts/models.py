import torch.nn as nn
import torch
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
from transformers import ViTImageProcessor, ViTModel
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from mamba_ssm import Mamba
torch.manual_seed(1)


class DinoModel(nn.Module):
    def __init__(self,augmantations=None):
        super(DinoModel, self,).__init__()
        self.Dino = ViTModel.from_pretrained('facebook/dino-vits8')#.requires_grad_(False)
        self.augmantations = augmantations
        self.dim_reduction = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU()
        )
        
        # Reduce the sequence length to a fixed size
        self.pool = nn.AdaptiveAvgPool1d(100)
        
        # Fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(in_features=785 * 100, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.1))
        self.fc2 = nn.Sequential(nn.Linear(in_features=1024, out_features=10),
                                    nn.Softmax(dim=1))
        self.aug_list = AugmentationSequential(
                    K.Resize(size=(224, 224)),  # Resize to 224x224
                    )
    def forward(self, inp,mode='test'):
        ## Padd input 
        padded = torch.cat([inp,inp,inp],dim=1)
        padded = self.aug_list(padded)

        ## Take prediction
        outputs = self.Dino(padded)
        last_hidden_states = outputs.last_hidden_state

        # Reduce feature dimension
        x = self.dim_reduction(last_hidden_states)  # Shape: [batch, reduced_feature_dim, sequence_length]
        
        # Apply pooling
        x = self.pool(x)  # Shape: [batch, reduced_feature_dim, pooled_sequence_length]

        # Flatten
        x = x.view(x.size(0), -1)  # Shape: [batch, reduced_feature_dim * pooled_sequence_length]
  
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Shape: [batch, num_classes]
        
        return x

## Special augmantation
import random
class CircularShiftTransform:
    def __init__(self, max_shift, axis, probability=0.5):
        self.max_shift = max_shift
        self.axis = axis
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            shift = random.randint(0, self.max_shift)
            return torch.roll(image, shifts=shift, dims=self.axis)
        return image
    
class MambaVisionModel(nn.Module):
    def __init__(self,augmantations=None):
        super(MambaVisionModel, self,).__init__()
        self.MambaVision = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)#.requires_grad_(False)
        self.augmantations = augmantations
        self.SpecialAug = CircularShiftTransform(max_shift=120, axis=2, probability=0.9)
        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=20480, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
        
        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=10240, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))


        self.fcBlock3 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
        #self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=10),
        #                            nn.Softmax(dim=1))
        self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=10))

    def forward(self, inp,mode='test'):
        ## Padd input 
        padded = torch.cat([inp,inp,inp],dim=1)

        ## Augmantations 
        if self.augmantations is not None and mode == 'train': # Augmantations
            padded = self.augmantations(padded)
            padded = self.SpecialAug(padded)

        ## Take prediction
        out_avg_pool1, features = self.MambaVision(padded)

        ## FC layers     
        out1 = self.fcBlock1(features[2].reshape(padded.shape[0],20480))
        out2 = self.fcBlock2(features[3].reshape(padded.shape[0],10240))

        ## Merge path
        out = torch.cat((out1,out2),dim=1)

        ## Final layers
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

    def forward(self, inp,mode='test'):
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



