import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,last_block=False,stride=(1,1),padding=(1,1)):
        super(ConvBlock,self).__init__()
        self.last_block=last_block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if not self.last_block:
            self.relu3 = nn.ReLU()
            self.bn3 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.conv3(x)
        if self.last_block:
            return x
        else:
            x = self.bn3(self.relu3(x))
            return x

class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, device='gpu', **kwargs):
        super(model, self).__init__()

        self.visualModel = None
        self.audioModel = None
        self.fusionModel = None
        self.fcModel = None

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.createVisualModel()
        self.createAudioModel()
        self.createFusionModel()
        self.createFCModel()
        
        self.visualModel = self.visualModel.to(self.device)
        self.audioModel = self.audioModel.to(self.device)
        self.fcModel = self.fcModel.to(self.device)
        
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def createVisualModel(self):
        self.visualModel = nn.Sequential(
            ConvBlock(1, 32, 3),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, 3),
            nn.MaxPool2d(2),
            ConvBlock(64, 64, 3),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, 3, last_block=True),
            nn.Flatten()
        )

    def createAudioModel(self):
        self.audioModel = nn.Sequential(
            ConvBlock(1, 32, 3),
            nn.MaxPool2d(2,(2, 1)),
            ConvBlock(32, 64, 3),
            nn.MaxPool2d(2,(2, 1)),
            ConvBlock(64, 64, 3),
            nn.MaxPool2d(2,(2, 1)),
            ConvBlock(64, 64, 3),
            nn.MaxPool2d(2),
            ConvBlock(64, 64, 3, last_block=True),
            nn.Flatten()
        )


    def createFusionModel(self):
        pass

    def createFCModel(self):
        self.fcModel = nn.Sequential(
            nn.Linear(30848, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,2)
        )
    
    def train_network(self, loader, epoch, **kwargs):
        
        self.train()
        self.scheduler.step(epoch-1)
        lr = self.optim.param_groups[0]['lr']
        index, top1, loss = 0, 0, 0
        for num, (audioFeatures, visualFeatures, labels) in enumerate(loader, start=1):
                self.zero_grad()

                # print('audioFeatures shape: ', audioFeatures.shape)
                # print('visualFeatures shape: ', visualFeatures.shape)
                # print('labels shape: ', labels.shape)
                
                # print('audio feature: ', audioFeatures)
                # print('visual features: ', visualFeatures)

                audioFeatures = torch.unsqueeze(audioFeatures, dim=1)  
                # print('audioFeatures after unsqueeze: ', audioFeatures.shape)            
                
                audioFeatures = audioFeatures.to(self.device)
                visualFeatures = visualFeatures.to(self.device)
                labels = labels.squeeze().to(self.device)
                                
                audioEmbed = self.audioModel(audioFeatures)
                # print('audio embed shape: ', audioEmbed.shape)
                visualEmbed = self.visualModel(visualFeatures)
                # print('visual embed shape: ', visualEmbed.shape)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
                # print('avfusion shape: ', avfusion.shape)
                
                fcOutput = self.fcModel(avfusion)
                # print('fc output shape: ', fcOutput.shape)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                self.optim.zero_grad()
                nloss.backward()
                self.optim.step()
                
                loss += nloss.detach().cpu().numpy()
                
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
                " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
                sys.stderr.flush()  
        sys.stdout.write("\n")
        
        return loss/num, lr
        
    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []
        
        loss, top1, index, numBatches = 0, 0, 0, 0
        
        for audioFeatures, visualFeatures, labels in tqdm.tqdm(loader):
            
            audioFeatures = torch.unsqueeze(audioFeatures, dim=1)
            audioFeatures = audioFeatures.to(self.device)
            visualFeatures = visualFeatures.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            with torch.no_grad():
                
                audioEmbed = self.audioModel(audioFeatures)
                visualEmbed = self.visualModel(visualFeatures)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
                
                fcOutput = self.fcModel(avfusion)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                loss += nloss.detach().cpu().numpy()
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                numBatches += 1
                
        print('eval loss ', loss/numBatches)
        print('eval accuracy ', top1/index)
        
        return top1/index

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)
        
    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)