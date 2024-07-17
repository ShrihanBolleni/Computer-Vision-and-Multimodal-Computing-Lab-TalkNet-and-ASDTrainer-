import sys
import torch
import time
from tqdm import tqdm
from models_util import *


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.args = kwargs
        print(kwargs)

        # move models to gpu if possible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.visualModel = createVisualModel().to(self.device)
        self.audioModel = createAudioModel().to(self.device)
        self.fusionModel = createFusionModel().to(self.device)
        self.fcModel = createFCModel().to(self.device)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=self.args['lrDecay'])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_prop(self, loader):
        for audioFeatures, visualFeatures, labels in loader:
            # move features to GPU
            audioFeatures = torch.unsqueeze(audioFeatures, dim=1).to(self.device)
            visualFeatures = visualFeatures.to(self.device)
            labels = labels.squeeze().to(self.device)

            # generate embeddings
            audioEmbed = self.audioModel(audioFeatures)
            visualEmbed = self.visualModel(visualFeatures)

            # audio visual fusion
            avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)

            # fc layer
            fcOutput = self.fcModel(avfusion)

            # detach since we don't need gradients
            audioFeatures.detach()
            visualFeatures.detach()
            avfusion.detach()
            fcOutput.detach()

            yield fcOutput, labels

    def train_step(self, loader, epoch):
        self.train()
        lr = self.optim.param_groups[0]['lr']
        index, num_correct, loss = 0, 0, 0

        for num, (fcOutput, labels) in enumerate(self.forward_prop(loader), start=1):
            # gradient descent
            batch_loss = self.loss_fn(fcOutput, labels)
            batch_loss.backward()

            # optimizer steps
            self.optim.step()
            self.optim.zero_grad()

            # calculate performance metrics
            loss += batch_loss.detach().cpu().numpy()
            num_correct += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()

            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / len(loader))) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / num, 100 * (num_correct / index)))
            sys.stderr.flush()
        sys.stdout.write("\n")

        # run scheduler step
        self.scheduler.step()

        # return loss and lr
        return loss / len(loader), lr
    
    def train_network(self, start_epoch, train_loader, val_loader):
        mAPs, losses = [], [] # holds mAP and losses for all epochs
        scoreFile = open(self.args['scoreSavePath'], "a+")
        bestmAP = 0
        for epoch in range(start_epoch, self.args['maxEpoch'] + 1):
            # get loss and learning rate for current epoch
            loss, lr = self.train_step(epoch=epoch, loader=train_loader)

            # check if it's time to evaluate on validation dataset
            if epoch % self.args['testInterval'] == 0:
                test_loss, test_mAP = self.evaluate_network(loader=val_loader)
                losses.append(test_loss)
                mAPs.append(test_mAP)
                
                # update best mAP and save model parameters
                if mAPs[-1] >= bestmAP:
                    bestmAP = mAPs[-1]
                    self.saveParameters(self.args['modelSavePath'] + "/best.model")
                
                # output information about performance on validation dataset
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%" % (epoch, mAPs[-1], max(mAPs)))
                scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n" % (epoch, lr, loss, mAPs[-1], max(mAPs)))
                scoreFile.flush()

        return mAPs, losses

    def evaluate_network(self, loader):
        self.eval()
        loss, num_correct = 0, 0

        with torch.no_grad():
            for fcOutput, labels in tqdm(self.forward_prop(loader)):
                # calculate performance metrics
                batch_loss = self.loss_fn(fcOutput, labels)
                loss += batch_loss.detach().cpu().numpy()
                num_correct += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()

        avg_loss = loss / len(loader)
        accuracy = num_correct / (len(loader) * self.args['batchSize'])
        print('eval loss ', avg_loss)
        print('eval accuracy ', accuracy)

        return avg_loss, accuracy

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
                    print("%s is not in the model." % origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
