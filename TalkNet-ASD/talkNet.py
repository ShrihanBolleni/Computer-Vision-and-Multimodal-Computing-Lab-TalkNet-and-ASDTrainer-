import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas
from tqdm import tqdm
from loss import lossAV, lossA, lossV
from model.talkNetModel import TalkNetModel

class TalkNet(nn.Module):
    def __init__(self, **kwargs):
        super(TalkNet, self).__init__()

        # move models to gpu if possible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using {self.device}")

        self.args = kwargs
        self.model = TalkNetModel().to(self.device)
        self.lossAV = lossAV().to(self.device)
        self.lossA = lossA().to(self.device)
        self.lossV = lossV().to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=self.args['lrDecay'])
        
        #self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=0.0001, steps_per_epoch=len(data_loader), epochs=25)

        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        for num, (audioFeature, visualFeature, context_speaker_features, labels) in enumerate(loader, start=1):
            self.zero_grad()

            # generate visual embeddings for target speaker and context speakers
            combined_visual_features = torch.concat((visualFeature[0], context_speaker_features[0]))
            visualEmbeds = self.model.forward_visual_frontend(combined_visual_features.to(self.device))

            # split into target speaker visual embeddings and context speakers visual embeddings
            target_speaker = visualEmbeds[0].unsqueeze(0)
            context_speakers = visualEmbeds[1:]
            
            # "zero pad" when there are less than 3 context speakers
            # positions where the speakers should be are filled by zeros
            if len(context_speakers) < 3:
                needed = 3 - len(context_speakers)
                num_frames = target_speaker.shape[1]
                context_speakers = torch.concat((context_speakers, torch.zeros((needed, num_frames, 128)).to(self.device)))

            # generate audio embeddigns for target speaker
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(self.device))

            # cross attention for target speaker and each context speakers
            # queries come from target speaker, keys and values from each context speaker
            target_speaker = self.model.forward_context(target_speaker, context_speakers)
            
            # cross attention for audio and visual embeddings
            audioEmbed, target_speaker = self.model.forward_cross_attention(audioEmbed, target_speaker)
            
            # audio-visual self-attention
            outsAV = self.model.forward_audio_visual_backend(audioEmbed, target_speaker)

            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(target_speaker)
            
            # reshape labels
            labels = labels[0].reshape((-1)).to(self.device)
            
            # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)

            nloss = nlossAV + 0.4 * nlossA.detach() + 0.4 * nlossV.detach()
            nloss.backward()
            self.optim.step()

            loss += nloss.detach().cpu().numpy()
            top1 += prec
            index += len(labels)
            
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig):
        self.eval()
        predScores = []
        for (audioFeature, visualFeature, context_speaker_features, labels) in tqdm(loader):
            with torch.no_grad():
                # generate visual embeddings for target speaker and context speakers
                combined_visual_features = torch.concat((visualFeature[0], context_speaker_features[0]))
                visualEmbeds = self.model.forward_visual_frontend(combined_visual_features.to(self.device))

                # split into target speaker visual embeddings and context speakers visual embeddings
                target_speaker = visualEmbeds[0].unsqueeze(0)
                context_speakers = visualEmbeds[1:]
                
                # "zero pad" when there are less than 3 context speakers
                # positions where the speakers should be are filled by zeros
                if len(context_speakers) < 3:
                    needed = 3 - len(context_speakers)
                    num_frames = target_speaker.shape[1]
                    context_speakers = torch.concat((context_speakers, torch.zeros((needed, num_frames, 128)).to(self.device)))

                # generate audio embeddigns for target speaker
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(self.device))

                # cross attention for target speaker and each context speakers
                # queries come from target speaker, keys and values from each context speaker
                context_features = self.model.forward_context(target_speaker.repeat(3, 1, 1), context_speakers)
                context_features = context_features.sum(dim=0)
                
                # cross attention for audio and visual embeddings
                audioEmbed, visualEmbeds = self.model.forward_cross_attention(audioEmbed, target_speaker)
                
                # audio-visual self-attention
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, target_speaker)
                idkwhattocallthis = torch.concat((context_features, outsAV), dim=1)


                # reshape labels
                labels = labels[0].reshape((-1)).to(self.device)

                # lossAV includes final FC layer used for predictions
                _, predScore, _, _ = self.lossAV.forward(idkwhattocallthis, labels)

                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = pandas.Series(['SPEAKING_AUDIBLE' for _ in evalLines])
        scores = pandas.Series(predScores)

        evalRes = pandas.read_csv(evalOrig)
        evalRes['label'] = labels
        evalRes['score'] = scores
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        
        # change path if working directory is elsewhere
        cmd = "python3 -O TalkNet-ASD/utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(' ')[2][:5])
        return mAP

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

        for param in self.model.visualFrontend.parameters():
            param.requires_grad = False 

        for param in self.model.visualTCN.parameters():
            param.requires_grad = False

        for param in self.model.visualConv1D.parameters():
            param.requires_grad = False

        for param in self.model.audioEncoder.parameters():
            param.requires_grad = False