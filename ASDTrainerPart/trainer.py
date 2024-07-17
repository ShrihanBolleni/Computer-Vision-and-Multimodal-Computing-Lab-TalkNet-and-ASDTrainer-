import os, glob, time
import argparse
from model import *
from dataLoader_Image_audio import TrainLoader, ValLoader
from matplotlib import pyplot as plt
import numpy as np

def parser():
    args = argparse.ArgumentParser(description="ASD Trainer")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--lrDecay', type=float, default=0.95, help='Learning rate decay rate')
    args.add_argument('--maxEpoch', type=int, default=10, help='Maximum number of epochs')
    args.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    args.add_argument('--batchSize', type=int, default=64, help='Dynamic batch size, default is 500 frames.')
    args.add_argument('--nDataLoaderThread', type=int, default=8, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="/mnt/c/Users/james/PycharmProjects/CVMC/AVDIAR_ASD/", help='Path to the ASD Dataset')
    args.add_argument('--loadAudioSeconds', type=float, default=3, help='Number of seconds of audio to load for each training sample')
    args.add_argument('--loadNumImages', type=int, default=5, help='Number of images to load for each training sample')
    args.add_argument('--savePath', type=str, default="Assignment_0")
    args.add_argument('--evalDataType', type=str, default="val", help='The dataset for evaluation, val or test')
    args.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation')
    args.add_argument('--eval_model_path', type=str, default="path not specified", help="model path for evaluation")

    args = args.parse_args()

    return args


def display_history(loss_history, mAP_history, path):
    plt.plot(np.arange(1, len(loss_history) + 1, step=1), loss_history, label='loss')
    plt.plot(np.arange(1, len(mAP_history) + 1, step=1), mAP_history, label='mAP')
    plt.xticks(np.arange(1, len(mAP_history) + 1, step=1))
    plt.savefig(path + '/history')


def main(args):
    loader = TrainLoader(trialFileName=os.path.join(args.datasetPath, 'csv/train_loader.csv'),
                         audioPath=os.path.join(args.datasetPath, 'clips_audios/'),
                         visualPath=os.path.join(args.datasetPath, 'clips_videos/train'),
                         **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size=args.batchSize, shuffle=True,
                                              num_workers=args.nDataLoaderThread)

    loader = ValLoader(trialFileName=os.path.join(args.datasetPath, 'csv/val_loader.csv'),
                       audioPath=os.path.join(args.datasetPath, 'clips_audios'),
                       visualPath=os.path.join(args.datasetPath, 'clips_videos', args.evalDataType),
                       **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size=args.batchSize, shuffle=False, num_workers=4)


    # if user selects to only evaluate the model:
    if args.evaluation:
        s = Model(**vars(args))

        # no path to saved model, cannot continue
        if args.eval_model_path == "path not specified":
            print('Evaluation model parameters path has not been specified')
            quit()

        # load parameters from path and evaluate on specified set
        s.loadParameters(args.eval_model_path)
        print("Parameters loaded from path ", args.eval_model_path)
        loss, mAP = s.evaluate_network(loader=valLoader)
        print("mAP %2.2f%%" % mAP)
        quit()

    
    # Either loads a previous checkpoint or starts training from scratch
    args.modelSavePath = os.path.join(args.savePath, 'model')
    os.makedirs(args.modelSavePath, exist_ok=True)
    args.scoreSavePath = os.path.join(args.savePath, 'score.txt')
    modelfiles = glob.glob('%s/model_0*.model' % args.modelSavePath)
    modelfiles.sort()

    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        start_epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = Model(**vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        start_epoch = 1
        s = Model(**vars(args))

    # output device usage
    print(f"Using {s.device}")

    mAPs, losses = s.train_network(start_epoch=start_epoch, train_loader=trainLoader, val_loader=valLoader)
    # graph history of loss and mAP
    display_history(losses, mAPs, args.savePath)
        

if __name__ == "__main__":
    args = parser()

    main(args)
