# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from torchsummary import summary
from tqdm import tqdm
import pickle
import re

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
#DATA_DIR = '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/Image-Text/IIU/NLMCXR_png/'
DATA_DIR = '/home/zaheer/pythonCode/R2Gen-main/data/mimic/images/'
#TEST_IMAGE_LIST = '/home/zaheer/pythonCode/ImageClassification/chexnet/Data/all_data.csv'
TEST_IMAGE_LIST = '/home/zaheer/pythonCode/MIMIC_CXR/mimic_50k.json'
BATCH_SIZE = 1


def main():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # if os.path.isfile(CKPT_PATH):
    #     print("=> loading checkpoint")
    #     checkpoint = torch.load(CKPT_PATH)
    #     model.load_state_dict(checkpoint['state_dict'],strict=False )
    #     print("=> loaded checkpoint")
    # else:
    #     print("=> no checkpoint found")
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)

        # model.load_state_dict(checkpoint['state_dict'])

        # Update checkpoint to new pytorch version
        # See https://github.com/KaiyangZhou/deep-person-reid/issues/23
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)

        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    ##### Added by Zaheer
    children_counter=0
    for n, c in model.module.densenet121.features.named_children():
        print("Children Counter: ", children_counter, " Layer Name: ", n, )
        children_counter += 1

    #### Added by Zaheer
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    output1=None
    # switch to evaluate mode
    model.eval()
    image_feat={}
    image_feat_vector = {}
    with torch.no_grad():
        for i, (file, inp, target) in enumerate(tqdm(test_loader)):
            #target = target.cuda()
            #gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            #print(bs, n_crops, c, h, w )
            input_var = inp.view(-1, c, h, w).cuda()
            #print(input_var.shape)
            output,feat = model(input_var)
            feat_mean=feat.view(bs, n_crops,7, 7, 1024).mean(1)
            #output1=model.module.densenet121.features(input_var)
            #print(feat_mean.cpu().detach().numpy().shape)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            #print(output_mean.cpu().detach().numpy().shape)
            pred = torch.cat((pred, output_mean.data), 0)
            image_feat[file[0]]=feat_mean.cpu().detach().numpy()
            #image_feat_vector[file[0]] = output_mean.cpu().detach().numpy()
    #pred=torch.argmax(pred,1)
    #pred=pred.cpu().detach().numpy()

    with open('mimic_images_features_2d.pickle', 'wb') as handle:
        pickle.dump(image_feat, handle, protocol=2)

    # with open('mimic_images_features.pickle', 'wb') as handle:
    #     pickle.dump(image_feat_vector, handle, protocol=2)
    #AUROCs = compute_AUCs(gt, pred)
    #AUROC_avg = np.array(AUROCs).mean()
    #print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    #for i in range(N_CLASSES):
     #   print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        #print(self.densenet121)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )


    def forward(self, x):
        feat=self.densenet121.features(x)
        x = self.densenet121(x)

        return x,feat


if __name__ == '__main__':
    main()