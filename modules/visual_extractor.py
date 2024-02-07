import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pickle

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        #modules = list(model.children())[:-1]
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        #print(modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        with open('images_semantic_features_2d_report.pickle', 'rb') as handle:
        #with open('./data/mimic/chexnet_mimic_features.pickle', 'rb') as handle:
        #with open('/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/r2gen_annotations/densenet_report_features.pkl', 'rb') as handle:
            self.chexnet_features = pickle.load(handle)

    def forward(self, images,image_ids=None,image_num=0):
        size = 15
        patch_feats=np.zeros((len(image_ids),size,7,7))
        for idx,id in enumerate(image_ids):
            # print(id)
            patch_feats[idx,:]=self.chexnet_features[id[0].split('/')[-1]].reshape(1,size,7,7)
        patch_feats = torch.from_numpy(patch_feats).float().cuda()
        #patch_feats = self.model(images)
        #print(patch_feats.shape)

        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        #print(type(patch_feats), patch_feats.shape, avg_feats.shape)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
