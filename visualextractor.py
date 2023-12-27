# visual_extractor.py

import os
import ssl
import torch
import torch.nn as nn
import torchvision.models as models

# Set the path to the CA bundle
os.environ['REQUESTS_CA_BUNDLE'] = '/path/to/cacert.pem'

# Bypass SSL verification (temporary solution)
ssl._create_default_https_context = ssl._create_unverified_context

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args['visual_extractor']
        self.pretrained = args['visual_extractor_pretrained']
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # print("below the self.visual extractor pretrained =", self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
