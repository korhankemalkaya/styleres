from numpy import product
import torch
from torch import nn
from arcface.model_irse import Backbone


class ArcFace(nn.Module):
    def __init__(self, path):
        super(ArcFace, self).__init__()
        #print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=100, drop_ratio=0.4, mode='ir', affine=False)
        self.facenet.load_state_dict(torch.load(path, map_location='cpu'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        #self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, real, masked):
        real_feats = self.extract_feats(real)
        masked_feats = self.extract_feats(masked)
        #cosine_similarity = torch.sum(real_feats * masked_feats, dim=1)
        cosine_similarity = self.cos(real_feats, masked_feats)
        return cosine_similarity
