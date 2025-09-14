import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from timm.layers import DropPath, trunc_normal_
class SuperTileRNADataset(Dataset):
    def __init__(self, ref_path: str, feature_path,extraction_model,cluster_type,cancer_project):
        self.feature_path = feature_path
        self.extraction_model = extraction_model
        self.data = ref_path
        self.cluster_type = cluster_type
        self.cancer_project = cancer_project
        # find the number of genes
        row = self.data.iloc[0]
        Omics = row[[x for x in row.keys()[1:]]].values.astype(np.float32)
        self.num_genes = len(Omics)
        # find the feature dimension, assume all images in the reference file have the same dimension
        feature_path_list = os.path.join(feature_path,row["patient_id"] +".h5")
        f = h5py.File(feature_path_list, 'r')

        features = f[cluster_type][:]
        self.feature_dim = features.shape[1]
        f.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature_path_list = os.path.join(self.feature_path,row["patient_id"] +".h5")
        Omics = row[[x for x in row.keys()[1:]]].values.astype(np.float32)
        Omics = torch.tensor(Omics, dtype=torch.float32)
        try:
            f = h5py.File(feature_path_list, 'r')
            features = f[self.cluster_type][:]
            f.close()
            features = torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(e)
            print(path)
            features = None
        return features,Omics,row["patient_id"],self.cancer_project