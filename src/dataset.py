import torch
from torch.utils.data import Dataset

import uproot
import awkward as ak

class MinMaxScaler:
    def __init__(self, data):
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max - self.min + 1e-8) + self.min
    
    def __call__(self, sample):
        data, label = sample
        return self.transform(data), label


class TrackDataset(Dataset):
    def __init__(self, input_files, transform=None):
        self.data = []
        self.labels = []

        columns = [
            'trk_px', 'trk_py', 'trk_pz', 'trk_pt', 
            'trk_inner_px', 'trk_inner_py', 'trk_inner_pz', 'trk_inner_pt', 
            'trk_outer_px', 'trk_outer_py', 'trk_outer_pz', 'trk_outer_pt',
            'trk_eta', 'trk_lambda', 'trk_cotTheta', 'trk_phi', 
            'trk_dxy', 'trk_dz', 'trk_dxyPV', 'trk_dzPV', 'trk_dxyClosestPV', 'trk_dzClosestPV',
            'trk_ptErr', 'trk_etaErr', 'trk_lambdaErr', 'trk_phiErr', 'trk_dxyErr', 'trk_dzErr',
            'trk_refpoint_x', 'trk_refpoint_y', 'trk_refpoint_z',
            'trk_nChi2', 'trk_nChi2_1Dmod', 'trk_ndof', 'trk_q', 
            'trk_nValid', 'trk_nLost', 'trk_nInactive', 'trk_nPixel', 'trk_nStrip', 
            'trk_nOuterLost', 'trk_nInnerLost', 'trk_nOuterInactive', 'trk_nInnerInactive',
            'trk_nPixelLay', 'trk_nStripLay', 'trk_n3DLay', 'trk_nLostLay', 'trk_nCluster', 'trk_simTrkShareFrac'
        ]

        for file in input_files:
            with uproot.open(file + ":trackingNtuple/tree") as tree:
                features = ak.to_dataframe(tree.arrays(columns, library="ak"))
                labels = ak.to_dataframe(tree.arrays(columns[-1], library="ak"))

                features.drop(columns=['trk_simTrkShareFrac'], inplace=True)
    
                self.data.append(torch.Tensor(features.values))
                self.labels.append(torch.Tensor(labels.values))
                
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        
        if transform is not None and isinstance(transform, type):
            self.transform = transform(self.data)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = (self.data[idx], self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
    
# testing 
if __name__ == "__main__":
    dataset = TrackDataset(["/home/users/aaarora/phys/tracking/purity-mva/data/trackingNtuple.root"])