import uproot
import awkward as ak
import numpy as np

import seaborn as sns
import pandas as pd
import mplhep as hep
hep.style.use(hep.style.CMS)
import matplotlib.pyplot as plt

TRACK_FEATURES = [
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
    'trk_nPixelLay', 'trk_nStripLay', 'trk_n3DLay', 'trk_nLostLay', 'trk_nCluster'
]

def load_file(file_path):
    try:
        with uproot.open(file_path + ":trackingNtuple/tree") as tree:
            arrays = tree.arrays(TRACK_FEATURES + ["trk_simTrkIdx"], library="ak")
            features = {col: arrays[col] for col in TRACK_FEATURES}
            labels = ak.fill_none(ak.firsts(ak.flatten(arrays["trk_simTrkIdx"], axis=1)), -1) != -1

            return features, labels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
    

def plot_distribution(data, labels, feature_name, bins=50, range=None):
    plt.figure(figsize=(8,6))
    
    fake_data = data[labels == 0]
    real_data = data[labels == 1]
    
    plt.hist(fake_data, bins=bins, range=range, alpha=0.5, label='Fake Tracks', color='red', density=True)
    plt.hist(real_data, bins=bins, range=range, alpha=0.5, label='Real Tracks', color='blue', density=True)
    
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'Distribution of {feature_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{feature_name}_distribution.png')

def plot_correlation_matrix(data, feature_names):
    df = pd.DataFrame(data, columns=feature_names)
    corr = df.corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.savefig('plots/feature_correlation_matrix.png')

def main():
    file_path = "/ceph/cms/store/user/aaarora/tracking/output_TT_TuneCP5_14TeV-powheg-pythia8_0.root"
    features, labels = load_file(file_path)
    
    if features is None:
        print("Failed to load data.")
        return
    
    for feature in TRACK_FEATURES:
        data_array = ak.to_numpy(ak.flatten(features[feature]))
        plot_distribution(data_array, labels, feature)

    data_matrix = np.column_stack([ak.to_numpy(ak.flatten(features[feat])) for feat in TRACK_FEATURES])
    plot_correlation_matrix(data_matrix, TRACK_FEATURES)
    
if __name__ == "__main__":
    main()