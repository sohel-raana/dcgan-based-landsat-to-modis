import glob
import numpy as np
from torch.utils.data import Dataset

class LandsatMODISDataset(Dataset):
    def __init__(self, datapath, vi='NDVI', mode='train', month=None, transform=None):
        if mode == 'train':
            if month:
                self.files = glob.glob(datapath + f'crop0*_2020{month}*.npz') + \
                             glob.glob(datapath + f'crop0*_2021{month}*.npz') + \
                             glob.glob(datapath + f'crop0*_2022{month}*.npz')
            else:
                whole = glob.glob(datapath + '*.npz')
                test = glob.glob(datapath + '*_2023*.npz')
                self.files = [item for item in whole if item not in test]
        elif mode == 'test':
            if month:
                self.files = glob.glob(datapath + f'crop0*_2023{month}*.npz')
            else:
                self.files = glob.glob(datapath + '*_2023*.npz')

        self.vi = vi
        self.transform = transform

        # Debug: Print out the list of files
        print(f"Mode: {mode}, Number of files: {len(self.files)}")
        if len(self.files) == 0:
            print("Warning: No files found. Check the dataset path or file patterns.")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        temp = np.load(self.files[idx])
        landsat = temp['landsat']
        modis = temp['modis']
        dem = temp['dem']
        aoi = temp['aoi']

        if self.vi == 'NDVI':
            landsat_vi = (landsat[1] - landsat[0]) / (landsat[1] + landsat[0])
            modis_vi = (modis[1] - modis[0]) / (modis[1] + modis[0])

            landsat_vi = (landsat_vi - (-0.2)) / (1 - (-0.2))
            modis_vi = (modis_vi - (-0.2)) / (1 - (-0.2))

        dem[0] = dem[0] / 1900
        dem[1] = dem[1] / 90
        dem[2] = (dem[2] + 1) / 2

        landsat_vi = np.expand_dims(landsat_vi, axis=0)
        modis_vi = np.expand_dims(modis_vi, axis=0)
        aoi = np.expand_dims(aoi, axis=0)

        targets = landsat_vi
        inputs = np.concatenate((modis_vi, dem, aoi), axis=0)

        targets = np.nan_to_num(targets)
        inputs = np.nan_to_num(inputs)

        if self.transform:
            targets = self.transform(targets)
            inputs = self.transform(inputs)

        return inputs, targets