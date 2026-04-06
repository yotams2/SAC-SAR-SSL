import numpy as np
import scipy.signal
import soundfile
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from data_generation.utils_real_micsig import *
from data_generation.utils_src import *
from data_generation.utils_noise import *
import data_generation.gen_sig_from_real_rir as real_dataset
import data_generation.utils_simu_rir_sig as simu_dataset 
import data_generation.utils_LOCATA as locata_dataset

class RandomRealDataset(Dataset):
    """ Load ramdom real-world microphone signals or presaved microphone signals generated with real RIRs 
        1. for fine-tuning with ACE dataset
    """
    def __init__(
        self, 
        data_dirs, 
        T, 
        fs, 
        mic_dist_range, 
        nmic_selected, 
        stage = 'train',
        seed = 1,
        dataset_sz = None, 
        transforms = None, 
        prob_mode = ['duration', 'micpair'],
        remove_spkoverlap = True,
        dataset_list = ['LOCATA', 'MCWSJ', 'LibriCSS', 'AMI', 'AISHELL4', 'M2MeT', 'RealMAN', # RealSig
					    'DCASE', 'MIR', 'Mesh', 'ACE', 'dEchorate', 'BUTReverb'], # RealRIR
        dataset_probs = None, 
        sound_speed = 343.0,
        load_spec_feats = False):
        # RealSig
        # train duration (hours): [33, 74, 83, 101, 4124, 835979, 514811], 
        # train duration (hours, utterance level) [0.07, 1.3, 2.0, 3.4, 74, 95, 111]
        # room condition: [1 3 1 5 3 10 13]
        # [1, 5, 5, 5, 8, 8, 8]

        print(f'dataset list: {dataset_list}')
        print(f'dataset ratio: {dataset_probs}')
 
        self.dataset_list = []
        for dataset_name in dataset_list:
            data_dir = data_dirs[dataset_name]
            if dataset_name in ['LOCATA', 'MCWSJ', 'LibriCSS', 'AMI', 'AISHELL4', 'M2MeT', 'RealMAN', 'RealMANOri']:
                if (dataset_name == 'AISHELL4') or (dataset_name == 'M2MeT'):
                    remove_spkoverlap = True
                else:
                    remove_spkoverlap = False
                dataset_name_str = dataset_name+'Dataset'

                dataset = globals()[dataset_name_str](
                    data_dir = data_dir,
                    T = T,
                    fs = fs,
                    stage = stage, 
                    mic_dist_range = mic_dist_range,
                    nmic_selected = nmic_selected,
                    prob_mode = prob_mode,
                    dataset_sz = None, 
                    remove_spkoverlap = remove_spkoverlap,
                    sound_speed = sound_speed,
                    load_spec_feats = load_spec_feats)
               
                self.dataset_list += [dataset]
            elif dataset_name in ['DCASE', 'MIR', 'Mesh', 'ACE', 'dEchorate', 'BUTReverb']:
                ds_sz = {'train': 102400, 'val':2560, 'test':2560}
                dataset_rir = FixMicSigDataset(
                    data_dir = data_dir,
                    load_anno = False,
                    fs = fs,
                    dataset_sz = ds_sz[stage],
                    transforms = None,
                    load_spec_feats = load_spec_feats
                )
                self.dataset_list += [dataset_rir]

        self.dataset_sz = dataset_sz
        self.transforms = transforms
        self.seed = seed
      
        assert len(self.dataset_list) == len(dataset_probs), [len(self.dataset_list), len(dataset_probs)]
            
        dataset_probs_sum = sum(dataset_probs)
        dataset_probs = [prob/dataset_probs_sum for prob in dataset_probs]
        self.ds_probs_cumsum = np.cumsum(dataset_probs, dtype=np.float32)
        self.ds_probs_cumsum[-1] = 1
 
    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):

        dataset_idx = np.searchsorted(self.ds_probs_cumsum, np.random.uniform())
        ins_idx = np.random.randint(0, len(self.dataset_list[dataset_idx]))
        # np.random.seed(seed=self.seed+ins_idx)
        mic_sig = self.dataset_list[dataset_idx].__getitem__(ins_idx)
 
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        if isinstance(mic_sig, list):
             mic_sig += [idx]
        else:
             mic_sig = [mic_sig, idx]

        return mic_sig 


class FixMicSigDataset(Dataset):
    """ Load fixed presaved microphone signals (simulated data or microphones signal generated with real RIRs)
        1. for pretraining with simulated data
        2. for pretraining with microphone signals generated with real RIR
        3. for fine-tuning with only simulated data
	"""
    def __init__(self, data_dir, fs, load_anno, dataset_sz, load_dp=False, transforms=None, sort=False, load_spec_feats=False):

        self.data_paths = []
        
        if isinstance(data_dir, list):
            files = []
            dp_files = []
            for d in data_dir:
                files += list(Path(d).rglob('*.wav'))
                dp_files += list(Path(d).rglob('*_dp.wav'))
            np.random.shuffle(files)
        else:
            files = list(Path(data_dir).rglob('*.wav'))
            dp_files = list(Path(data_dir).rglob('*_dp.wav'))

        if sort:
            files.sort(key=lambda x: int(x.stem))
            
        self.files = [item for item in files if item not in dp_files]

        if dataset_sz is not None:
            self.dataset_sz = np.min([len(self.files), dataset_sz])
        else:
            self.dataset_sz = len(self.files)
        self.fs = fs
        self.load_anno = load_anno
        self.load_dp = load_dp
        self.transforms = transforms
        self.data_dir = data_dir
        self.load_spec_feats = load_spec_feats

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        file_name = str(self.files[idx])
        mic_sig, fs = soundfile.read(file_name)

        if self.fs != fs:
            mic_sig = scipy.signal.resample_poly(mic_sig, self.fs, fs)
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        return_data = [mic_sig.astype(np.float32)]
        if self.load_anno:
            info_file_name = file_name.replace('.wav', '_info.npz')
            info = dict(np.load(info_file_name))
            vol = info['room_sz'][0] * info['room_sz'][1] * info['room_sz'][2]
            sur = info['room_sz'][0] * info['room_sz'][1] + info['room_sz'][0] * info['room_sz'][2] + info['room_sz'][1] * info['room_sz'][2]
            annos = {
                'TDOA': info['TDOA'].astype(np.float32), 
                'T60': info['T60_edc'].astype(np.float32),  
                'DRR': info['DRR'].astype(np.float32),
                'C50': info['C50'].astype(np.float32),
                'ABS': np.array(0.161*vol/sur/info['T60_edc']).astype(np.float32),
                }
            if 'src_idx' in info:
                annos['src_idx'] = np.array(info['src_idx']).astype(np.float32)
            if 'SNR' in info:
                annos['SNR'] = np.array(info['SNR']).astype(np.float32)
            return_data += [annos]

        if self.load_dp:
            dp_file_name = file_name.replace('.wav', '_dp.wav')
            dp_sig, _ = soundfile.read(dp_file_name)
            if self.fs != fs:
                dp_sig = scipy.signal.resample_poly(dp_sig, self.fs, fs)
            if self.transforms is not None:
                for t in self.transforms:
                    dp_sig = t(dp_sig)
            return_data += [dp_sig]

        if getattr(self, 'load_spec_feats', False):
            import os
            import torch
            pt_file = str(file_name).replace('.wav', '_spec_feats.pt')
            
            nsample_desired_16k = mic_sig.shape[0] if fs == 16000 else int((mic_sig.shape[0] / fs) * 16000)
            hop_length = 256
            nt = (nsample_desired_16k - 512) // hop_length + 1
            
            spec_feats = None
            if os.path.exists(pt_file):
                try:
                    spec_feats_full = torch.load(pt_file, map_location='cpu')
                    st_16k = 0
                    k = st_16k // hop_length
                    fm = spec_feats_full['feature_matrix'][:, k : k + nt]
                    vp = spec_feats_full['voiced_prob'][k : k + nt]
                    if fm.shape[-1] < nt:
                        pad_len = nt - fm.shape[-1]
                        import torch.nn.functional as F
                        fm = F.pad(fm, (0, pad_len))
                        vp = F.pad(vp, (0, pad_len))
                    spec_feats = {'feature_matrix': fm, 'voiced_prob': vp}
                except Exception as e:
                    print(f"Error loading {pt_file}: {e}")
            
            if spec_feats is None:
                spec_feats = {
                    'feature_matrix': torch.zeros((16, nt), dtype=torch.float32),
                    'voiced_prob': torch.zeros((nt,), dtype=torch.float32)
                }
            return_data.append(spec_feats)

        return_data += [idx]
        return return_data

 
class FixMicSigDatasetLOCATA(Dataset):
    """ Load fixed presaved microphone signals from LOCATA dataset
	"""
    def __init__(self, data_dir, fs, load_anno, dataset_sz, transforms=None, load_spec_feats=False):

        self.data_paths = []
        
        if isinstance(data_dir, list):
            self.files = []
            for d in data_dir:
                self.files += list(Path(d).rglob('*.wav'))
            np.random.shuffle(self.files)
        else:
            self.files = list(Path(data_dir).rglob('*.wav'))

        if dataset_sz is not None:
            self.dataset_sz = np.min([len(self.files), dataset_sz])
        else:
            self.dataset_sz = len(self.files)
        self.fs = fs
        self.load_anno = load_anno
        self.transforms = transforms
        self.load_spec_feats = load_spec_feats

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):

        file_name = str(self.files[idx])
        mic_sig, fs = soundfile.read(file_name)

        if self.fs != fs:
            mic_sig = scipy.signal.resample_poly(mic_sig, self.fs, fs)
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        return_data = [mic_sig.astype(np.float32)]
        if self.load_anno:
            info_file_name = file_name.replace('.wav', '_info.npz')
            info = dict(np.load(info_file_name))
            annos = {
                'TDOA': info['TDOA'].astype(np.float32), 
                'T60': np.array(np.NAN), 
                'DRR': np.array(np.NAN),
                'C50': np.array(np.NAN),
                'ABS': np.array(np.NAN),
                }
            if 'src_idx' in info:
                annos['src_idx'] = np.array(info['src_idx']).astype(np.float32)
            if 'SNR' in info:
                annos['SNR'] = np.array(info['SNR']).astype(np.float32)
            return_data += [annos]

        if getattr(self, 'load_spec_feats', False):
            import os
            import torch
            pt_file = str(file_name).replace('.wav', '_spec_feats.pt')
            
            nsample_desired_16k = mic_sig.shape[0] if fs == 16000 else int((mic_sig.shape[0] / fs) * 16000)
            hop_length = 256
            nt = (nsample_desired_16k - 512) // hop_length + 1
            
            spec_feats = None
            if os.path.exists(pt_file):
                try:
                    spec_feats_full = torch.load(pt_file, map_location='cpu')
                    st_16k = 0
                    k = st_16k // hop_length
                    fm = spec_feats_full['feature_matrix'][:, k : k + nt]
                    vp = spec_feats_full['voiced_prob'][k : k + nt]
                    if fm.shape[-1] < nt:
                        pad_len = nt - fm.shape[-1]
                        import torch.nn.functional as F
                        fm = F.pad(fm, (0, pad_len))
                        vp = F.pad(vp, (0, pad_len))
                    spec_feats = {'feature_matrix': fm, 'voiced_prob': vp}
                except Exception as e:
                    print(f"Error loading {pt_file}: {e}")
            
            if spec_feats is None:
                spec_feats = {
                    'feature_matrix': torch.zeros((16, nt), dtype=torch.float32),
                    'voiced_prob': torch.zeros((nt,), dtype=torch.float32)
                }
            return_data.append(spec_feats)

        return return_data

class RandomMicSigDataset(Dataset):
    """ Load random presaved microphone signals (*wav)
        1. for fine-tuning with both LOCATA and presaved simulated data 
    """
    def __init__(self, 
        real_sig_dir, 
        sim_sig_dir, 
        real_sim_ratio, 
        fs,
        stage,
        load_anno, 
        dataset_sz, 
        transforms=None,
        load_spec_feats=False):

        realdataset = FixMicSigDatasetLOCATA(
            data_dir = os.path.join(real_sig_dir, stage),
            load_anno = load_anno,
            dataset_sz = None,
            fs = fs,
            transforms = transforms,
            load_spec_feats = load_spec_feats
            )

        simdataset = FixMicSigDataset(
            data_dir = sim_sig_dir,
            load_anno = load_anno,
            dataset_sz = None,
            fs = fs,
            transforms = transforms,
            load_spec_feats = load_spec_feats
            )
        
        assert real_sim_ratio in [[0,1], [1,0], [1,1]], real_sim_ratio
        if real_sim_ratio == [0,1]:
            self.dataset_list = [simdataset]
        elif real_sim_ratio == [1,0]:
            self.dataset_list = [realdataset]
        elif real_sim_ratio == [1,1]:
            self.dataset_list = [simdataset, realdataset]
        self.dataset_sz = dataset_sz
        self.load_anno = load_anno
    
    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx=None):

        dataset_idx = np.random.randint(0, len(self.dataset_list))
        dataset = self.dataset_list[dataset_idx]
        idx = np.random.randint(0, len(dataset))
        if self.load_anno:
            mic_sig, annos = dataset.__getitem__(idx)
            return mic_sig.astype(np.float32), annos 
        else:
            mic_sig = dataset.__getitem__(idx)
            return mic_sig.astype(np.float32)

class RandomMicSigFromRIRDataset(Dataset):
    """ Generate microphone signals from real RIRs
        1. for fine-tuning with microphone signals generated with real RIR+ simulated RIR
	"""
    def __init__(
        self, 
        real_rir_dir_list, 
        sim_rir_dir_list, 
        src_dir,
        dataset_sz, 
        T, 
        fs, 
        c,
        nmic,
        snr_range, 
        real_sim_ratio,
        transforms=None,
        seed=1
        ):
        
        srcdataset = WSJ0Dataset(
            path = src_dir,
            T = T,
            fs = fs
            )
        noidataset = NoiseSignal(
            T = T,
            fs = fs,
            nmic = nmic,
            noise_type = 'diffuse_white',
            noise_path = '',
            c = c
            )
        realrirdataset = real_dataset.RIRDataset(
            fs=fs, 
            rir_dir_list=real_rir_dir_list, 
            dataset_sz=None, 
            load_info=True, 
            load_noise=True, 
            load_noise_duration=T
            ) 
        realdataset = real_dataset.MicSigFromRIRDataset(
            rirnoidataset=realrirdataset,
            srcdataset=srcdataset,
            snr_range=snr_range,
            fs=fs,
            dataset_sz=None,
            seed=seed,
            load_info=True,
            save_anno=False,
            save_to=None,
            )
        simrirdataset = simu_dataset.RIRDataset(
            fs=fs,
            rir_dir_list=sim_rir_dir_list,
            dataset_sz=None,
            load_dp=True,
            load_info=True
            )
        simdataset = simu_dataset.MicSigFromRIRDataset(
            rirdataset=simrirdataset,
            srcdataset=srcdataset,
            noidataset=noidataset,
            snr_range=snr_range,
            fs=fs,
            dataset_sz=None,
            seed=seed,
            load_info=True,
            save_anno=False,
            save_to=None, 
            )
        assert real_sim_ratio in [[0,1], [1,0], [1,1]], real_sim_ratio
        if real_sim_ratio == [0,1]:
            self.dataset_list = [simdataset]
        elif real_sim_ratio == [1,0]:
            self.dataset_list = [realdataset]
        elif real_sim_ratio == [1,1]:
            self.dataset_list = [realdataset, simdataset]
        self.seed = seed
        self.dataset_sz = dataset_sz
        self.transforms = transforms

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx=None):
        
        dataset_idx = np.random.randint(0, len(self.dataset_list))
        dataset = self.dataset_list[dataset_idx]
        idx = np.random.randint(0, len(dataset))
        mic_sig, annos = dataset.__getitem__(idx)
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        return mic_sig.astype(np.float32), annos
 

## Transform classes
class Selecting(object):
    def __init__(self, select_range):
        self.select_range = select_range

    def __call__(self, mic_sig):
        nsample = mic_sig.shape[0]
        assert self.select_range[-1]<=nsample, f'Selecting range ({self.select_range[-1]}) is larger than signal length ({nsample})~'
        mic_sig = mic_sig[self.select_range[0]:self.select_range[1], ...]
        
        return mic_sig


if __name__ == "__main__":
    pass