"""	
	Run training and test processes for self-supervised learning of spatial acoustic representation
	Reference:  Self-Supervised Learning of Spatial Acoustic Representation with Cross-Channel Signal Reconstruction and Multi-Channel Conformer
	Author:     Bing Yang
	History:    2024-07 - Initial version
	Copyright Bing Yang

	Examples:
		python run_pretrain.py --pretrain --simu-exp --gpu-id 0,

		# * denotes the time version of pre-training model  
		# --test-mode all: all or ins
		python run_pretrain.py --test --simu-exp --time * --test-mode all --gpu-id 0, 

		python run_pretrain.py --pretrain --gpu-id 0, 
"""

import os
import sys
import json
cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num) # Limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import wandb

from opt import opt_pretrain
opts = opt_pretrain()
args = opts.parse()
dirs = opts.dir()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# Load Config from config.json if testing
if args.test:
    config_path = os.path.join(dirs['log_pretrain'], 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                # config_data is a list, first element is args dict
                if isinstance(config_data, list) and len(config_data) > 0:
                    args_dict = config_data[0]
                    args.spec_dembed = args_dict.get('spec_dembed', 512)
                    args.spat_dembed = args_dict.get('spat_dembed', 256)
                    args.spectral_enc_input = args_dict.get('spectral_enc_input', 'all')
                    args.pretrain_mode = args_dict.get('pretrain_mode', 'masked')
                    print(f"Loaded config from {config_path}: spec_dembed={args.spec_dembed}, spat_dembed={args.spat_dembed}, spectral_enc_input={args.spectral_enc_input}")
                else:
                    print(f"Warning: config.json structure unexpected at {config_path}. Using defaults/args.")
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}. Using defaults/args.")
    else:
        print(f"Warning: config.json not found in {dirs['log_pretrain']}. Using defaults/args.")

import numpy as np
from torch.utils.data import Sampler
from pathlib import Path

def count_wav_files(directory_or_dict):
    """
    Recursively counts .wav files in a directory or dict of directories.
    Excludes *_dp.wav files.
    """
    count = 0
    if isinstance(directory_or_dict, dict):
        for key, val in directory_or_dict.items():
            count += count_wav_files(val)
    elif isinstance(directory_or_dict, list):
        for d in directory_or_dict:
             count += count_wav_files(d)
    elif isinstance(directory_or_dict, str):
        path = Path(directory_or_dict)
        if path.exists():
            # Logic from dataset.py:FixMicSigDataset
            files = list(path.rglob('*.wav'))
            dp_files = list(path.rglob('*_dp.wav'))
            # Convert to sets for faster exclusion if needed, but list check is fine for moderate size
            # dataset.py uses: [item for item in files if item not in dp_files]
            # Since Path objects compare by path, this works.
            valid_files = [f for f in files if f not in dp_files]
            count = len(valid_files)
    return count


class ContrastiveBatchSampler(Sampler):
    def __init__(self, num_rooms, num_utterances, batch_size, rooms_per_batch=8, utterances_per_batch=16):
        self.num_rooms = num_rooms
        self.num_utterances = num_utterances
        self.batch_size = batch_size
        self.rooms_per_batch = rooms_per_batch
        self.utterances_per_batch = utterances_per_batch
        assert rooms_per_batch * utterances_per_batch == batch_size
        
        self.num_batches = (num_rooms * num_utterances) // batch_size 

    def __iter__(self):
        for _ in range(self.num_batches):
            rooms = np.random.choice(self.num_rooms, self.rooms_per_batch, replace=False)
            utterances = np.random.choice(self.num_utterances, self.utterances_per_batch, replace=False)
            
            batch = []
            for r in rooms:
                for u in utterances:
                    idx = r * self.num_utterances + u
                    batch.append(idx)
            yield batch

    def __len__(self):
        return self.num_batches

class ContrastiveAugmentationBatchSampler(Sampler):
    def __init__(self, total_samples, batch_size, augmentations_per_sample=8, n_distinct_samples=0):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.augmentations_per_sample = augmentations_per_sample
        self.num_base_samples = total_samples // augmentations_per_sample
        
        # If n_distinct_samples is not specified or 0, default to using all augmentations (old behavior)
        # Old behavior: samples_per_batch = batch_size // augmentations_per_sample (e.g. 128 // 8 = 16)
        if n_distinct_samples <= 0:
            self.n_distinct_samples = batch_size // augmentations_per_sample
        else:
            self.n_distinct_samples = n_distinct_samples

        self.samples_per_batch = self.n_distinct_samples
        
        # Calculate how many augmentations to pick per distinct sample to fill the batch
        # e.g., Batch 128, Distinct 64 -> 128 // 64 = 2 augs per sample
        self.augs_per_batch_item = batch_size // self.samples_per_batch
        
        # Verify batch size consistency
        if self.batch_size != self.samples_per_batch * self.augs_per_batch_item:
             print(f"Warning: Batch size {batch_size} is not divisible by n_distinct_samples {self.samples_per_batch}. Actual batch size will be {self.samples_per_batch * self.augs_per_batch_item}")
        
        assert self.augs_per_batch_item <= augmentations_per_sample, f"Requested {self.augs_per_batch_item} augmentations per sample, but only {augmentations_per_sample} available."

        self.num_batches = self.total_samples // batch_size # Approx number of batches

    def __iter__(self):
        for _ in range(self.num_batches):
            base_indices = np.random.choice(self.num_base_samples, self.samples_per_batch, replace=False)
            
            batch = []
            for base_idx in base_indices:
                start_idx = base_idx * self.augmentations_per_sample
                # Available indices for this base sample: [start_idx, start_idx + 8)
                available_indices = np.arange(start_idx, start_idx + self.augmentations_per_sample)
                
                # Select random subset of augmentations
                selected_indices = np.random.choice(available_indices, self.augs_per_batch_item, replace=False)
                batch.extend(selected_indices)
            yield batch

    def __len__(self):
        return self.num_batches

import torch
import matplotlib.pyplot as plt
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.set_float32_matmul_precision('medium')
# torch.set_num_threads(cpu_num)

import scipy.io
import soundfile
from tensorboardX import SummaryWriter
import dataset as at_dataset
import learner as at_learner
import model as at_model
from common.utils import set_seed, set_random_seed, create_learning_rate_schedule, get_nparams, get_FLOPs, save_config_to_file, vis_time_fre_data, plot_learning_curve, vis_TSNE, cross_validation_datadir
from common.utils_module import spatial_features_from_stft

def calculate_feature_stats(dataloader, device, learner, fs):
	print("Calculating global feature statistics...")
	ild_values = []
	tdoa_values = []
	coherence_values = []
	coherence_low_values = []
	coherence_mid_values = []
	coherence_high_values = []
	gcc_values = []
	hf_lf_values = []
	iacc_values = []
	decay_median_values = []
	decay_p10_values = []
	decay_p90_values = []
	mod_ratio_values = []
	elr_mean_values = []
	elr_median_values = []
	num_batches_for_std = 50
	
	# Helper for stats
	def get_stats(tensor_list):
		all_vals = torch.cat(tensor_list)
		mean = torch.mean(all_vals).item()
		std = torch.std(all_vals).item()
		return mean, std

	with torch.no_grad():
		for i, data in enumerate(dataloader):
			if i >= num_batches_for_std:
				break
			mic_sig_batch = data[0].to(device)
			# Calculate STFT using learner's stft module
			# learner.stft returns [B, F, T, C]
			stft = learner.stft(mic_sig_batch) 
			stft = stft.permute(0, 3, 1, 2) # [B, C, F, T]
			feats = spatial_features_from_stft(stft, fs, max_ipd_freq=1500.0)
			ild_values.append(feats['ild_mean'])
			tdoa_values.append(feats['tdoa'])
			# collect banded coherence features
			coherence_values.append(feats['coherence_mean'])
			coherence_low_values.append(feats['coherence_low'])
			coherence_mid_values.append(feats['coherence_mid'])
			coherence_high_values.append(feats['coherence_high'])
			# collect GCC-PHAT peak and HF/LF ratio
			gcc_values.append(feats['gcc_phat_max'])
			hf_lf_values.append(feats['hf_lf_ratio'])

			# New features
			iacc_values.append(feats['iacc_max'])
			decay_median_values.append(feats['decay_slope_median'])
			decay_p10_values.append(feats['decay_slope_p10'])
			decay_p90_values.append(feats['decay_slope_p90'])
			mod_ratio_values.append(feats['mod_low_high_ratio'])
			elr_mean_values.append(feats['early_late_ratio_mean'])
			elr_median_values.append(feats['early_late_ratio_median'])

	feature_stats = {}
	if len(ild_values) > 0:
		# ILD
		mean_ild, std_ild = get_stats(ild_values)
		feature_stats['ild_mean'] = {'mean': mean_ild, 'std': std_ild}
		print(f"ILD Mean: {mean_ild:.4f}, STD: {std_ild:.4f}")

		# TDOA
		mean_tdoa, std_tdoa = get_stats(tdoa_values)
		feature_stats['tdoa'] = {'mean': mean_tdoa, 'std': std_tdoa}
		print(f"TDOA Mean: {mean_tdoa:.6f}, STD: {std_tdoa:.6f}")

		# Coherence
		mean_coh, std_coh = get_stats(coherence_values)
		feature_stats['coherence_mean'] = {'mean': mean_coh, 'std': std_coh}
		print(f"Coherence Mean: {mean_coh:.4f}, STD: {std_coh:.4f}")

		mean_coh_low, std_coh_low = get_stats(coherence_low_values)
		feature_stats['coherence_low'] = {'mean': mean_coh_low, 'std': std_coh_low}
		print(f"Coherence Low (<500Hz) Mean: {mean_coh_low:.4f}, STD: {std_coh_low:.4f}")

		mean_coh_mid, std_coh_mid = get_stats(coherence_mid_values)
		feature_stats['coherence_mid'] = {'mean': mean_coh_mid, 'std': std_coh_mid}
		print(f"Coherence Mid (500-2500Hz) Mean: {mean_coh_mid:.4f}, STD: {std_coh_mid:.4f}")

		mean_coh_high, std_coh_high = get_stats(coherence_high_values)
		feature_stats['coherence_high'] = {'mean': mean_coh_high, 'std': std_coh_high}
		print(f"Coherence High (>2500Hz) Mean: {mean_coh_high:.4f}, STD: {std_coh_high:.4f}")

		mean_gcc, std_gcc = get_stats(gcc_values)
		feature_stats['gcc_phat_max'] = {'mean': mean_gcc, 'std': std_gcc}
		print(f"GCC-PHAT Max Mean: {mean_gcc:.4f}, STD: {std_gcc:.4f}")

		mean_hf_lf, std_hf_lf = get_stats(hf_lf_values)
		feature_stats['hf_lf_ratio'] = {'mean': mean_hf_lf, 'std': std_hf_lf}
		print(f"HF/LF Ratio (log10) Mean: {mean_hf_lf:.4f}, STD: {std_hf_lf:.4f}")

		# New Features Stats
		mean_iacc, std_iacc = get_stats(iacc_values)
		feature_stats['iacc_max'] = {'mean': mean_iacc, 'std': std_iacc}
		print(f"IACC Max Mean: {mean_iacc:.4f}, STD: {std_iacc:.4f}")

		mean_decay_med, std_decay_med = get_stats(decay_median_values)
		feature_stats['decay_slope_median'] = {'mean': mean_decay_med, 'std': std_decay_med}
		print(f"Decay Slope Median Mean: {mean_decay_med:.4f}, STD: {std_decay_med:.4f}")

		mean_decay_p10, std_decay_p10 = get_stats(decay_p10_values)
		feature_stats['decay_slope_p10'] = {'mean': mean_decay_p10, 'std': std_decay_p10}
		print(f"Decay Slope P10 Mean: {mean_decay_p10:.4f}, STD: {std_decay_p10:.4f}")

		mean_decay_p90, std_decay_p90 = get_stats(decay_p90_values)
		feature_stats['decay_slope_p90'] = {'mean': mean_decay_p90, 'std': std_decay_p90}
		print(f"Decay Slope P90 Mean: {mean_decay_p90:.4f}, STD: {std_decay_p90:.4f}")

		mean_mod, std_mod = get_stats(mod_ratio_values)
		feature_stats['mod_low_high_ratio'] = {'mean': mean_mod, 'std': std_mod}
		print(f"Modulation Ratio Mean: {mean_mod:.4f}, STD: {std_mod:.4f}")

		mean_elr, std_elr = get_stats(elr_mean_values)
		feature_stats['early_late_ratio_mean'] = {'mean': mean_elr, 'std': std_elr}
		print(f"ELR Mean Mean: {mean_elr:.4f}, STD: {std_elr:.4f}")

		mean_elr_med, std_elr_med = get_stats(elr_median_values)
		feature_stats['early_late_ratio_median'] = {'mean': mean_elr_med, 'std': std_elr_med}
		print(f"ELR Median Mean: {mean_elr_med:.4f}, STD: {std_elr_med:.4f}")

	else:
		print("Warning: Could not calculate feature statistics.")
		
	return feature_stats


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

# Save config file
if args.pretrain:
	os.makedirs(dirs['log_pretrain'], exist_ok=True)
	file_path = os.path.join(dirs['log_pretrain'],"config.json")
	command = 'python ' + ' '.join(sys.argv)
	save_config_to_file([args.__dict__, dirs, {'command': command}], file_path)

	if args.description is not None:
		with open(os.path.join(dirs['log_pretrain'], "description.txt"), "w") as f:
			f.write(args.description)

# WandB
if args.test:
	args.wandb = False
if args.wandb:
	wandb.init(project='SAR-SSL', config=args, name=args.description)

# Acoustic setting parameters
assert args.source_state == 'static', 'Source state model unrecognized~'
nmic = args.acoustic_setting['nmic']
speed = args.acoustic_setting['sound_speed']	
fs = args.acoustic_setting['fs']
T = args.acoustic_setting['T']
mic_dist_range = args.acoustic_setting['mic_dist_range'] 
seeds = {'train': int(args.seed+4e8), 'val': int(args.seed+1e8), 'test': int(args.seed+1)} 
 
# STFT parameters
win_len = 512
nfft = 512
win_shift_ratio = 0.5
fre_used_ratio = 1
nf = nfft//2
nt = int((T * fs - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
print(f"T: {T:.3f}, nt: {nt}, nf: {nf}")

# Network
if args.pretrain | args.test:
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=True, device=device, contrastive_learning=args.contrastive_learning, num_utterances=args.num_utterances_cl, contrastive_learning_data_aug=args.contrastive_learning_data_aug, obt=args.obt, hsic=args.hsic, feature_sim_loss=args.feature_sim_loss, feature_sim_loss_weight=args.feature_sim_loss_weight, feature_sim_sigma=args.feature_sim_sigma, feature_sim_temperature=args.feature_sim_temperature, hierarchical_feature_sim_room_temperature=args.hierarchical_feature_sim_room_temperature, hierarchical_feature_sim_room_sigma=args.hierarchical_feature_sim_room_sigma, hierarchical_feature_sim_doa_temperature=args.hierarchical_feature_sim_doa_temperature, hierarchical_feature_sim_doa_sigma=args.hierarchical_feature_sim_doa_sigma, nepoch=args.nepoch, pretrain_mode=args.pretrain_mode, spec_dembed=args.spec_dembed, spat_dembed=args.spat_dembed, spectral_enc_input=args.spectral_enc_input, debug=args.debug, hierarchical=args.hierarchical, rec_loss=args.rec_loss)
if args.pretrain_frozen_encoder:
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=False, device=device, pretrain_frozen_encoder=True, spec_dembed=args.spec_dembed, spat_dembed=args.spat_dembed, spectral_enc_input=args.spectral_enc_input)

layer_keys = ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head','spec_encoder.patch_embed','spec_encoder.embed','spat_encoder.patch_embed','spat_encoder.embed']
nparam, nparam_sum = get_nparams(net, param_key_list=layer_keys)
print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 2)) for key in nparam.keys()])
nreim = 2
flops_forward_eval, _ = get_FLOPs(net, input_shape=(2, nmic, nf, nt, nreim), duration=T)
print(f"FLOPs_forward: {flops_forward_eval/2:.2f}G/s")

# Pre-Train
if args.pretrain:
	
	print('Pre-Training stage!')
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	# Dataset
	if args.simu_exp:
		if args.num_train_samples is not None:
			train_count = args.num_train_samples
			print(f"Using provided num_train_samples: {train_count}")
		else:
			train_count = count_wav_files(dirs['micsig_simu_pretrain'])
		val_count = count_wav_files(dirs['micsig_simu_preval'])
		# For test, we might use pretest_ins or just pretest dir. 
		# dataset.py uses pretest dir for 'all' mode and pretest_ins list for 'ins' mode.
		# run_pretrain.py logic for test isn't fully unified in pretrain block, 
		# but let's assume standard pretest dir for count if needed.
		# However, `data_num` keys are 'train', 'val', 'test'.
		test_count = count_wav_files(dirs['micsig_simu_pretest'])
	else:
		train_count = count_wav_files(dirs['micsig_real_pretrain'])
		val_count = count_wav_files(dirs['micsig_real_preval'])
		test_count = count_wav_files(dirs['micsig_real_pretest'])

	print(f"Dynamic Data Counts - Train: {train_count}, Val: {val_count}, Test: {test_count}")
	data_num = {'train': train_count, 'val': val_count, 'test': test_count}

	
	history = {'loss_train': []}
	if args.simu_exp:
		history['loss_val_sim'] = []
	else:
		history['loss_val_real'] = []
		history['loss_test_locata'] = []
		history['loss_test_ace'] = []

	if args.contrastive_learning or args.contrastive_learning_data_aug:
		history['loss_cl_spat_train'] = []
		history['loss_cl_spec_train'] = []
		if args.simu_exp:
			history['loss_cl_spat_val_sim'] = []
			history['loss_cl_spec_val_sim'] = []
		else:
			history['loss_cl_spat_val_real'] = []
			history['loss_cl_spec_val_real'] = []

	if args.obt:
		history['loss_obt_train'] = []
		if args.simu_exp:
			history['loss_obt_val_sim'] = []
		else:
			history['loss_obt_val_real'] = []

	if args.feature_sim_loss:
		history['loss_feat_sim_train'] = []
		if args.simu_exp:
			history['loss_feat_sim_val_sim'] = []
		else:
			history['loss_feat_sim_val_real'] = []

	remove_spkoverlap = True
	prob_mode = ['duration', 'micpair']
	if args.simu_exp:
		dataset_pretrain = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_pretrain'], 
			load_anno=False, 
			load_dp=False,
			fs = fs,
			dataset_sz=data_num['train'],
			transforms = None,
            sort=args.contrastive_learning or args.contrastive_learning_data_aug
			)
		dataset_preval = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_preval'], 
			load_anno=False, 
			load_dp=False,
			fs = fs,
			dataset_sz=data_num['val'],
			transforms = None,
			)
	else:
		dataset_list_train = ['LOCATA', 'RealMAN', # FIXME 'LOCATA', 'MCWSJ', 'LibriCSS', 'AMI', 'AISHELL4', 'M2MeT', 'RealMAN', # RealSig
							  'DCASE', 'Mesh', 'ACE', 'dEchorate', 'BUTReverb'] # FIXME , 'MIR' RealRIR
		dataset_probs_train = [1, 15, # FIXME 1, 5, 5, 8, 8, 8, 15,
							   5, 5, 5, 5, 5] # FIXME 5, 5, 5, 5, 5, 5

		dataset_pretrain = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_pretrain'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'train',
			seed = seeds['train'],
			dataset_sz = data_num['train'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = dataset_list_train,
			dataset_probs = dataset_probs_train,
			sound_speed = speed) 

		dataset_list_val = ['RealMAN', # FIXME 'AISHELL4', 'M2MeT',  RealSig
							'DCASE', 'BUTReverb'] 			# RealRIR
		dataset_probs_val = [5,  # FIXME 1, 2, 5,
							 2, 1]
		dataset_preval_real = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_preval'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'val',
			seed = seeds['val'],
			dataset_sz = data_num['val'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = dataset_list_val,
			dataset_probs = dataset_probs_val,
			sound_speed = speed)

		dataset_pretest_locata = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_pretest'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'test',
			seed = seeds['test'],
			dataset_sz = data_num['test'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = ['LOCATA'],
			dataset_probs = [1],
			sound_speed = speed)
		
		dataset_pretest_ace = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_pretest'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'test',
			# seed = seeds['test'],
			dataset_sz = data_num['test'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = ['ACE'],
			dataset_probs = [1],
			sound_speed = speed)

	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}

	if args.contrastive_learning:
		batch_sampler = ContrastiveBatchSampler(
			num_rooms=args.num_rooms_cl,
			num_utterances=args.num_utterances_cl,
			batch_size=args.bs[0],
			rooms_per_batch=8,
			utterances_per_batch=16
		)
		dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_sampler=batch_sampler, **kwargs)
	elif args.contrastive_learning_data_aug:
		batch_sampler = ContrastiveAugmentationBatchSampler(
			total_samples=data_num['train'],
			batch_size=args.bs[0],
			augmentations_per_sample=8,
            n_distinct_samples=args.cl_n_distinct
		)
		dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_sampler=batch_sampler, **kwargs)
	else:
		dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)

	if args.simu_exp:
		dataloader_preval_sim = torch.utils.data.DataLoader(dataset=dataset_preval, batch_size=args.bs[1], shuffle=False, **kwargs)
	else:
		dataloader_preval_real = torch.utils.data.DataLoader(dataset=dataset_preval_real, batch_size=args.bs[1], shuffle=False, **kwargs)
		dataloader_pretest_locata = torch.utils.data.DataLoader(dataset=dataset_pretest_locata, batch_size=args.bs[2], shuffle=False, **kwargs)
		dataloader_pretest_ace = torch.utils.data.DataLoader(dataset=dataset_pretest_ace, batch_size=args.bs[2], shuffle=False, **kwargs)

	# Learner
	learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=None, ch_mode='M')
	if args.debug:
		debug_log_path = os.path.join(dirs['log_pretrain'], 'debug.log')
		learner.set_debug(True, debug_log_path)
		print(f"Debug mode enabled. Logging to {debug_log_path}")
	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	if args.checkpoint_start:
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=True, as_all_state=True) # Train from latest checkpoints
	if args.checkpoint_from_best_epoch:
		# list(Path(dirs['log_pretrain']).glob(f"model*.tar"))
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=False, as_all_state=True) # Train from best checkpoints
		print(learner.start_epoch)
		max_epoch = 10
		for epoch in range(learner.start_epoch, learner.start_epoch+max_epoch):
			old_name = dirs['log_pretrain'] + '/model' + str(epoch) + '.tar' 
			new_name = dirs['log_pretrain'] + '/model' + str(epoch) + '_.tar' 
			if os.path.exists(old_name):
				print(epoch)
				os.rename(old_name, new_name)

	# Learning rate
	lr_schedule = create_learning_rate_schedule(total_steps=args.nepoch, base = args.lr, decay_type='cosine', warmup_steps=1, linear_end=1e-6)

	# Tensorboard
	train_writer = SummaryWriter(dirs['log_pretrain'] + '/train/', 'train')
	if args.simu_exp:
		val_sim_writer = SummaryWriter(dirs['log_pretrain'] + '/val_sim/', 'val')
	else:
		val_real_writer = SummaryWriter(dirs['log_pretrain'] + '/val_real/', 'val')
		test_locata_writer = SummaryWriter(dirs['log_pretrain'] + '/test_locata/', 'test')
		test_ace_writer = SummaryWriter(dirs['log_pretrain'] + '/test_ace/', 'test')

	# Calculate global feature statistics
	feature_stats = calculate_feature_stats(dataloader_pretrain, device, learner, fs)
	model_core = getattr(learner.model, 'module', learner.model)
	model_core.feature_stats = feature_stats
	print("Updated model with feature stats.")

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
 
		# lr = set_learning_rate(epoch=epoch, lr_init=args.lr, step=100, gamma=0.6)
		if args.simu_exp:
			lr = lr_schedule(epoch)	
		else:
			lr = 0.0001
			
		set_random_seed(seeds['train']+epoch)
		loss_train, diff_train, data_vis_train, loss_cl_spat_train, loss_cl_spec_train, obt_train, loss_gen_train, loss_feat_sim_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		if args.simu_exp:
			set_random_seed(seeds['val']) # can be fixed when there is one card, where therea are multiple cards, inputs are fix but network are not
			loss_val_sim, diff_val_sim, data_vis_val_sim, loss_cl_spat_val_sim, loss_cl_spec_val_sim, cka_val_sim, obt_val_sim, loss_gen_val_sim, loss_feat_sim_val_sim, _, _ = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
			print('Val loss sim: {:.4f}'.format(loss_val_sim) )
		else:
			set_random_seed(seeds['val']) 
			loss_val_real, diff_val_real, data_vis_val_real, loss_cl_spat_val_real, loss_cl_spec_val_real, cka_val_real, obt_val_real, loss_gen_val_real, loss_feat_sim_val_real, _, _ = learner.pretest_epoch(dataloader_preval_real, return_diff=True)
			set_random_seed(seeds['test'])
			loss_test_locata, diff_test_locata, data_vis_test_locata, _, _, _, _, loss_gen_test_locata, _, _, _ = learner.pretest_epoch(dataloader_pretest_locata, return_diff=True)
			set_random_seed(seeds['test'])
			loss_test_ace, diff_test_ace, data_vis_test_ace, _, _, _, _, loss_gen_test_ace, _, _, _ = learner.pretest_epoch(dataloader_pretest_ace, return_diff=True)

			print('Val loss real: {:.4f}'.format(loss_val_real) )
			print('Test loss locata: {:.4f}'.format(loss_test_locata) )
			print('Test loss ace: {:.4f}'.format(loss_test_ace) )

		# Save model
		# is_best_epoch = learner.is_best_epoch(current_score=loss_val_real*(-1))
		early_stop_patience = 100
		if args.simu_exp:
			current_score = loss_val_sim*(-1)
		else:
			current_score = loss_val_real*(-1)
		stop_flag, is_best_epoch = learner.early_stopping(current_score=current_score, patience=early_stop_patience)
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain'], is_best_epoch = is_best_epoch, save_extra_hist=True)

		# Monitor parameters with tensorboard
		train_writer.add_scalar('lr', lr, epoch)

		train_writer.add_scalar('loss', loss_train, epoch)
		if args.simu_exp:
			val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
		else:
			val_real_writer.add_scalar('loss', loss_val_real, epoch)
			test_locata_writer.add_scalar('loss', loss_test_locata, epoch)
			test_ace_writer.add_scalar('loss', loss_test_ace, epoch)
		
		train_writer.add_scalar('diff', diff_train, epoch) 
		if args.simu_exp:
			val_sim_writer.add_scalar('diff', diff_val_sim, epoch)
		else:
			val_real_writer.add_scalar('diff', diff_val_real, epoch)
			test_locata_writer.add_scalar('diff', diff_test_locata, epoch)
			test_ace_writer.add_scalar('diff', diff_test_ace, epoch)

		if epoch == 1:
			train_writer.add_scalar('nparam', nparam_sum, epoch)

		if args.wandb:
			log_dict = {
				'train/loss': loss_train,
				'train/lr': lr,
				'train/diff': diff_train,
				'train/loss_cl_spat': loss_cl_spat_train,
				'train/loss_cl_spec': loss_cl_spec_train,
			}
			if args.rec_loss:
				log_dict['train/loss_recon'] = loss_gen_train
			if args.obt:
				log_dict['train/obt'] = obt_train
			
			if args.feature_sim_loss:
				log_dict['train/loss_feat_sim'] = loss_feat_sim_train
			
			if args.simu_exp:
				log_dict.update({
					'val_sim/loss': loss_val_sim,
					'val_sim/diff': diff_val_sim,
					'val_sim/loss_cl_spat': loss_cl_spat_val_sim,
					'val_sim/loss_cl_spec': loss_cl_spec_val_sim,
				})
				if args.rec_loss:
					log_dict['val_sim/loss_recon'] = loss_gen_val_sim
				if args.obt:
					log_dict['val_sim/obt'] = obt_val_sim
			else:
				log_dict.update({
					'val_real/loss': loss_val_real,
					'val_real/diff': diff_val_real,
					'val_real/loss_cl_spat': loss_cl_spat_val_real,
					'val_real/loss_cl_spec': loss_cl_spec_val_real,
				})
				if args.rec_loss:
					log_dict['val_real/loss_recon'] = loss_gen_val_real

				if args.obt:
					log_dict['val_real/obt'] = obt_val_sim
				
				log_dict.update({
					'test_locata/loss': loss_test_locata,
					'test_locata/diff': diff_test_locata,
					'test_ace/loss': loss_test_ace,
					'test_ace/diff': diff_test_ace
				})
				if args.rec_loss:
					log_dict.update({
						'test_locata/loss_recon': loss_gen_test_locata,
						'test_ace/loss_recon': loss_gen_test_ace,
					})
			
			wandb.log(log_dict, step=epoch)

		# Save 
		nepoch_save_data = [5, 10, 15, 20, 25, 30, 35, 40]
		if epoch in nepoch_save_data:
			data_path = dirs['log_pretrain'] + '/result/'
			exist_flag = os.path.exists(data_path)
			if exist_flag==False:
				os.makedirs(data_path)
			data_vis_train_filtered = {k: v for k, v in data_vis_train.items() if k in ['mask', 'pred', 'tar']}
			if not args.pretrain_mode == 'autoenc':
				vis_train = vis_time_fre_data(data_vis_train_filtered, ins_idx=1)
				vis_train.savefig(data_path + str(epoch) + '_train')
		
		# Update history and plot learning curve
		history['loss_train'].append(loss_train)
		if args.simu_exp:
			history['loss_val_sim'].append(loss_val_sim)
		else:
			history['loss_val_real'].append(loss_val_real)
			history['loss_test_locata'].append(loss_test_locata)
			history['loss_test_ace'].append(loss_test_ace)

		if args.contrastive_learning or args.contrastive_learning_data_aug:
			history['loss_cl_spat_train'].append(loss_cl_spat_train)
			history['loss_cl_spec_train'].append(loss_cl_spec_train)
			if args.simu_exp:
				history['loss_cl_spat_val_sim'].append(loss_cl_spat_val_sim)
				history['loss_cl_spec_val_sim'].append(loss_cl_spec_val_sim)
			else:
				history['loss_cl_spat_val_real'].append(loss_cl_spat_val_real)
				history['loss_cl_spec_val_real'].append(loss_cl_spec_val_real)

		if args.obt:
			history['loss_obt_train'].append(obt_train)
			if args.simu_exp:
				history['loss_obt_val_sim'].append(obt_val_sim)
			else:
				history['loss_obt_val_real'].append(obt_val_real)
		
		if args.feature_sim_loss:
			history['loss_feat_sim_train'].append(loss_feat_sim_train)

		plot_learning_curve(history, dirs['log_pretrain'] + '/learning_curve.png')

		if stop_flag:
			break

	print('\nPre-Training finished\n')


if (args.pretrain_frozen_encoder):
	print('Frozen encoders and continue pre-training!')
	assert args.simu_exp == True, 'Frozen encoder mode only for simulated data'
	set_random_seed(args.seed)
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	# Dataset
	# Dataset
	if args.simu_exp:
		train_count = count_wav_files(dirs['micsig_simu_pretrain'])
		val_count = count_wav_files(dirs['micsig_simu_preval'])
		test_count = count_wav_files(dirs['micsig_simu_pretest'])
	else:
		# Frozen encoder essentially assumes simu_exp according to assert above, but for completeness:
		train_count = count_wav_files(dirs['micsig_real_pretrain'])
		val_count = count_wav_files(dirs['micsig_real_preval'])
		test_count = count_wav_files(dirs['micsig_real_pretest'])

	print(f"Dynamic Data Counts (Frozen) - Train: {train_count}, Val: {val_count}, Test: {test_count}")
	data_num = {'train': train_count, 'val': val_count, 'test': test_count}


	history = {'loss_train': [], 'loss_val_sim': []}
	if args.contrastive_learning or args.contrastive_learning_data_aug:
		history['loss_cl_spat_train'] = []
		history['loss_cl_spec_train'] = []
		history['loss_cl_spat_val_sim'] = []
		history['loss_cl_spec_val_sim'] = []

	dataset_pretrain = at_dataset.FixMicSigDataset( 
		data_dir = dirs['micsig_simu_pretrain'], 
		load_anno=False, 
		load_dp=False,
		fs = fs,
		dataset_sz=data_num['train'],
		transforms = None,
		)
	dataset_preval = at_dataset.FixMicSigDataset( 
		data_dir = dirs['micsig_simu_preval'], 
		load_anno=False, 
		load_dp=False,
		fs = fs,
		dataset_sz=data_num['val'],
		transforms = None,
		)

	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	dataloader_preval_sim = torch.utils.data.DataLoader(dataset=dataset_preval, batch_size=args.bs[1], shuffle=False, **kwargs)

	# Learner
	learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=None, ch_mode='M')
	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	if args.checkpoint_start:
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain_frozen_encoder'], from_latest=True, as_all_state=True) # Train from latest checkpoints
	learner.load_checkpoint_best(checkpoints_dir=dirs['log_pretrain'], as_all_state=False, param_frozen=False) # Train from pretrained frozen encoder 
	
	# Learning rate
	lr_schedule = create_learning_rate_schedule(total_steps=args.nepoch, base = args.lr, decay_type='cosine', warmup_steps=1, linear_end=1e-6)

	# Fix partial parameters
	# fix_key = 'spec_encoder'
	fix_key = 'encoder'
	cnt = 0
	for key, value in learner.model.named_parameters():
		if fix_key in key:
			cnt = cnt+1
			value.requires_grad = False
	print('# matched keys:', cnt)

	# TensorboardX
	train_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/train/', 'train')
	val_sim_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/val_sim/', 'val')

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		
		lr = lr_schedule(epoch)
		set_random_seed(epoch)
		loss_train, diff_train, data_vis_train, loss_cl_spat_train, loss_cl_spec_train, _, loss_gen_train, _ = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		set_random_seed(args.seed+10001)
		loss_val_sim, diff_val_sim, data_vis_val_sim, loss_cl_spat_val_sim, loss_cl_spec_val_sim, cka_val_sim, _, loss_gen_val_sim, _, _, _ = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
		print('Val loss sim: {:.4f}'.format(loss_val_sim) )

		# Update history and plot learning curve
		history['loss_train'].append(loss_train)
		history['loss_val_sim'].append(loss_val_sim)

		if args.contrastive_learning or args.contrastive_learning_data_aug:
			history['loss_cl_spat_train'].append(loss_cl_spat_train)
			history['loss_cl_spec_train'].append(loss_cl_spec_train)
			history['loss_cl_spat_val_sim'].append(loss_cl_spat_val_sim)
			history['loss_cl_spec_val_sim'].append(loss_cl_spec_val_sim)
		
		plot_learning_curve(history, dirs['log_pretrain_frozen_encoder'] + '/learning_curve.png')

		# Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_val_sim*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain_frozen_encoder'], is_best_epoch = is_best_epoch, save_extra_hist=False)

		# Visualize parameters with tensorboardX
		train_writer.add_scalar('lr', lr, epoch)
		train_writer.add_scalar('loss', loss_train, epoch)
		train_writer.add_scalar('diff', diff_train, epoch)
		if epoch == 1:
			train_writer.add_scalar('nparam', nparam_sum, epoch)

		val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
		val_sim_writer.add_scalar('diff', diff_val_sim, epoch)

		if args.wandb:
			log_dict = {
				'train/loss': loss_train,
				'train/loss_recon': loss_gen_train,
				'train/lr': lr,
				'train/diff': diff_train,
				'train/loss_cl_spat': loss_cl_spat_train,
				'train/loss_cl_spec': loss_cl_spec_train,
				'val_sim/loss': loss_val_sim,
				'val_sim/loss_recon': loss_gen_val_sim,
				'val_sim/diff': diff_val_sim,
				'val_sim/loss_cl_spat': loss_cl_spat_val_sim,
				'val_sim/loss_cl_spec': loss_cl_spec_val_sim
			}
			wandb.log(log_dict, step=epoch)

	print('\nFrozen Pre-Training finished\n')

# Test
if (args.test):
	print('Test data from ', dirs['micsig_simu_pretest'])
	assert args.simu_exp == True, 'Test mode only for simulated data'
 
	kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

	# learner
	learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=None, ch_mode='M')
	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	epoch = learner.load_checkpoint_best(checkpoints_dir=dirs['log_pretrain'], as_all_state=True) # best checkpoints
	
	if args.test_mode == 'all':
		set_seed(args.seed)
		
		# Dataset
		dataset_pretest = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_pretest'], 
			load_anno=False, 
			load_dp=False,
			fs=fs,
			dataset_sz=4000,
			transforms = None,
			)
		dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=args.bs[2], shuffle=False, **kwargs)

		# Update model with test data stats for visualization purposes
		feature_stats = calculate_feature_stats(dataloader_pretest, device, learner, fs)
		model_core = getattr(learner.model, 'module', learner.model)
		model_core.feature_stats = feature_stats
		print("Updated model with feature stats from Test Data.")

		# Test
		loss_test, diff_test, data_vis_test, _, _, cka_test, _, loss_gen_test, _, rank_analysis, metrics_data = learner.pretest_epoch(dataloader_pretest, return_diff=True, analyze_rank=True, gen_metrics=args.gen_metrics)

		# Gen Metrics Histograms
		if args.gen_metrics and metrics_data:
			metrics_dir = os.path.join(dirs['log_pretrain'], 'metrics')
			os.makedirs(metrics_dir, exist_ok=True)
			
			def plot_metrics(m_data, prefix=''):
				# Helper to plot raw, norm, sim_weights, cue_dist
				# 1. Raw features
				if 'raw' in m_data:
					for k, v_list in m_data['raw'].items():
						data_cat = torch.cat(v_list).numpy()
						plt.figure()
						plt.hist(data_cat, bins=50)
						plt.title(f'{prefix}Raw {k}')
						plt.savefig(os.path.join(metrics_dir, f'{prefix}raw_{k}.png'))
						plt.close()
					
				# 2. Normalized features
				if 'norm' in m_data:
					for k, v_list in m_data['norm'].items():
						data_cat = torch.cat(v_list).numpy()
						plt.figure()
						plt.hist(data_cat, bins=50)
						plt.title(f'{prefix}Normalized {k}')
						plt.savefig(os.path.join(metrics_dir, f'{prefix}norm_{k}.png'))
						plt.close()

				# 3. Similarity Weights
				if 'sim_weights' in m_data:
					w_list = m_data['sim_weights']
					w_all = torch.cat([x.flatten() for x in w_list]).numpy()
					plt.figure()
					plt.hist(w_all, bins=50)
					plt.title(f'{prefix}Similarity Weights (w) - Global')
					plt.savefig(os.path.join(metrics_dir, f'{prefix}sim_weights_global.png'))
					plt.close()
					
					# Anchor Sim Weights
					if len(w_list) > 0:
						w_anchor = w_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(w_anchor, bins=50)
						plt.title(f'{prefix}Similarity Weights (w) - Anchor')
						plt.savefig(os.path.join(metrics_dir, f'{prefix}sim_weights_anchor.png'))
						plt.close()

				# 4. Cue Dist
				if 'cue_dist' in m_data:
					cd_list = m_data['cue_dist']
					cd_all = torch.cat([x.flatten() for x in cd_list]).numpy()
					plt.figure()
					plt.hist(cd_all, bins=50)
					plt.title(f'{prefix}Cue Distance - Global')
					plt.savefig(os.path.join(metrics_dir, f'{prefix}cue_dist_global.png'))
					plt.close()
					
					# Anchor Cue Dist
					if len(cd_list) > 0:
						cd_anchor = cd_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(cd_anchor, bins=50)
						plt.title(f'{prefix}Cue Distance - Anchor')
						plt.savefig(os.path.join(metrics_dir, f'{prefix}cue_dist_anchor.png'))
						plt.close()

				# 5. Sim
				if 'sim' in m_data:
					sim_list = m_data['sim']
					sim_all = torch.cat([x.flatten() for x in sim_list]).numpy()
					plt.figure()
					plt.hist(sim_all, bins=50)
					plt.title(f'{prefix}Similarity (sim) - Global')
					plt.savefig(os.path.join(metrics_dir, f'{prefix}sim_global.png'))
					plt.close()
					
					# Anchor Sim
					if len(sim_list) > 0:
						sim_anchor = sim_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(sim_anchor, bins=50)
						plt.title(f'{prefix}Similarity (sim) - Anchor')
						plt.savefig(os.path.join(metrics_dir, f'{prefix}sim_anchor.png'))
						plt.close()

			# Check if hierarchical (keys like 'doa', 'room') or flat
			# We assume if 'raw' is not in top level, it might be nested
			# Generic hierarchical handling
			def handle_metrics_recursive(data, prefix=''):
				if 'raw' in data:
					plot_metrics(data, prefix)
				else:
					# This is a branch
					for sub_key, sub_data in data.items():
						if isinstance(sub_data, dict):
							handle_metrics_recursive(sub_data, prefix=f"{prefix}{sub_key}_")
			
			handle_metrics_recursive(metrics_data)
			
			print(f"Saved metric histograms to {metrics_dir}")

        # Plot Effective Rank Analysis
		if rank_analysis:
			plt.figure(figsize=(10, 6))
			thresholds = [0.7, 0.9, 0.99]
			stats_text = []

			if rank_analysis['cumulative_variance_spat'] is not None:
				spat_var = np.array(rank_analysis['cumulative_variance_spat'])
				plt.plot(spat_var, label='Spatial (Room)')
				spat_dims = []
				for t in thresholds:
					idx = np.where(spat_var >= t)[0]
					dim = idx[0] + 1 if len(idx) > 0 else len(spat_var)
					spat_dims.append(dim)
				stats_text.append(f"Spat: 70%@{spat_dims[0]}, 90%@{spat_dims[1]}, 99%@{spat_dims[2]}")

			if rank_analysis['cumulative_variance_spec'] is not None:
				spec_var = np.array(rank_analysis['cumulative_variance_spec'])
				plt.plot(spec_var, label='Spectral (Source)')
				spec_dims = []
				for t in thresholds:
					idx = np.where(spec_var >= t)[0]
					dim = idx[0] + 1 if len(idx) > 0 else len(spec_var)
					spec_dims.append(dim)
				stats_text.append(f"Spec: 70%@{spec_dims[0]}, 90%@{spec_dims[1]}, 99%@{spec_dims[2]}")

			for t in thresholds:
				plt.axhline(y=t, color='gray', linestyle='--', alpha=0.3)

			plt.ylabel('Cumulative Explained Variance')
			plt.xlabel('Number of Dimensions')
			plt.legend()
			plt.grid(True)
			plt.title('Effective Rank Analysis\n' + ' | '.join(stats_text))
			plt.tight_layout()
			plot_path = os.path.join(dirs['log_pretrain'], 'effective_rank_test.png')
			plt.savefig(plot_path)
			plt.close()
			# print(f"Saved effective rank plot to {plot_path}")

		print('Test loss: {:.4f}'.format(loss_test) )
		print('Test CKA: {:.4f}'.format(cka_test) )
		
		# Save results to json
		res_file = os.path.join(dirs['log_pretrain'], 'pretext_test_res.json')
		res_data = {'loss_test': loss_test, 'cka_test': cka_test, 'n_params': nparam_sum}
		try:
			with open(res_file, 'w') as f:
				json.dump(res_data, f, indent=4)
			print(f"Saved test results to {res_file}")

			if args.wandb:
				wandb.log({
					'test/loss': loss_test,
					'test/loss_recon': loss_gen_test,
					'test/cka': cka_test
				})
		except Exception as e:
			print(f"Failed to save test results: {e}")

	elif args.test_mode == 'ins':
		for dir_pretest in dirs['micsig_simu_pretest_ins']:
			set_seed(args.seed)
			
			# Dataset
			dataset_pretest = at_dataset.FixMicSigDataset( 
				data_dir = dirs['micsig_simu_pretest_ins'],
				load_anno=False, 
				load_dp=True,
				dataset_sz=None,
				fs=fs,
				transforms=None)
			dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=len(dataset_pretest), shuffle=False, **kwargs)

			# Test
			loss_test, diff_test, data_vis_test, results_test = learner.pretest_epoch(dataloader_pretest, return_diff=True, return_eval=True)
			# Only doing rank analysis for 'all' mode to average over enough samples

			name_pretest = dir_pretest.split('/')[-1]
			print(name_pretest, 'Test loss: {:.4f}'.format(loss_test) )
	
			data_path = dirs['log_pretrain'] + '/test_result/'
			exist_flag = os.path.exists(data_path)
			if exist_flag==False:
				os.makedirs(data_path)
 
			rt = dir_pretest.split('T')[-1]
			keys = data_vis_test.keys()
			for key in keys:
				print(key)
			for ins_idx in range(len(dataset_pretest)):
				vis_test = vis_time_fre_data(data_vis_test, ins_idx=ins_idx)
				vis_test.savefig(data_path + 'rt'+ rt +'_ins' + str(ins_idx) + '_epoch' + str(epoch) + '_test.png')
				soundfile.write(data_path + 'rt'+ rt + '_ins' + str(ins_idx) + '_epoch' + str(epoch) + '_test_pred.wav', results_test['sig_pred'][ins_idx, :, :].cpu().detach().numpy(), fs)
				soundfile.write(data_path + 'rt'+ rt + '_ins' + str(ins_idx) + '_epoch' + str(epoch) + '_test_tar.wav', results_test['sig_tar'][ins_idx, :, :].cpu().detach().numpy(), fs)

			scipy.io.savemat(data_path + 'rt'+ rt + '_ins' + '_epoch' + str(epoch) + '_test.mat', 
				{'mask': data_vis_test['mask'].cpu().detach().numpy(),
				'pred': data_vis_test['pred'].cpu().detach().numpy(),
				'tar': data_vis_test['tar'].cpu().detach().numpy(),
				# 'dp_tar': data_vis_test['dp_tar'].cpu().detach().numpy(),
				'pesq': results_test['pesq'].cpu().detach().numpy(),
				'pesq_mask_ch': results_test['pesq_mask_ch'].cpu().detach().numpy()})