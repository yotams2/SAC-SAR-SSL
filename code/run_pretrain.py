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
from common.utils_module import spatial_features_from_stft, spectral_features_from_stft

def calculate_feature_stats(dataloader, device, learner, fs, oracle_supcon=False):
	print("Calculating global feature statistics...")
	num_batches_for_std = 50
	
	feature_collections = {}
	
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			if i >= num_batches_for_std:
				break
			mic_sig_batch = data[0].to(device)
			
			gt_spatial_features = {}
			if len(data) > 2:
				for item in data[1:-1]:
					if isinstance(item, dict) and 'TDOA' in item:
						gt_spatial_features = {k: v.to(device).float() for k, v in item.items()}
						break
			
			stft = learner.stft(mic_sig_batch) 
			stft = stft.permute(0, 3, 1, 2)
			feats = spatial_features_from_stft(stft, fs, max_ipd_freq=1500.0)
			
			if oracle_supcon and gt_spatial_features:
				feats.update(gt_spatial_features)

			for key, val in feats.items():
				if isinstance(val, torch.Tensor) and val.dim() == 1 and not torch.is_complex(val):
					if key not in feature_collections:
						feature_collections[key] = []
					feature_collections[key].append(val)

	feature_stats = {}
	if len(feature_collections) > 0:
		for key, tensor_list in feature_collections.items():
			all_vals = torch.cat(tensor_list)
			mean = torch.mean(all_vals).item()
			std = torch.std(all_vals).item()
			feature_stats[key] = {'mean': mean, 'std': std}
			print(f"{key} Mean: {mean:.6f}, STD: {std:.6f}")
            
		feature_stats['family_medians'] = {}
		families = {'tdoa': ['tdoa'], 'room': ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max']}
		if oracle_supcon:
			families = {'tdoa': ['TDOA'], 'room': ['T60', 'DRR', 'C50', 'ABS']}
        
		for fam_name, feats in families.items():
			valid_feats = [f for f in feats if f in feature_collections]
			if len(valid_feats) > 0:
				c_family_list = []
				for key in valid_feats:
					val = torch.cat(feature_collections[key])
					stats = feature_stats[key]
					val_norm = (val - stats['mean']) / (3 * stats['std'] + 1e-6)
					val_norm = torch.clamp(val_norm, -1.0, 1.0)
					c_family_list.append(val_norm.unsqueeze(1))
				c_family = torch.cat(c_family_list, dim=1)
                
				N = c_family.shape[0]
				if N > 2000:
					subset_indices = torch.randperm(N)[:2000]
					c_family_subset = c_family[subset_indices]
				else:
					c_family_subset = c_family
                
				cue_dist = torch.cdist(c_family_subset, c_family_subset, p=2)
				B_sub = c_family_subset.shape[0]
				diag_mask_med = torch.eye(B_sub, device=cue_dist.device, dtype=torch.bool)
				off_diag_dist = cue_dist[~diag_mask_med]
				median_dist = off_diag_dist.median().item()
				feature_stats['family_medians'][fam_name] = median_dist
				print(f"Family {fam_name} Median Dist: {median_dist:.6f}")
	else:
		print("Warning: Could not calculate feature statistics.")
		
	return feature_stats

def calculate_spectral_feature_stats(dataloader, device, learner, fs):
	from tqdm import tqdm
	print("Calculating global spectral feature statistics...")
	feature_matrices = []
	num_batches_for_std = 50
	
	with torch.no_grad():
		try:
			total_batches = min(num_batches_for_std, len(dataloader))
		except TypeError:
			total_batches = num_batches_for_std
		
		for i, data in enumerate(tqdm(dataloader, total=total_batches, desc="Spectral Stats")):
			if i >= num_batches_for_std:
				break
			mic_sig_batch = data[0].to(device)
			
			if len(data) > 2 and isinstance(data[-2], dict) and 'feature_matrix' in data[-2]:
				spec_feats = data[-2]
				spec_feats = {k: v.to(device) for k,v in spec_feats.items()}
			else:
				stft = learner.stft(mic_sig_batch) 
				stft = stft.permute(0, 3, 1, 2) # [B, C, F, T]
				spec_feats = spectral_features_from_stft(stft, fs)
				
			feature_matrix = spec_feats['feature_matrix'] # [B, 16, T]
			voiced_prob = spec_feats['voiced_prob'] # [B, T]
			
			B, num_features, T = feature_matrix.shape
			
			# For F0 (index 0), only keep voiced frames (prob > 0.5)
			f0_vals = feature_matrix[:, 0, :]
			voiced_mask = voiced_prob > 0.5
			if voiced_mask.any():
				valid_f0 = f0_vals[voiced_mask]
			else:
				valid_f0 = torch.zeros(0, device=device)
				
			# Extract other features directly flattened
			other_feats = []
			for f_idx in range(1, num_features):
				other_feats.append(feature_matrix[:, f_idx, :].flatten())
			
			feature_matrices.append((valid_f0, other_feats))

	feature_stats = {}
	if len(feature_matrices) > 0:
		all_f0 = torch.cat([fm[0] for fm in feature_matrices])
		if all_f0.numel() > 0:
			feature_stats['mean'] = [torch.mean(all_f0).item()]
			feature_stats['std'] = [torch.std(all_f0).item()]
		else:
			feature_stats['mean'] = [0.0]
			feature_stats['std'] = [1e-6]
			
		for f_idx in range(1, 16):
			all_f = torch.cat([fm[1][f_idx-1] for fm in feature_matrices])
			feature_stats['mean'].append(torch.mean(all_f).item())
			feature_stats['std'].append(torch.std(all_f).item())
			
		print("Spectral Feature Stats Calculated")
	else:
		print("Warning: Could not calculate spectral feature statistics.")
	
	return feature_stats

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

# Save config file
if args.pretrain:
	os.makedirs(dirs['log_pretrain'], exist_ok=True)
	file_path = os.path.join(dirs['log_pretrain'],"config.json")
	import shlex
	command = 'python ' + ' '.join(shlex.quote(arg) for arg in sys.argv)
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
	fixed_fw = getattr(args, 'fixed_feature_weights_path', None) is not None
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=True, device=device, contrastive_learning=args.contrastive_learning, num_utterances=args.num_utterances_cl, contrastive_learning_data_aug=args.contrastive_learning_data_aug, obt=args.obt, hsic=args.hsic, feature_sim_loss=args.feature_sim_loss, diff_feat_selection=args.diff_feat_selection, diff_feat_selection_lambda=args.diff_feat_selection_lambda, feature_sim_loss_weight=args.feature_sim_loss_weight, feature_sim_sigma=args.feature_sim_sigma, feature_sim_temperature=args.feature_sim_temperature, hierarchical_feature_sim_room_temperature=args.hierarchical_feature_sim_room_temperature, hierarchical_feature_sim_room_sigma=args.hierarchical_feature_sim_room_sigma, hierarchical_feature_sim_doa_temperature=args.hierarchical_feature_sim_doa_temperature, hierarchical_feature_sim_doa_sigma=args.hierarchical_feature_sim_doa_sigma, spectral_feature_sim_loss=args.spectral_feature_sim_loss, spectral_feature_sim_weight=args.spectral_feature_sim_weight, spectral_feature_sim_sigma=args.spectral_feature_sim_sigma, spectral_feature_sim_temperature=args.spectral_feature_sim_temperature, nepoch=args.nepoch, pretrain_mode=args.pretrain_mode, spec_dembed=args.spec_dembed, spat_dembed=args.spat_dembed, spectral_enc_input=args.spectral_enc_input, debug=args.debug, hierarchical=args.hierarchical, hierarchical_w_stop_grad=getattr(args, 'hierarchical_w_stop_grad', False), hierarchical_w_ch_att=getattr(args, 'hierarchical_w_ch_att', False), hierarchical_stop_grad_branch=getattr(args, 'hierarchical_stop_grad_branch', 'room'), split_spatial_encoder=args.split_spatial_encoder, hierarchical_adv=args.hierarchical_adv, adv_weight=args.adv_weight, rec_loss=args.rec_loss, fixed_feature_weights=fixed_fw, factorized_sac=args.factorized_sac, use_pcgrad=args.use_pcgrad, normalization_mode=getattr(args, 'normalization_mode', 'z_norm'), oracle_supcon=getattr(args, 'oracle_supcon', False), use_cross_attention=getattr(args, 'use_cross_attention', False), family_specific_bn=getattr(args, 'family_specific_bn', False), use_pfa_features=getattr(args, 'use_pfa_features', False), use_pfa_features_w_hueristic=getattr(args, 'use_pfa_features_w_hueristic', False))
if args.pretrain_frozen_encoder:
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=False, device=device, pretrain_frozen_encoder=True, spec_dembed=args.spec_dembed, spat_dembed=args.spat_dembed, spectral_enc_input=args.spectral_enc_input, split_spatial_encoder=args.split_spatial_encoder, factorized_sac=args.factorized_sac, use_pcgrad=args.use_pcgrad, use_cross_attention=getattr(args, 'use_cross_attention', False), family_specific_bn=getattr(args, 'family_specific_bn', False))

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
			load_anno=getattr(args, 'oracle_supcon', False), 
			load_dp=False,
			fs = fs,
			dataset_sz=data_num['train'],
			transforms = None,
            sort=args.contrastive_learning or args.contrastive_learning_data_aug,
            load_spec_feats=args.spectral_feature_sim_loss
			)
		dataset_preval = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_preval'], 
			load_anno=getattr(args, 'oracle_supcon', False), 
			load_dp=False,
			fs = fs,
			dataset_sz=data_num['val'],
			transforms = None,
            load_spec_feats=args.spectral_feature_sim_loss
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
			sound_speed = speed,
            load_spec_feats=args.spectral_feature_sim_loss) 

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
			sound_speed = speed,
            load_spec_feats=args.spectral_feature_sim_loss)

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
			sound_speed = speed,
            load_spec_feats=args.spectral_feature_sim_loss)
		
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
			sound_speed = speed,
            load_spec_feats=args.spectral_feature_sim_loss)

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
	feature_stats = calculate_feature_stats(dataloader_pretrain, device, learner, fs, oracle_supcon=getattr(args, 'oracle_supcon', False))
	model_core = learner.model.module if hasattr(learner.model, 'module') else learner.model
	model_core.feature_stats = feature_stats
	
	if args.spectral_feature_sim_loss:
		spectral_feature_stats = calculate_spectral_feature_stats(dataloader_pretrain, device, learner, fs)
		model_core.spectral_feature_stats = spectral_feature_stats
		
	print("Updated model with feature stats.")

	if getattr(args, 'fixed_feature_weights_path', None):
		import json
		print(f"Loading fixed feature weights from {args.fixed_feature_weights_path}")
		with open(args.fixed_feature_weights_path, 'r') as f:
			imp_history = json.load(f)
		
		# Get the last epoch
		epochs = sorted([int(k.split('_')[1]) for k in imp_history.keys()])
		last_epoch_key = f"epoch_{epochs[-1]}"
		last_epoch_weights = imp_history[last_epoch_key]
		
		# Map the weights to the model's spatial_feature_weights
		# Based on user instruction: take the weights as-is from the json file.
		if hasattr(model_core, 'spatial_feature_weights') and hasattr(model_core, 'candidate_features'):
			new_weights = []
			for feat in model_core.candidate_features:
				val = last_epoch_weights.get(feat, 0.5)
				new_weights.append(val)
				
			model_core.spatial_feature_weights.data = torch.tensor(new_weights, dtype=torch.float32, device=device)
			model_core.spatial_feature_weights.requires_grad = False
			model_core.fixed_feature_weights = True
			print(f"Fixed feature weights assigned: {last_epoch_weights}")
		else:
			print("Warning: Model does not have spatial_feature_weights or candidate_features, skipping fixed weights.")

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
 
		# lr = set_learning_rate(epoch=epoch, lr_init=args.lr, step=100, gamma=0.6)
		if args.simu_exp:
			lr = lr_schedule(epoch)	
		else:
			lr = 0.0001
			
		set_random_seed(seeds['train']+epoch)
		loss_train, diff_train, data_vis_train, loss_cl_spat_train, loss_cl_spec_train, obt_train, loss_gen_train, loss_feat_sim_train, loss_spec_feat_sim_train, loss_adv_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		if args.simu_exp:
			set_random_seed(seeds['val']) # can be fixed when there is one card, where therea are multiple cards, inputs are fix but network are not
			loss_val_sim, diff_val_sim, data_vis_val_sim, loss_cl_spat_val_sim, loss_cl_spec_val_sim, cka_val_sim, obt_val_sim, loss_gen_val_sim, loss_feat_sim_val_sim, loss_spec_feat_sim_val_sim, loss_adv_val_sim, _, _ = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
			print('Val loss sim: {:.4f}'.format(loss_val_sim) )
		else:
			set_random_seed(seeds['val']) 
			loss_val_real, diff_val_real, data_vis_val_real, loss_cl_spat_val_real, loss_cl_spec_val_real, cka_val_real, obt_val_real, loss_gen_val_real, loss_feat_sim_val_real, loss_spec_feat_sim_val_real, loss_adv_val_real, _, _ = learner.pretest_epoch(dataloader_preval_real, return_diff=True)
			set_random_seed(seeds['test'])
			loss_test_locata, diff_test_locata, data_vis_test_locata, _, _, _, _, loss_gen_test_locata, _, _, _, _, _ = learner.pretest_epoch(dataloader_pretest_locata, return_diff=True)
			set_random_seed(seeds['test'])
			loss_test_ace, diff_test_ace, data_vis_test_ace, _, _, _, _, loss_gen_test_ace, _, _, _, _, _ = learner.pretest_epoch(dataloader_pretest_ace, return_diff=True)

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
			if 'factorized_losses_avg' in data_vis_train:
				for k, v in data_vis_train['factorized_losses_avg'].items():
					log_dict[f'train/sac_{k.replace("loss_", "")}'] = v
			if args.spectral_feature_sim_loss:
				log_dict['train/loss_spec_feat_sim'] = loss_spec_feat_sim_train
			if getattr(args, 'hierarchical_adv', False):
				log_dict['train/loss_adv'] = loss_adv_train
			
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
				if getattr(args, 'hierarchical_adv', False):
					log_dict['val_sim/loss_adv'] = loss_adv_val_sim
				if 'factorized_losses_avg' in data_vis_val_sim:
					for k, v in data_vis_val_sim['factorized_losses_avg'].items():
						log_dict[f'val_sim/sac_{k.replace("loss_", "")}'] = v
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
					log_dict['val_real/obt'] = obt_val_real
				if getattr(args, 'hierarchical_adv', False):
					log_dict['val_real/loss_adv'] = loss_adv_val_real
				if 'factorized_losses_avg' in data_vis_val_real:
					for k, v in data_vis_val_real['factorized_losses_avg'].items():
						log_dict[f'val_real/sac_{k.replace("loss_", "")}'] = v
				
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

		if getattr(args, 'diff_feat_selection', False):
			import json
			metrics_dir = os.path.join(dirs['log_pretrain'], 'metrics')
			os.makedirs(metrics_dir, exist_ok=True)
			
			importance_file = os.path.join(metrics_dir, 'features_importance.json')
			
			# Read existing
			if os.path.exists(importance_file):
				with open(importance_file, 'r') as f:
					imp_history = json.load(f)
			else:
				imp_history = {}
				
			# Add current epoch if the model has the weights enabled
			if hasattr(model_core, 'spatial_feature_weights'):
				# learned_importance = torch.sigmoid(model_core.spatial_feature_weights).detach().cpu().numpy().tolist()
				learned_importance = model_core.get_active_spatial_feature_weights().detach().cpu().numpy().tolist()
				# learned_importance = model_core.spatial_feature_weights.detach().cpu().numpy().tolist()
				epoch_data = dict(zip(model_core.candidate_features, learned_importance))
				imp_history[f"epoch_{epoch}"] = epoch_data
				
				with open(importance_file, 'w') as f:
					json.dump(imp_history, f, indent=4)

		plot_learning_curve(history, dirs['log_pretrain'] + '/learning_curve.png')

		if stop_flag:
			break

	print('\nPre-Training finished\n')

	if getattr(args, 'diff_feat_selection', False):
		import json
		import matplotlib.pyplot as plt
		metrics_dir = os.path.join(dirs['log_pretrain'], 'metrics')
		importance_file = os.path.join(metrics_dir, 'features_importance.json')
		
		if os.path.exists(importance_file):
			with open(importance_file, 'r') as f:
				imp_history = json.load(f)
			
			epochs = sorted([int(k.split('_')[1]) for k in imp_history.keys()])
			if len(epochs) > 0:
				features = list(imp_history[f"epoch_{epochs[0]}"].keys())
				
				plt.figure(figsize=(12, 8))
				cmap = plt.get_cmap('tab20')
				
				for i, feature in enumerate(features):
					values = [imp_history[f"epoch_{e}"].get(feature, 0) for e in epochs]
					final_val = values[-1]
					
					# Highlight features that are actually being used, fade out noise
					alpha = 1.0 if final_val > 0.1 else 0.3
					linewidth = 2.5 if final_val > 0.1 else 1.0
					
					plt.plot(epochs, values, label=f"{feature} (Final: {final_val:.2f})", 
							 color=cmap(i % 20), alpha=alpha, linewidth=linewidth)
				
				plt.title('Differentiable Feature Selection: Importance over Epochs')
				plt.xlabel('Epoch')
				plt.ylabel('Learned Importance')
				# plt.ylim(0, 1.05)
				plt.grid(True, linestyle='--', alpha=0.7)
				
				# Place legend completely outside the plot to avoid overlapping lines
				plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
				plt.tight_layout()
				
				plot_file = os.path.join(metrics_dir, 'features_importance_curve.png')
				plt.savefig(plot_file, dpi=300, bbox_inches='tight')
				plt.close()
				print(f"Saved feature importance plot to {plot_file}")


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
        load_spec_feats=args.spectral_feature_sim_loss
		)
	dataset_preval = at_dataset.FixMicSigDataset( 
		data_dir = dirs['micsig_simu_preval'], 
		load_anno=False, 
		load_dp=False,
		fs = fs,
		dataset_sz=data_num['val'],
		transforms = None,
        load_spec_feats=args.spectral_feature_sim_loss
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
		loss_train, diff_train, data_vis_train, loss_cl_spat_train, loss_cl_spec_train, _, loss_gen_train, _, _, _ = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		set_random_seed(args.seed+10001)
		loss_val_sim, diff_val_sim, data_vis_val_sim, loss_cl_spat_val_sim, loss_cl_spec_val_sim, cka_val_sim, _, loss_gen_val_sim, _, _, _, _, _ = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
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
			load_anno=getattr(args, 'oracle_supcon', False), 
			load_dp=False,
			fs=fs,
			dataset_sz=4000,
			transforms = None,
			load_spec_feats=args.spectral_feature_sim_loss
			)
		dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=args.bs[2], shuffle=False, **kwargs)

		# Update model with test data stats for visualization purposes
		feature_stats = calculate_feature_stats(dataloader_pretest, device, learner, fs, oracle_supcon=getattr(args, 'oracle_supcon', False))
		model_core = getattr(learner.model, 'module', learner.model)
		model_core.feature_stats = feature_stats
		if args.spectral_feature_sim_loss:
			spectral_feature_stats = calculate_spectral_feature_stats(dataloader_pretest, device, learner, fs)
			model_core.spectral_feature_stats = spectral_feature_stats

		print("Updated model with feature stats from Test Data.")

		# Test
		loss_test, diff_test, data_vis_test, _, _, cka_test, _, loss_gen_test, _, _, _, rank_analysis, metrics_data = learner.pretest_epoch(dataloader_pretest, return_diff=True, analyze_rank=True, gen_metrics=args.gen_metrics)

		# Gen Metrics Histograms
		if args.gen_metrics and metrics_data:
			spatial_metrics_dir = os.path.join(dirs['log_pretrain'], 'metrics')
			spectral_metrics_dir = os.path.join(dirs['log_pretrain'], 'spectral_metrics')
			os.makedirs(spatial_metrics_dir, exist_ok=True)
			os.makedirs(spectral_metrics_dir, exist_ok=True)
			
			def _flatten_list(nl):
				fl = []
				for el in nl:
					if isinstance(el, list):
						fl.extend(_flatten_list(el))
					else:
						fl.append(el)
				return fl

			def plot_metrics(m_data, out_dir, prefix=''):
				# Helper to plot raw, norm, sim_weights, cue_dist
				# 1. Raw features
				if 'raw' in m_data:
					for k, v_list in m_data['raw'].items():
						flat_v = _flatten_list(v_list)
						valid_v = [x for x in flat_v if x is not None]
						if len(valid_v) > 0:
							data_cat = torch.cat(valid_v).numpy()
							plt.figure()
							plt.hist(data_cat, bins=50)
							plt.title(f'{prefix}Raw {k}')
							plt.savefig(os.path.join(out_dir, f'{prefix}raw_{k}.png'))
							plt.close()
					
				# 2. Normalized features
				if 'norm' in m_data:
					for k, v_list in m_data['norm'].items():
						flat_v = _flatten_list(v_list)
						valid_v = [x for x in flat_v if x is not None]
						if len(valid_v) > 0:
							data_cat = torch.cat(valid_v).numpy()
							plt.figure()
							plt.hist(data_cat, bins=50)
							plt.title(f'{prefix}Normalized {k}')
							plt.savefig(os.path.join(out_dir, f'{prefix}norm_{k}.png'))
							plt.close()

				# 3. Similarity Weights
				if 'sim_weights' in m_data:
					w_list = _flatten_list(m_data['sim_weights'])
					valid_w_list = [x for x in w_list if x is not None]
					if len(valid_w_list) > 0:
						w_all = torch.cat([x.flatten() for x in valid_w_list]).numpy()
						plt.figure()
						plt.hist(w_all, bins=50)
						plt.title(f'{prefix}Similarity Weights (w) - Global')
						plt.savefig(os.path.join(out_dir, f'{prefix}sim_weights_global.png'))
						plt.close()
						
						# Anchor Sim Weights
						w_anchor = valid_w_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(w_anchor, bins=50)
						plt.title(f'{prefix}Similarity Weights (w) - Anchor')
						plt.savefig(os.path.join(out_dir, f'{prefix}sim_weights_anchor.png'))
						plt.close()
						
				# 3.5 Normalized Similarity Weights
				if 'sim_weights_norm' in m_data:
					wn_list = _flatten_list(m_data['sim_weights_norm'])
					valid_wn_list = [x for x in wn_list if x is not None]
					if len(valid_wn_list) > 0:
						wn_all = torch.cat([x.flatten() for x in valid_wn_list]).numpy()
						plt.figure()
						plt.hist(wn_all, bins=50)
						plt.title(f'{prefix}Normalized Target Weights (w_norm) - Global')
						plt.savefig(os.path.join(out_dir, f'{prefix}sim_weights_norm_global.png'))
						plt.close()
						
						# Anchor Sim Weights Norm
						wn_anchor = valid_wn_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(wn_anchor, bins=50)
						plt.title(f'{prefix}Normalized Target Weights (w_norm) - Anchor')
						plt.savefig(os.path.join(out_dir, f'{prefix}sim_weights_norm_anchor.png'))
						plt.close()

				# 4. Cue Dist
				if 'cue_dist' in m_data:
					cd_list = _flatten_list(m_data['cue_dist'])
					valid_cd_list = [x for x in cd_list if x is not None]
					if len(valid_cd_list) > 0:
						cd_all = torch.cat([x.flatten() for x in valid_cd_list]).numpy()
						plt.figure()
						plt.hist(cd_all, bins=50)
						plt.title(f'{prefix}Cue Distance - Global')
						plt.savefig(os.path.join(out_dir, f'{prefix}cue_dist_global.png'))
						plt.close()
						
						# Anchor Cue Dist
						cd_anchor = valid_cd_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(cd_anchor, bins=50)
						plt.title(f'{prefix}Cue Distance - Anchor')
						plt.savefig(os.path.join(out_dir, f'{prefix}cue_dist_anchor.png'))
						plt.close()

				# 5. Sim
				if 'sim' in m_data:
					sim_list = _flatten_list(m_data['sim'])
					valid_sim_list = [x for x in sim_list if x is not None]
					if len(valid_sim_list) > 0:
						sim_all = torch.cat([x.flatten() for x in valid_sim_list]).numpy()
						plt.figure()
						plt.hist(sim_all, bins=50)
						plt.title(f'{prefix}Similarity (sim) - Global')
						plt.savefig(os.path.join(out_dir, f'{prefix}sim_global.png'))
						plt.close()
						
						# Anchor Sim
						sim_anchor = valid_sim_list[0][0, :].numpy() # [B]
						plt.figure()
						plt.hist(sim_anchor, bins=50)
						plt.title(f'{prefix}Similarity (sim) - Anchor')
						plt.savefig(os.path.join(out_dir, f'{prefix}sim_anchor.png'))
						plt.close()

			# Check if hierarchical (keys like 'doa', 'room') or flat
			# We assume if 'raw' is not in top level, it might be nested
			# Generic hierarchical handling
			def handle_metrics_recursive(data, out_dir, prefix=''):
				if 'raw' in data:
					plot_metrics(data, out_dir, prefix)
				else:
					# This is a branch
					for sub_key, sub_data in data.items():
						if isinstance(sub_data, dict):
							handle_metrics_recursive(sub_data, out_dir, prefix=f"{prefix}{sub_key}_")
			
			if metrics_data.get('spatial'):
				handle_metrics_recursive(metrics_data['spatial'], spatial_metrics_dir)
			if metrics_data.get('spectral'):
				handle_metrics_recursive(metrics_data['spectral'], spectral_metrics_dir)
			
			print(f"Saved spatial metric histograms to {spatial_metrics_dir} and spectral metric histograms to {spectral_metrics_dir}")

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
				transforms=None,
				load_spec_feats=args.spectral_feature_sim_loss
			)
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