""" 
    Define some basic modules
"""

import random
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def complex_multiplication(x, y):
    return torch.stack([ x[...,0]*y[...,0] - x[...,1]*y[...,1],   x[...,0]*y[...,1] + x[...,1]*y[...,0]  ], dim=-1)


def complex_conjugate_multiplication(x, y):
    return torch.stack([ x[...,0]*y[...,0] + x[...,1]*y[...,1],   x[...,1]*y[...,0] - x[...,0]*y[...,1]  ], dim=-1)


# def complex_cart2polar(x):
#     mod = torch.sqrt( complex_conjugate_multiplication(x, x)[..., 0] )
#     phase = torch.atan2(x[..., 1], x[..., 0])
#     return torch.stack((mod, phase), dim=-1)


class STFT(nn.Module):
    """ Get STFT coefficients of microphone signals (batch processing by pytorch)
        Args:       win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    win             - window type 
                                      'boxcar': a rectangular window (equivalent to no window at all)
                                      'hann': a Hann window
					signal          - the microphone signals in time domain (nbatch, nsample, nch)
        Returns:    stft            - STFT coefficients (nbatch, nf, nt, nch)
    """

    def __init__(self, win_len, win_shift_ratio, nfft, win='hann', inv=False):
        super(STFT, self).__init__()

        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.win = win
        self.inv = inv

    def forward(self, signal):

        nsample = signal.shape[-2]
        nch = signal.shape[-1]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nf = int(self.nfft / 2) + 1

        nb = signal.shape[0]
        if self.inv:
            nt = int((nsample) / win_shift) + 1  # for iSTFT
        else:
            nt = np.floor((nsample - self.win_len) / win_shift + 1).astype(int)
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64).to(signal.device)

        if self.win == 'hann':
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        for ch_idx in range(0, nch, 1):
            if self.inv:
                stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft = self.nfft, hop_length = win_shift, win_length = self.win_len,
                window = window, center = True, normalized = False, return_complex = True)  # for iSTFT
            else:
                stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                    window=window, center=False, normalized=False, return_complex=True)
        return stft


class ISTFT(nn.Module):
    """ Get inverse STFT (batch processing by pytorch) 
		Args:		stft            - STFT coefficients (nbatch, nf, nt, nch)
					win_len         - the length of frame / window
					win_shift_ratio - the ratio between frame shift and frame length
					nfft            - the number of fft points
		Returns:	signal          - time-domain microphone signals (nbatch, nsample, nch)
	"""
    def __init__(self, win_len, win_shift_ratio, nfft, inv=False):
        super(ISTFT, self).__init__()

        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.inv = inv

    def forward(self, stft):
        # stft: nb, nf, nt, nch
        nf = stft.shape[-3]
        nt = stft.shape[-2]
        nch = stft.shape[-1]
        nb = stft.shape[0]
        win_shift = int(self.win_len * self.win_shift_ratio)
        if self.inv:
            nsample = (nt - 1) * win_shift
        else:
            nsample = (nt + 1) * win_shift #-1
        signal = torch.zeros((nb, nsample, nch)).to(stft.device)
        win_shift = int(self.win_len * self.win_shift_ratio)
        for ch_idx in range(0, nch, 1):
            if self.inv:
                signal_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                                                    center=True, normalized=False, return_complex=False)
                signal[:, :, ch_idx] = signal_temp[:, 0:nsample] # for STFT
            else:
                signal_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                                                    center=False, normalized=False, return_complex=False)
                signal[:, :, ch_idx] = signal_temp
        return signal


class AddChToBatch(nn.Module):
    """ Change dimension from  (nb, nch, ...) to (nb*(nch-1), ...) 
	"""
    def __init__(self, ch_mode):
        super(AddChToBatch, self).__init__()
        self.ch_mode = ch_mode
        assert self.ch_mode in ['MM', 'M', '1'], 'Unrecognized microphone channel mode~'

    def forward(self, data):
        nb = data.shape[0]
        nch = data.shape[1]

        if self.ch_mode == 'M':
            data_adjust = torch.zeros((nb*(nch-1),2)+data.shape[2:], dtype=torch.complex64).to(data.device) # (nb*(nch-1),2,nf,nt)
            for b_idx in range(nb):
                st = b_idx*(nch-1)
                ed = (b_idx+1)*(nch-1)
                data_adjust[st:ed, 0, ...] = data[b_idx, 0 : 1, ...].expand((nch-1,)+data.shape[2:])
                data_adjust[st:ed, 1, ...] = data[b_idx, 1 : nch, ...]

        elif self.ch_mode == 'MM':
            data_adjust = torch.zeros((nb*int((nch-1)*nch/2),2)+data.shape[2:], dtype=torch.complex64).to(data.device) # (nb*(nch-1)*nch/2,2,nf,nt)
            for b_idx in range(nb):
                for ch_idx in range(nch-1):
                    st = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx+1)*ch_idx/2)
                    ed = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx)*(ch_idx+1)/2)
                    data_adjust[st:ed, 0, ...] = data[b_idx, ch_idx:ch_idx+1, ...].expand((nch-ch_idx-1,)+data.shape[2:])
                    data_adjust[st:ed, 1, ...] = data[b_idx, ch_idx+1:, ...]
                    
        else:
            data_adjust = deepcopy(data)       

        return data_adjust.contiguous()

class RemoveChFromBatch(nn.Module):
	""" Change dimension from (nb*nmic, nt, nf) to (nb, nmic, nt, nf)
	"""
	def __init__(self, ch_mode):
		super(RemoveChFromBatch, self).__init__()
		self.ch_mode = ch_mode
		assert self.ch_mode in ['MM', 'M', '1'], 'Unrecognized microphone channel mode~'

	def forward(self, data, nb):
		if (self.ch_mode == 'MM') | (self.ch_mode == 'M'):
			nmic = int(data.shape[0]/nb)
			data_adjust = torch.zeros((nb, nmic)+data.shape[1:], dtype=torch.float32).to(data.device)
			for b_idx in range(nb):
				st = b_idx * nmic
				ed = (b_idx + 1) * nmic
				data_adjust[b_idx, ...] = data[st:ed, ...]
		else:
			data_adjust = deepcopy(data)

		return data_adjust.contiguous()



# %% Patch Split and Patch Mask

class PatchSplit(nn.Module):
    def __init__(self, patch_shape, f_first=False):
        super(PatchSplit, self).__init__()

        self.patch_shape = patch_shape
        self.f_first = f_first

    def forward(self, data):
		# data: (nbatch, nf, nt, nmic)/ (nbatch, nf, nt, nreim, nmic)

        if len(data.shape) == 4:
            nbatch, _, _, nmic = data.shape
            data = data.permute(0, 3, 1, 2) # (nbatch, nmic, nf, nt)
            if self.f_first:
                data = data.permute(0, 1, 3, 2) # (nbatch, nmic, nt, nf)
                vec = F.unfold(data, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]])
            else:
                vec = F.unfold(data, kernel_size=self.patch_shape, stride=self.patch_shape)  # (nbatch, nmic*dpatch, npatch) unfold first the last dim, then the second last, like conv layers
            vec = vec.reshape(nbatch, nmic, self.patch_shape[0] * self.patch_shape[1], vec.shape[-1])  # (nbatch, nmic, dpatch, npatch)
            vec = vec.permute(0, 3, 1, 2) # (nbatch, npatch, dpatch, nmic)

        elif len(data.shape) == 5:
            nbatch, nf, nt, nreim, nmic = data.shape
            data = data.permute(0, 3, 4, 1, 2).reshape(nbatch, nreim*nmic, nf, nt) # (nbatch, nreim*nmic, nf, nt)
            if self.f_first:
                data = data.permute(0, 1, 3, 2) # (nbatch, nreim*nmic, nt, nf)
                vec = F.unfold(data, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]])  
            else:
                vec = F.unfold(data, kernel_size=self.patch_shape, stride=self.patch_shape)  # (nbatch, nreim*nmic*dpatch, npatch)
            vec = vec.reshape(nbatch, nreim, nmic, self.patch_shape[0] * self.patch_shape[1], vec.shape[-1])  # (nbatch, nreim, nmic, dpatch, npatch)
            vec = vec.permute(0, 4, 3, 1, 2) # (nbatch, npatch, dpatch, nreim, nmic)

        return vec  # (nbatch, npatch, dpatch, nreim, nmic)


class PatchRecover(nn.Module):
    def __init__(self, output_shape, patch_shape, f_first=False):
        super(PatchRecover, self).__init__()

        self.output_shape = output_shape
        self.patch_shape = patch_shape
        self.f_first = f_first

    def forward(self, data):
        # data: (nbatch, npatch, dpatch, nmic) / (nbatch, npatch, dpatch, nreim, nmic)
        nf = self.output_shape[0]
        nt = self.output_shape[1]
        if len(data.shape) == 4:
            nbatch, npatch, _, nmic = data.shape
            vec = data.permute(0, 3, 2, 1).reshape(nbatch, -1, npatch)  # (nbatch, nmic, dpatch, npatch)
            if self.f_first:
                vec = F.fold(vec, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]], output_size=(nt, nf))  # (nbatch, nmic, nt, nf)
                vec = vec.permute(0, 1, 3, 2) # (nbatch, nmic, nf, nt)
            else:
                vec = F.fold(vec, kernel_size=self.patch_shape, stride=self.patch_shape, output_size=(nf, nt))  # (nbatch, nmic, nf, nt)
            vec = vec.reshape(nbatch, nmic, nf, nt)  # (nbatch, nmic, nf, nt)
            vec = vec.permute(0, 2, 3, 1) # (nbatch, nf, nt, nmic)

        elif len(data.shape) == 5: 
            nbatch, npatch, _, nreim, nmic = data.shape
            vec = data.permute(0, 3, 4, 2, 1).reshape(nbatch, -1, npatch)  # (nbatch, nreim, nmic, dpatch, npatch)
            if self.f_first:
                vec = F.fold(vec, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]], output_size=(nt, nf))  # (nbatch, nreim*nmic, nt, nf)
                vec = vec.permute(0, 1, 3, 2) # (nbatch, nreim*nmic, nf, nt)
            else:
                vec = F.fold(vec, kernel_size=self.patch_shape, stride=self.patch_shape, output_size=(nf, nt))  # (nbatch, nreim*nmic, nf, nt)
            vec = vec.reshape(nbatch, nreim, nmic, nf, nt)  # (nbatch, nreim, nmic, nf, nt)
            vec = vec.permute(0, 3, 4, 1, 2) # (nbatch, nf, nt, nreim, nmic)

        return vec  # (nbatch, nf, nt, nmic) / (nbatch, nf, nt, nreim, nmic)


class PatchMask(nn.Module):
    def __init__(self, patch_mode, nmasked_patch, npatch_shape, device):
        super(PatchMask, self).__init__()
        self.patch_mode = patch_mode
        self.nmasked_patch = nmasked_patch
        self.npatch_shape = npatch_shape
        self.device = device

    def forward(self, data_shape):
        nbatch, npatch, dpatch, _, nmic = data_shape

        mask_dense = torch.ones((nbatch, npatch, dpatch, nmic), device=self.device)  # (nbatch, npatch, dpatch, nmic)
        mask_patch_dense = torch.ones((nbatch, npatch, dpatch, nmic), device=self.device)  # (nbatch, npatch, dpatch, nmic)
        mask_ch_dense = torch.ones((nbatch, npatch, dpatch, nmic), device=self.device)  # (nbatch, npatch, dpatch, nmic)
        mask_patch_idx = torch.empty((nbatch, self.nmasked_patch), device=self.device, requires_grad=False).long()  # (nbatch, nmasked_patch)
        mask_ch_idx = torch.empty((nbatch, 1), device=self.device, requires_grad=False).long() # (nbatch, 1)
        for b_idx in range(nbatch):
            # randomly generate # nmasked_patch mask indexes without duplicate
            mask_patch_idx[b_idx, :] = self.gen_mask_idx(npatch_shape=self.npatch_shape, nmasked_patch=self.nmasked_patch, patch_mode=self.patch_mode)
            # mask_ch_idx[b_idx, :] = random.randrange(nmic)
            mask_ch_idx[b_idx, :] = random.randint(0, nmic-1)
            mask_dense[b_idx, mask_patch_idx[b_idx], :, mask_ch_idx[b_idx, :]] = 0
            mask_patch_dense[b_idx, mask_patch_idx[b_idx], :, :] = 0
            mask_ch_dense[b_idx, :, :, mask_ch_idx[b_idx, :]] = 0

        return mask_dense, mask_patch_dense, mask_ch_dense, mask_patch_idx, mask_ch_idx
        # mask_dense/mask_patch_dense/mask_ch_dense: (nbatch, npatch, dpatch, nmic), mask_patch_idx: (nbatch, nmasked_patch), mask_ch_idx: (nbatch, 1)

    def gen_mask_idx(self, npatch_shape=[16, 16], nmasked_patch=10, cluster=1, patch_mode='TF'):
        npatch = npatch_shape[0] * npatch_shape[1]

        if nmasked_patch > npatch:
            raise Exception('Number of masked patches is out of range')

        if patch_mode == 'TF':
            """ 
				generate mask for patches with a dimension of 16*16 (time frames*frequency bins)
				Process: find one block of patches, and then another .... until the number of patches reaches nmasked_patch 

			"""
            mask_id = []

            # randomize clutering factor in [3,6)
            cur_clus = random.randrange(cluster) + 3

            while len(list(set(mask_id))) <= nmasked_patch:
                start_id = random.randrange(npatch)

                cur_mask = []
                for i in range(0, cur_clus):
                    for j in range(0, cur_clus):
                        mask_cand = start_id + npatch_shape[1] * i + j
                        if mask_cand > 0 and mask_cand < npatch:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:nmasked_patch]
            return torch.tensor(mask_id)

        elif patch_mode == 'T':

            mask_id = random.sample(range(0, npatch), nmasked_patch)
            return torch.tensor(mask_id)
        
        elif patch_mode == 'T_cluster':
            mask_id = []

            # randomize clutering factor in [0,5)
            cur_clus = random.randrange(cluster) + 5

            while len(list(set(mask_id))) <= nmasked_patch:
                start_id = random.randrange(npatch)

                cur_mask = []
                for i in range(0, cur_clus):
                        mask_cand = start_id + i
                        if mask_cand > 0 and mask_cand < npatch:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:nmasked_patch]

            return torch.tensor(mask_id)
        
        elif patch_mode == 'T_cluster_inverse':
            mask_id = []
            mask_id_set = list(range(npatch))

            # randomize clutering factor in [0,5)
            cur_clus = random.randrange(cluster) + 5

            while len(list(set(mask_id))) <= nmasked_patch:
                start_id = random.randrange(npatch)

                cur_mask = []
                for i in range(0, cur_clus):
                        mask_cand = start_id + i
                        if mask_cand > 0 and mask_cand < npatch:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:nmasked_patch]

            for id in mask_id:
                mask_id_set.remove(id)

            return torch.tensor(mask_id_set)
        
        elif patch_mode == 'T_cluster2':
            clu_size = 5
            clu_start_id = random.sample(range(0, npatch, clu_size), math.ceil(nmasked_patch/clu_size)+1)
            mask_id = []
            for clu_idx in range(len(clu_start_id)):
                for idx in range(clu_size):
                    mask_cand = clu_start_id[clu_idx] + idx
                    if mask_cand > 0 and mask_cand < npatch:
                        mask_id = mask_id + [mask_cand]
            mask_id = list(set(mask_id))[:nmasked_patch]

            return torch.tensor(mask_id)
        
        elif patch_mode == 'T_1s':

            mask_id = list(range(192, 256)) # 1s masked
            return torch.tensor(mask_id)

        else:
            raise Exception('Patch mode is unrecognized')


class DPIPD(nn.Module):
    """ Complex-valued Direct-path inter-channel phase difference (torch version)	
	"""
    def __init__(self, ndoa_candidate, mic_location, nf=257, fre_max=8000, ch_mode='M', speed=343.0):
        super(DPIPD, self).__init__()

        if not isinstance(mic_location, torch.Tensor):
            mic_location = torch.from_numpy(mic_location)
        self.mic_location = mic_location
        self.ndoa_candidate = ndoa_candidate
        self.nf = nf
        self.fre_max = fre_max
        self.speed = speed
        self.ch_mode = ch_mode

        nmic = mic_location.shape[-2]
        nele = ndoa_candidate[0]
        nazi = ndoa_candidate[1]
        ele_candidate = torch.linspace(0, torch.pi, nele)
        azi_candidate = torch.linspace(-torch.pi, torch.pi, nazi)
        ITD = torch.empty((nele, nazi, nmic, nmic))  # Time differences, floats
        IPD = torch.empty((nele, nazi, nf, nmic, nmic))  # Phase differences
        fre_range = torch.linspace(0.0, fre_max, nf)
        r = torch.stack([torch.outer(torch.sin(ele_candidate), torch.cos(azi_candidate)),
                     torch.outer(torch.sin(ele_candidate), torch.sin(azi_candidate)),
                     torch.tile(torch.cos(ele_candidate), [nazi, 1]).transpose(1,0)], axis=2)
        for m1 in range(nmic):
            for m2 in range(nmic):
                ITD[:, :, m1, m2] = (r * (mic_location[m2, :] - mic_location[m1, :])).sum(dim=-1) / speed
                IPD[:, :, :, m1, m2] = -2 * torch.pi * torch.tile(fre_range[np.newaxis, np.newaxis, :], [nele, nazi, 1]) * \
                        torch.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])

        dpipd_template_ori = torch.exp(1j * IPD)
        self.dpipd_template = self.data_adjust(dpipd_template_ori) # (nele, nazi, nf, nmic-1) / (nele, nazi, nf, nmic*(nmic-1)/2)
        self.doa_candidate = [ele_candidate, azi_candidate]         
        # 	# import scipy.io
        # 	# scipy.io.savemat('dpipd_template_nele_nazi_2nf_nmic-1.mat',{'dpipd_template': self.dpipd_template})
        # 	# print(a)
        
        del ITD, IPD

    def forward(self, source_doa=None):
        # source_doa: (nb, ntimestep, 2, nsource)
        if source_doa is not None:
            device = source_doa.device
            nf = self.nf
            fre_max = self.fre_max
            speed = self.speed
            mic_location = self.mic_location.to(device)
            source_doa = source_doa.permute(0, 1, 3, 2) # (nb, ntimestep, nsource, 2)
            nmic = mic_location.shape[-2]
            nb = source_doa.shape[0]
            nsource = source_doa.shape[-2]
            ntime = source_doa.shape[-3]
            ITD = torch.empty((nb, ntime, nsource, nmic, nmic)).to(device)  # Time differences, floats
            IPD = torch.empty((nb, ntime, nsource, nf, nmic, nmic)).to(device) # Phase differences
            fre_range = torch.linspace(0.0, fre_max, nf).to(device)
            r = torch.stack([torch.sin(source_doa[:, :, :, 0]) * torch.cos(source_doa[:, :, :, 1]),
                         torch.sin(source_doa[:, :, :, 0]) * torch.sin(source_doa[:, :, :, 1]),
                         torch.cos(source_doa[:, :, :, 0])], axis=3)
            for m1 in range(nmic):
                for m2 in range(nmic): 
                    ITD[:, :, :, m1, m2] =(r * (mic_location[m1, :] - mic_location[m2, :])).sum(dim=-1) / speed # t2- t1
                    IPD[:, :, :, :, m1, m2] = -2 * torch.pi * torch.tile(fre_range[np.newaxis, np.newaxis, np.newaxis, :],
                         [nb, ntime, nsource, 1]) * torch.tile(ITD[:, :, :, np.newaxis, m1, m2], [1, 1, 1, nf])*(-1)   # !!!! delete -1
            dpipd_ori = torch.exp(1j * IPD)
            dpipd = self.data_adjust(dpipd_ori) # (nb, ntime, nsource, nf, nmic-1) / (nb, ntime, nsource, nf, nmic*(nmic-1)/2)

            dpipd = dpipd.permute(0, 1, 3, 4, 2) # (nb, ntime, nf, nmic-1, nsource)

        else:
            dpipd = None

        return self.dpipd_template, dpipd, self.doa_candidate

    def data_adjust(self, data):
        # change dimension from (..., nmic, nmic) to (..., nmic-1)/(..., nmic*(nmic-1)/2)
        if self.ch_mode == 'M':
            data_adjust = data[..., 0, 1:] # (..., nmic-1)
        elif self.ch_mode == 'MM':
            nmic = data.shape[-1]
            data_adjust = torch.empty(data.shape[:-2] + (int(nmic*(nmic-1)/2),), dtype=torch.complex64)
            for mic_idx in range(nmic - 1):
                st = int((2 * nmic - 2 - mic_idx + 1) * mic_idx / 2)
                ed = int((2 * nmic - 2 - mic_idx) * (mic_idx + 1) / 2)
                data_adjust[..., st:ed] = data[..., mic_idx, (mic_idx+1):] # (..., nmic*(nmic-1)/2)
        else:
            raise Exception('Microphone channel mode unrecognised')

        return data_adjust
    

# class GCC(nn.Module):
# 	""" Compute the Generalized Cross Correlation of the inputs.
# 	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K).
# 	You can use tau_max to output only the central part of the GCCs and transform='PHAT' to use the PHAT transform.
# 	"""

# 	def __init__(self, N, K, tau_max=None, transform=None):
# 		assert transform is None or transform == 'PHAT', 'Only the \'PHAT\' transform is implemented'
# 		assert tau_max is None or tau_max <= K // 2
# 		super(GCC, self).__init__()

# 		self.K = K
# 		self.N = N
# 		self.tau_max = tau_max if tau_max is not None else K // 2
# 		self.transform = transform

# 	def forward(self, x):
# 		x_fft_c = torch.fft.rfft(x)
# 		x_fft = torch.stack((x_fft_c.real, x_fft_c.imag), -1)  

# 		if self.transform == 'PHAT':
# 			mod = torch.sqrt(complex_conjugate_multiplication(x_fft, x_fft))[..., 0]
# 			mod += 1e-12  # To avoid numerical issues
# 			x_fft /= mod.reshape(tuple(x_fft.shape[:-1]) + (1,))

# 		gcc = torch.empty(list(x_fft.shape[0:-3]) + [self.N, self.N, 2 * self.tau_max + 1], device=x.device)
# 		for n in range(self.N):
# 			gcc_fft_batch = complex_conjugate_multiplication(x_fft[..., n, :, :].unsqueeze(-3), x_fft)
# 			gcc_fft_batch_c = torch.complex(gcc_fft_batch[..., 0], gcc_fft_batch[..., 1])
# 			gcc_batch = torch.fft.irfft(gcc_fft_batch_c)    

# 			gcc[..., n, :, 0:self.tau_max + 1] = gcc_batch[..., 0:self.tau_max + 1]
# 			gcc[..., n, :, -self.tau_max:] = gcc_batch[..., -self.tau_max:]

# 		return gcc


# class SRP_map(nn.Module):
#     """ Compute the SRP-PHAT maps from the GCCs taken as input.
# 	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
# 	desired resolution of the maps (resTheta and resPhi), the microphone positions relative to the center of the
# 	array (rn) and the sampling frequency (fs).
# 	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
# 	"""

#     def __init__(self, N, K, resTheta, resPhi, rn, fs, c=343.0, normalize=True, thetaMax=np.pi / 2):
#         super(SRP_map, self).__init__()

#         self.N = N
#         self.K = K
#         self.resTheta = resTheta
#         self.resPhi = resPhi
#         self.fs = float(fs)
#         self.normalize = normalize

#         self.cross_idx = np.stack([np.kron(np.arange(N, dtype='int16'), np.ones((N), dtype='int16')),
#                  np.kron(np.ones((N), dtype='int16'), np.arange(N, dtype='int16'))])

#         self.theta = np.linspace(0, thetaMax, resTheta)
#         self.phi = np.linspace(-np.pi, np.pi, resPhi + 1)
#         self.phi = self.phi[0:-1]

#         self.IMTDF = np.empty((resTheta, resPhi, self.N, self.N))  # Time differences, floats
#         for k in range(self.N):
#             for l in range(self.N):
#                 r = np.stack(
#                  [np.outer(np.sin(self.theta), np.cos(self.phi)), np.outer(np.sin(self.theta), np.sin(self.phi)),
#                   np.tile(np.cos(self.theta), [resPhi, 1]).transpose()], axis=2)
#                 self.IMTDF[:, :, k, l] = np.dot(r, rn[l, :] - rn[k, :]) / c

#         tau = np.concatenate([range(0, K // 2 + 1), range(-K // 2 + 1, 0)]) / float(fs)  # Valid discrete values
#         self.tau0 = np.zeros_like(self.IMTDF, dtype=np.int)
#         for k in range(self.N):
#             for l in range(self.N):
#                 for i in range(resTheta):
#                     for j in range(resPhi):
#                         self.tau0[i, j, k, l] = int(np.argmin(np.abs(self.IMTDF[i, j, k, l] - tau)))
#         self.tau0[self.tau0 > K // 2] -= K
#         self.tau0 = self.tau0.transpose([2, 3, 0, 1])

#     def forward(self, x):
#         tau0 = self.tau0
#         tau0[tau0 < 0] += x.shape[-1]
#         maps = torch.zeros(list(x.shape[0:-3]) + [self.resTheta, self.resPhi], device=x.device).float()
#         for n in range(self.N):
#             for m in range(self.N):
#                 maps += x[..., n, m, tau0[n, m, :, :]]

#         if self.normalize:
#             maps -= torch.mean(torch.mean(maps, -1, keepdim=True), -2, keepdim=True)
#             maps += 1e-12  # To avoid numerical issues
#             maps /= torch.max(torch.max(maps, -1, keepdim=True)[0], -2, keepdim=True)[0]

#         return maps


if __name__ == "__main__":
    import torch
    win_len = 512
    win_shift_ratio = 0.5
    nfft = 512
    a = torch.randn((100, 512, 1))
    inv = False
    dostft = STFT(
        win_len=win_len, 
		win_shift_ratio=win_shift_ratio, 
		nfft=nfft,
        inv=inv)
    doistft = ISTFT(
        win_len=win_len, 
		win_shift_ratio=win_shift_ratio, 
		nfft=nfft,
        inv=inv)
    stft = dostft(a)
    aa = doistft(stft)
    print(a.shape, stft.shape, aa.shape)  


def hsic(x, y, sigma=None):
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC) with RBF kernel.
    
    Args:
        x (torch.Tensor): Input tensor x (batch_size, dim_x)
        y (torch.Tensor): Input tensor y (batch_size, dim_y)
        sigma (float, optional): Kernel bandwidth. If None, median heuristic is used.
        
    Returns:
        torch.Tensor: HSIC value
    """
    m = x.shape[0]
    
    if m < 2:
        return torch.tensor(0.0, device=x.device)

    # Normalize inputs for stability (zero mean, unit variance)
    # This prevents extremely large distances in RBF kernel
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-6)
    
    # RBF Kernel for X
    K = _rbf_kernel(x, sigma)
    
    # RBF Kernel for Y
    L = _rbf_kernel(y, sigma)
    
    # Centering matrix H
    H = torch.eye(m, device=x.device) - (1.0 / m) * torch.ones((m, m), device=x.device)
    
    # HSIC = trace(K * H * L * H) / (m-1)^2  (unbiased estimator form often used varies, here using standard V-statistic form / m^2 or simply unnormalized)
    # Using the standard empirical HSIC: Tr(KHLH) / (m-1)^2
    
    # KH = torch.mm(K, H)
    # LH = torch.mm(L, H)
    # hsic_value = torch.trace(torch.mm(KH, LH)) / ((m - 1) ** 2)
    
    # More efficient: sum(sum( (K centered) * (L centered) ))
    # centralized kernel matrices
    Kc = torch.mm(torch.mm(H, K), H)
    Lc = torch.mm(torch.mm(H, L), H)
    
    hsic_value = torch.sum(Kc * Lc) / ((m - 1) ** 2)
    
    return hsic_value

def _rbf_kernel(x, sigma=None):
    """
    Compute RBF Kernel Matrix.
    K(x_i, x_j) = exp(- ||x_i - x_j||^2 / (2 * sigma^2))
    """
    # Calculate squared Euclidean distance
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    
    x_norm = (x ** 2).sum(1).view(-1, 1)
    dist_sq = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
    dist_sq = torch.clamp(dist_sq, min=0.0) # Numerical stability
    
    if sigma is None:
        # Median heuristic
        # Use median of the upper triangle of the distance matrix (excluding diagonal)
        
        # Flatten and take upper triangle
        mask = torch.triu(torch.ones_like(dist_sq), diagonal=1).bool()
        if mask.sum() > 0:
            distances = torch.sqrt(dist_sq[mask])
            median_dist = torch.median(distances)
            sigma = median_dist # use median distance as bandwidth
        else:
             sigma = 1.0 # Fallback for very small batch size 
             
    # Avoid division by zero or extremely small sigma
    if isinstance(sigma, torch.Tensor):
        sigma = torch.clamp(sigma, min=1e-6)
    elif sigma < 1e-6:
        sigma = 1e-6

    gamma = 1.0 / (2.0 * sigma ** 2)
    K = torch.exp(-gamma * dist_sq)
    return K


def spatial_features_from_stft(
    X: torch.Tensor,
    sample_rate: float,
    max_ipd_freq: float = 1500.0,
    eps: float = 1e-8,
    hop_length: int = 256,
    early_win_s: float = 0.05,
    late_win_s: float = 0.20,
    onset_db_threshold: float = 6.0,
) -> dict:
    """
    Compute spatial features from per-channel STFTs.
    
    Args:
        X: [B, 2, F, T] complex tensor.
        sample_rate: sampling rate in Hz.
        max_ipd_freq: max frequency for IPD features.
        eps: numerical stability.
        
    Returns:
        dict: feature dictionary.
    """
    device = X.device
    B, C, F_bins, T_frames = X.shape
    assert C == 2, "Expected binaural input"

    X_L = X[:, 0]
    X_R = X[:, 1]
    
    # 1. ILD (Broadband)
    min_ild_freq = 2000.0
    magL = X_L.abs() + eps
    magR = X_R.abs() + eps
    powL = magL**2
    powR = magR**2
    total_power = powL + powR # [B, F, T]
    
    # Raw ILD (log magnitude ratio)
    ild_bins = 10.0 * torch.log10((powL + eps) / (powR + eps))
    
    # Frequency Mask: Only keep High Frequencies (> 2000 Hz)
    # ILD is physically negligible at low freqs due to diffraction
    freqs = torch.linspace(0, sample_rate / 2, F_bins, device=device)
    hf_mask = freqs > min_ild_freq # [F]
    
    if hf_mask.sum() > 0:
        # Apply mask to freq dimension
        ild_hf = ild_bins[:, hf_mask, :]       # [B, F_hf, T]
        weight_hf = total_power[:, hf_mask, :] # [B, F_hf, T]
        
        # Weighted Mean: Sum(ILD * Weight) / Sum(Weight)
        # This automatically ignores silent bins (Weight ~ 0)
        numerator = (ild_hf * weight_hf).sum(dim=(1, 2))
        denominator = weight_hf.sum(dim=(1, 2)) + eps
        ild_mean = numerator / denominator
    else:
        ild_mean = torch.zeros(B, device=device)
    
    # 2. IPD (Low Frequency)
    phase_L = torch.angle(X_L)
    phase_R = torch.angle(X_R)
    ipd = phase_L - phase_R
    ipd = (ipd + math.pi) % (2 * math.pi) - math.pi
    
    freqs = torch.linspace(0, sample_rate / 2, F_bins, device=device)
    lf_mask = freqs <= max_ipd_freq
    
    if lf_mask.sum() > 0:
        ipd_lf = ipd[:, lf_mask, :]
        ipd_cos_mean = torch.cos(ipd_lf).mean(dim=(1, 2))
        ipd_sin_mean = torch.sin(ipd_lf).mean(dim=(1, 2))
    else:
        ipd_cos_mean = torch.zeros(B, device=device)
        ipd_sin_mean = torch.zeros(B, device=device)
        
    # 3. Coherence (Time-averaged)
    cross_spec = X_L * X_R.conj()
    # E[XY*]
    cross_spec_mean = cross_spec.mean(dim=2)  # [B, F] average over time
    # E[|X|^2]
    powL_mean = (magL**2).mean(dim=2)  # [B, F]
    powR_mean = (magR**2).mean(dim=2)  # [B, F]

    coh_freq = cross_spec_mean.abs() / (torch.sqrt(powL_mean * powR_mean) + eps)
    coherence_mean = coh_freq.mean(dim=1)  # [B] average over freq

    coh_var = coh_freq.var(dim=1) # [B]
    
    # Split coherence into frequency bands for feature-similarity losses
    # Bands: low < 500 Hz, mid 500-2500 Hz, high > 2500 Hz
    freqs = torch.linspace(0, sample_rate / 2, F_bins, device=device)
    low_mask = freqs < 500.0
    mid_mask = (freqs >= 500.0) & (freqs <= 2500.0)
    high_mask = freqs > 2500.0

    # Handle cases where a mask might select zero bins
    if low_mask.sum() > 0:
        coherence_low = coh_freq[:, low_mask].mean(dim=1)
    else:
        coherence_low = torch.zeros(B, device=device)

    if mid_mask.sum() > 0:
        coherence_mid = coh_freq[:, mid_mask].mean(dim=1)
    else:
        coherence_mid = torch.zeros(B, device=device)

    if high_mask.sum() > 0:
        coherence_high = coh_freq[:, high_mask].mean(dim=1)
    else:
        coherence_high = torch.zeros(B, device=device)

    # --- HF/LF spectral energy ratio (spectral tilt) ---
    # Use average binaural magnitude as monaural proxy
    mag_mean = 0.5 * (magL + magR)  # [B, F, T]
    mag_mean_freq = mag_mean.mean(dim=2)  # [B, F]
    if low_mask.sum() > 0:
        E_low = (mag_mean_freq[:, low_mask] ** 2).sum(dim=1)
    else:
        E_low = torch.zeros(B, device=device)
    if high_mask.sum() > 0:
        E_high = (mag_mean_freq[:, high_mask] ** 2).sum(dim=1)
    else:
        E_high = torch.zeros(B, device=device)

    hf_lf_ratio = torch.log10((E_high + eps) / (E_low + eps))
    
    # 4. GCC-PHAT & TDOA
    # R(f) = X_L(f) * X_R(f)* / |X_L(f) * X_R(f)*|
    cc = X_L * X_R.conj()
    cc_phat = cc / (cc.abs() + eps)
    
    # GCC = IFFT(R(f))
    # We need to handle the frequency dimension correctly. 
    # torch.fft.irfft expects the last dimension to be the frequency dimension of a REAL signal's FFT (size N/2+1).
    # X matches this since it came from rfft or stft(onesided=True).
    # We average over time frames to get a robust GCC for the whole clip.
    cc_phat_mean = cc_phat.mean(dim=2) # [B, F]
    
    # The length of the real signal for irfft is 2*(F_bins-1)
    # This is because F_bins is N/2 + 1, so N = 2*(F_bins-1)
    gcc_phat = torch.fft.irfft(cc_phat_mean, n=2*(F_bins-1), dim=-1) # [B, 2*(F-1)]
    
    # Shift so 0 lag is at center
    # The zero lag for irfft is at index 0.
    # To center it, we roll by half the length.
    # The length is 2*(F_bins-1). So roll by (F_bins-1).
    gcc_phat = torch.roll(gcc_phat, shifts=F_bins-1, dims=-1)
    
    # Find peak
    gcc_phat_max, max_idx = torch.max(gcc_phat.abs(), dim=1)
    
    # Convert index to TDOA
    # The center (zero lag) is now at index F_bins-1.
    sample_delay = max_idx.float() - (F_bins - 1)
    tdoa = sample_delay / sample_rate
    
    # 5. IACC (Raw Cross-Correlation) - Spatial Width Cue
    # Unlike GCC-PHAT, this is NOT whitened. It captures the energy decay shape.
    # CC = IFFT( E[X_L * X_R*] )
    n_fft = 2 * (F_bins - 1)
    raw_cc = torch.fft.irfft(cross_spec_mean, n=n_fft, dim=-1) # [B, Time]
    raw_cc = torch.roll(raw_cc, shifts=F_bins-1, dims=-1)
    
    # Normalize by energy (Standard IACC definition)
    # Denom = sqrt( E[x^2] * E[y^2] ) - scalar per sample
    # Note: using sum over F of the PSDs estimates total energy
    energy_L = powL_mean.sum(dim=1)
    energy_R = powR_mean.sum(dim=1)
    norm_factor = torch.sqrt(energy_L * energy_R) + eps
    raw_cc_norm = raw_cc / norm_factor.unsqueeze(1)
    
    # Feature: IACC (Peak of normalized raw cross-correlation)
    # High IACC (~1.0) = Narrow/Focused source
    # Low IACC (<0.3) = Diffuse/Enveloping sound
    iacc_max, _ = torch.max(raw_cc_norm.abs(), dim=1)

    # 6. Decay Slope Statistics (Room Acoustics Proxy)
    # Calculated per-sample as consistent with reference logic
    decay_slope_median = torch.zeros(B, device=device)
    decay_slope_p10 = torch.zeros(B, device=device)
    decay_slope_p90 = torch.zeros(B, device=device)

    # Log Energy of Left Channel
    # X_L is [B, F, T]
    logE_ft = torch.log(magL**2 + eps)
    # Difference along time: [B, F, T-1]
    dlogE_ft = torch.diff(logE_ft, dim=2)
    
    # Loop over batch to handle variable number of negative slopes
    for b in range(B):
        # Flatten time-freq for this sample
        slopes = dlogE_ft[b].flatten()
        # Keep only negative
        neg_slopes = slopes[slopes < 0.0]
        
        if neg_slopes.numel() == 0:
            decay_slope_median[b] = 0.0
            decay_slope_p10[b] = 0.0
            decay_slope_p90[b] = 0.0
        else:
            # Quantiles
            # torch.quantile requires float32/64
            decay_slope_median[b] = torch.quantile(neg_slopes, 0.5)
            decay_slope_p10[b] = torch.quantile(neg_slopes, 0.1)
            decay_slope_p90[b] = torch.quantile(neg_slopes, 0.9)

    # 7. Modulation Low/High Ratio (SRMR-like proxy)
    # Broadband energy calculation
    # sum over freq bins: [B, T]
    E_t = (magL**2).sum(dim=1) + eps
    logE_t = torch.log(E_t)
    
    # Remove mean and apply Hann window
    # Window should be [T]
    win = torch.hann_window(T_frames, device=device)
    logE_t_centered = logE_t - logE_t.mean(dim=1, keepdim=True)
    x_mod_in = logE_t_centered * win.unsqueeze(0) # [B, T]
    
    # Modulation spectrum
    X_mod = torch.fft.rfft(x_mod_in, dim=-1)
    P_mod = X_mod.abs() ** 2
    
    # Modulation Frequencies
    fs_mod = sample_rate / float(hop_length)
    # rfftfreq returns [T//2 + 1]
    freqs_mod = torch.fft.rfftfreq(T_frames, d=1.0/fs_mod).to(device)
    
    low_band_mask = (freqs_mod >= 0.5) & (freqs_mod <= 4.0)
    high_band_mask = (freqs_mod > 4.0) & (freqs_mod <= 20.0)
    
    # Integration
    # We can do this vectorized over B
    if low_band_mask.any():
        P_low = P_mod[:, low_band_mask].sum(dim=1)
    else:
        P_low = torch.zeros(B, device=device)
        
    if high_band_mask.any():
        P_high = P_mod[:, high_band_mask].sum(dim=1) + eps
    else:
        P_high = torch.ones(B, device=device) * eps # avoid div zero
        
    mod_low_high_ratio = P_low / P_high

    # 8. Early-to-Late Energy Ratio (Clarity proxy)
    # Onset detection on broadband energy E_t [B, T]
    # E_t was computed above for modulation ratio
    E_prev = torch.clamp(E_t[:, :-1], min=eps)
    E_curr = torch.clamp(E_t[:, 1:], min=eps)
    dB_change = 10.0 * torch.log10(E_curr / E_prev) # [B, T-1]
    
    early_win_frames = max(1, int(round(early_win_s * sample_rate / hop_length)))
    late_win_frames = max(1, int(round(late_win_s * sample_rate / hop_length)))
    
    early_late_ratio_mean = torch.zeros(B, device=device)
    early_late_ratio_median = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find onsets where dB change > threshold
        # dim 0 of result tuple
        onsets = torch.where(dB_change[b] > onset_db_threshold)[0] + 1
        
        ratios = []
        for n0 in onsets:
            n0 = n0.item()
            n_early_end = min(T_frames, n0 + early_win_frames)
            n_late_end = min(T_frames, n_early_end + late_win_frames)
            
            if n_late_end <= n_early_end:
                continue
                
            E_early = E_t[b, n0:n_early_end].sum()
            E_late = E_t[b, n_early_end:n_late_end].sum() + eps
            ratios.append(E_early / E_late)
            
        if len(ratios) == 0:
            early_late_ratio_mean[b] = 0.0
            early_late_ratio_median[b] = 0.0
        else:
            ratios_tensor = torch.tensor(ratios, device=device)
            early_late_ratio_mean[b] = ratios_tensor.mean()
            early_late_ratio_median[b] = torch.quantile(ratios_tensor, 0.5)

    return {
        'ipd_cos_mean': ipd_cos_mean,
        'ipd_sin_mean': ipd_sin_mean,
        'ild_mean': ild_mean,
        'coherence_mean': coherence_mean,
        'coherence_var': coh_var,
        'coherence_low': coherence_low,
        'coherence_mid': coherence_mid,
        'coherence_high': coherence_high,
        'hf_lf_ratio': hf_lf_ratio,
        'tdoa': tdoa,
        'gcc_phat_max': gcc_phat_max,
        'iacc_max': iacc_max,
        'decay_slope_median': decay_slope_median,
        'decay_slope_p10': decay_slope_p10,
        'decay_slope_p90': decay_slope_p90,
        'mod_low_high_ratio': mod_low_high_ratio,
        'early_late_ratio_mean': early_late_ratio_mean,
        'early_late_ratio_median': early_late_ratio_median
    }

def spectral_features_from_stft(
    X: torch.Tensor,
    sample_rate: float,
    hop_length: int = 256,
    n_fft: int = 512,
) -> dict:
    """
    Extracts a frame-wise, robust DSP feature vector for speech from STFT.
    
    Args:
        X (torch.Tensor): Complex STFT tensor of shape (B, F, T) or (B, C, F, T)
        sample_rate (float): Sample rate
        hop_length (int): Frame shift (should match the encoder's stride)
        n_fft (int): Window size for STFT
        
    Returns:
        dict containing:
            'feature_matrix' (torch.Tensor): Feature matrix of shape (B, 16, T)
            'voiced_prob' (torch.Tensor): Probability of voicing of shape (B, T)
    """
    import librosa
    
    device = X.device
    B = X.shape[0]
    
    # 1. Monaural Mixdown: Ensure spatial differences don't contaminate spectral targets
    if X.ndim == 4 and X.shape[1] > 1:
        X_mono = X.mean(dim=1)
    elif X.ndim == 4:
        X_mono = X.squeeze(1)
    else:
        X_mono = X
        
    feature_matrices = []
    voiced_probs = []
    
    for b in range(B):
        # Convert STFT to numpy array
        S_complex = X_mono[b].detach().cpu().numpy()
        
        # 1. Invert STFT to get the time-domain waveform
        waveform = librosa.istft(S_complex, hop_length=hop_length, n_fft=n_fft)
        
        # 2. Pitch / Fundamental Frequency (F0) & Voicing Mask
        f0, voiced_flag, voiced_prob = librosa.pyin(
            waveform, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate, 
            hop_length=hop_length,
            fill_na=0.0
        )
        
        T_target = S_complex.shape[1]
        
        def pad_or_trunc(arr, target_len):
            if len(arr) < target_len:
                return np.pad(arr, (0, target_len - len(arr)))
            return arr[:target_len]
            
        f0 = pad_or_trunc(f0, T_target)
        voiced_prob = pad_or_trunc(voiced_prob, T_target)
        
        # Use magnitude spectrogram for energy and spectral features
        S_mag = np.abs(S_complex)
        
        # 3. RMS Log-Energy
        rms = librosa.feature.rms(S=S_mag, frame_length=n_fft, hop_length=hop_length)[0]
        rms = pad_or_trunc(rms, T_target)
        log_energy = librosa.amplitude_to_db(rms, ref=np.max)
        
        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        centroid = pad_or_trunc(centroid, T_target)
        
        # 5. MFCCs
        mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(S_mag, ref=np.max), sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        if mfccs.shape[1] < T_target:
            mfccs = np.pad(mfccs, ((0, 0), (0, T_target - mfccs.shape[1])))
        else:
            mfccs = mfccs[:, :T_target]
            
        feature_matrix = np.vstack([
            f0,                 # Shape: (1, T)
            log_energy,         # Shape: (1, T)
            centroid,           # Shape: (1, T)
            mfccs               # Shape: (13, T)
        ])
        
        feature_matrices.append(feature_matrix)
        voiced_probs.append(voiced_prob)
        
    return {
        'feature_matrix': torch.tensor(np.array(feature_matrices), device=device, dtype=torch.float32),
        'voiced_prob': torch.tensor(np.array(voiced_probs), device=device, dtype=torch.float32)
    }

def compute_spatial_features(
    x: torch.Tensor,
    sample_rate: float,
    n_fft: int = 512,
    hop_length: int = 256,
    max_ipd_freq: float = 1500.0,
    eps: float = 1e-8,
    early_win_s: float = 0.05,
    late_win_s: float = 0.20,
    onset_db_threshold: float = 6.0,
) -> dict:
    """
    Compute spatial acoustic features from binaural recordings.
    
    Args:
        x: [B, 2, T] binaural waveform (L, R).
        sample_rate: sampling rate in Hz.
        n_fft, hop_length: STFT parameters.
        max_ipd_freq: only use frequencies <= this for IPD summary.
        eps: numerical stability.
        early_win_s, late_win_s, onset_db_threshold: ELR parameters.
        
      Returns:
      dict: feature dictionary containing 'ipd_cos_mean', 'ipd_sin_mean', 'ild_mean',
          'coherence_mean', 'coherence_low', 'coherence_mid', 'coherence_high',
          'tdoa', 'gcc_phat_max', 'decay_slope_*', 'mod_low_high_ratio', 'early_late_ratio_*'.
    """
    import math
    device = x.device
    B, C, T = x.shape
    assert C == 2, "Expected binaural input [B, 2, T]"

    # STFT: [B*C, F, T_frames]
    # Use center=True to match standard processing
    X = torch.stft(
        x.view(B * C, T),
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=device),
        return_complex=True,
        center=True
    )
    F_bins, T_frames = X.shape[-2], X.shape[-1]
    X = X.view(B, C, F_bins, T_frames) # [B, 2, F, T]

    return spatial_features_from_stft(X, sample_rate, max_ipd_freq, eps, hop_length, early_win_s, late_win_s, onset_db_threshold)
