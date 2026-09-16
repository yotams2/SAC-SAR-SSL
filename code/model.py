import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import common.utils_module as at_module

# Different architechture modules 
from common.Conformer import ConformerEncoder
from common.Transformer import TransformerEncoder
from common.CRNN import crnn, crnn_sim, tcrnn
# from common.NBC import NBC
# from common.FNSSL import FNblock
# from common.UNet import UNet
# from common.CNN import resnet50, res2net50, densenet121
# from ks.model_ds_compare import CausCRNN_predhead

class EmbedEncoder(nn.Module):

    def __init__(self, sig_shape, patch_shape, dembed, model=['cnn', 'conformer'], mode='spat', use_cls=False, device='cpu', hierarchical=False, hierarchical_w_stop_grad=False, hierarchical_w_ch_att=False, hierarchical_stop_grad_branch='room'):
        super(EmbedEncoder, self).__init__()

        self.sig_shape = sig_shape
        self.patch_shape = patch_shape
        self.dembed = dembed
        self.device = device
        self.model = model
        self.use_cls = use_cls
        self.hierarchical_w_stop_grad = hierarchical_w_stop_grad
        self.hierarchical_w_ch_att = hierarchical_w_ch_att
        self.hierarchical_stop_grad_branch = hierarchical_stop_grad_branch
            
        nf, nt, nreim, nmic = sig_shape
        nch = nreim*nmic
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        dpatch = patch_shape[0] * patch_shape[1]
        npatch = npatch_shape[0] * npatch_shape[1]
        dembed_in = dpatch*nreim*nmic

        if len(model) == 2:
            if mode == 'spec':
                mhsa_nlayer = 1
                conv_chs = 64
            elif mode == 'spat':
                mhsa_nlayer = 3
                conv_chs = 64

            # Local TF processing
            if model[0] == 'fc':
                self.patch_proj = nn.Linear(dembed_in, dembed)
            elif model[0] == 'cnn':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.patch_embed = nn.Sequential(
                    nn.Conv2d(nch, conv_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, nch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                    nn.BatchNorm2d(nch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(nch), dembed, kernel_size=patch_shape, stride=patch_shape, padding=0, bias=False),
                )
            elif model[0] == 'cnn_f_first':
                self.patch_recover = at_module.PatchRecover(output_shape=(nt, nf), patch_shape=[self.patch_shape[-1], self.patch_shape[0]])
                self.patch_embed = nn.Sequential(
                    nn.Conv2d(nch, conv_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(conv_chs),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_chs, nch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                    nn.BatchNorm2d(nch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(nch), dembed, kernel_size=[patch_shape[-1], patch_shape[0]], stride=[patch_shape[-1], patch_shape[0]], padding=0, bias=False),
                ) 

            # Global TF processing
            if (model[1] == 'transformer') | (model[1] == 'conformer'):
                if self.use_cls:
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dembed))
                    trunc_normal_(self.cls_token, std=.02)
                    nseq = npatch + 1
                else:
                    nseq = npatch
                if model[1] == 'transformer':
                    self.embed = TransformerEncoder(d_model=self.dembed, nlayer=mhsa_nlayer, nhead=4, d_ff=self.dembed*4, dropout=0.1)
                elif model[1] == 'conformer':
                    self.embed = ConformerEncoder(encoder_dim=self.dembed, num_layers=mhsa_nlayer, num_attention_heads=4, feed_forward_expansion_factor=4, hierarchical=hierarchical, hierarchical_w_stop_grad=hierarchical_w_stop_grad, hierarchical_w_ch_att=self.hierarchical_w_ch_att, hierarchical_stop_grad_branch=self.hierarchical_stop_grad_branch)

        elif len(model) == 1:
            if model[0] == 'crnn':
                if mode == 'spec':
                    conv_chs = 32
                    f_stride = [1,4,4]
                    res_flag = True
                    rnn_nlayer = 1
                elif mode == 'spat':
                    conv_chs = 16
                    f_stride = [1,1,4,4,4]
                    res_flag = True
                    rnn_nlayer = 1
               
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.crnn = crnn(nf=nf, 
                                cnn_inplanes=nch, 
                                planes=[conv_chs,conv_chs,conv_chs*2,conv_chs*4,conv_chs*8], 
                                f_stride=f_stride, 
                                res_flag=res_flag, 
                                rnn_nlayer=rnn_nlayer, 
                                rnn_bdflag =True,
                                out_dim=dembed)
            elif model[0] == 'crnn-sim':
                if mode == 'spec':
                    rnn_nlayer = 1
                    conv_chs = 64
                elif mode == 'spat':
                    rnn_nlayer = 1
                    conv_chs = 64
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.crnn = crnn_sim(cnn_inplanes=nch, 
                                res_flag=res_flag, 
                                conv_chs=conv_chs, 
                                rnn_in_dim=256*2, 
                                rnn_hid_dim=dembed, 
                                rnn_nlayer=rnn_nlayer, 
                                rnn_bdflag =True)
            elif model[0] == 'tcrnn':
                if mode == 'spec':
                    rnn_nlayer = 1
                    conv_chs = [256*2, 256]
                    res_flag = True
                elif mode == 'spat':
                    rnn_nlayer = 1
                    conv_chs = [256, 256, 128]
                    res_flag = True
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.crnn = tcrnn(cnn_inplanes=nch*nf, 
                                planes = conv_chs,
                                res_flag=res_flag, 
                                rnn_nlayer=rnn_nlayer, 
                                rnn_bdflag =True,
                                out_dim=dembed)
            elif model[0] == 'fn':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                dim = 128
                self.embed1 = FNblock(ori_in_size=nch, hidden_size=dim, is_online=False, is_first_block=True) 
                self.embed2 = FNblock(ori_in_size=dim, hidden_size=dim, is_online=False, is_first_block=False) 
                # self.embed3 = FNblock(ori_in_size=dim, hidden_size=dim, is_online=False, is_first_block=False) 
                self.embed_proj = nn.Linear(dim, nch)
                self.t_embed_proj = nn.Linear(nch*nf, dembed)
            elif model[0] == 'nbc':
                if mode == 'spec':
                    mhsa_nlayer = 1
                elif mode == 'spat':
                    mhsa_nlayer = 3
                if self.use_cls:
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dembed))
                    trunc_normal_(self.cls_token, std=.02)
                    nseq = npatch + 1
                else:
                    nseq = npatch
                self.embed = NBC(hidden_size=self.dembed, n_layers=mhsa_nlayer, n_heads=4, ffn_size=self.dembed*4)
            elif model[0] == 'spatialnet':
                pass

            elif model[0] == 'resnet':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = resnet50(in_planes=nch)
                self.cnn_proj = nn.Conv2d(1024, dembed, kernel_size=(1, 1), stride=(1, 1), padding=0)
            elif model[0] == 'res2net':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = res2net50(in_planes=nch)
                self.cnn_proj = nn.Conv2d(1024, dembed, kernel_size=(1, 1), stride=(1, 1), padding=0)
            elif model[0] == 'densenet':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = densenet121(in_planes=nch)
                self.cnn_proj = nn.Conv2d(1024, dembed, kernel_size=(1, 1), stride=(1, 1), padding=0)
            elif model[0] == 'unet':
                self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=self.patch_shape)
                self.cnn = UNet(n_channels=nch, n_classes=1)
                self.cnn_proj = nn.Conv2d(1, dpatch, kernel_size=patch_shape, stride=patch_shape, padding=0, bias=False)
            else:
                pass

        else:
            raise Exception('Unrecognized submodel~')
        
    def forward(self, embed, add_same_one=False):
        # embed: (nbatch, npatch, dpatch*2*nmic)
        nbatch, npatch, dim = embed.shape
        dpatch = self.patch_shape[0]*self.patch_shape[1]
        nch = int(dim/dpatch)

        if len(self.model) == 2:
            if self.model[0] == 'fc':
                embed = self.patch_proj(embed)  # (nbatch, npatch, dembed)
            elif (self.model[0] == 'cnn'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.patch_embed(embed)  # (nbatch, dpatch*nch, ..., ...)
                embed = embed.reshape(nbatch, embed.shape[1], npatch).permute(0, 2, 1)  # (nbatch, npatch, d_patch*nch) / (nbatch, npatch, dembed)
            elif (self.model[0] == 'cnn_f_first'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nt, nf, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nt, nf)
                embed = self.patch_embed(embed)  # (nbatch, dpatch*nch, ..., ...)
                embed = embed.reshape(nbatch, embed.shape[1], npatch).permute(0, 2, 1)  # (nbatch, npatch, d_patch*nch) / (nbatch, npatch, dembed)

            if (self.model[1] == 'transformer') | (self.model[1] == 'conformer'):
                if self.use_cls:
                    cls_token = self.cls_token.expand(nbatch, -1, -1)
                    embed = torch.cat((embed, cls_token), dim=1)
                
                # Unwrap hidden states
                embed_outputs = self.embed.forward(embed, add_same_one)  # (nbatch, npatch, dembed), hidden_states
                
                if isinstance(embed_outputs, tuple):
                    if len(embed_outputs) == 3: # Hierarchical Conformer
                        embed = torch.cat([embed_outputs[0], embed_outputs[1]], dim=-1)
                        hidden_states = embed_outputs[2]
                    else: # Standard Encoder
                        embed = embed_outputs[0]
                        hidden_states = embed_outputs[1]
                else:
                    embed = embed_outputs
                    hidden_states = [embed_outputs]
        
        if len(self.model) == 1:
            if (self.model[0] == 'crnn') | (self.model[0] == 'crnn-sim') | (self.model[0] == 'tcrnn'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.crnn(embed)  # (nbatch, nt, dembed) = (nbatch, npatch, *) suited for frame-wise patch spliting only
            elif self.model[0] == 'fn':
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed, fb_skip, nb_skip = self.embed1(embed) # (nb, nf, nt, hid_dim)
                embed, fb_skip, nb_skip = self.embed2(embed, fullband_skip_in=fb_skip, narrband_skip_in=nb_skip) # (nb, nf, nt, hid_dim)
                # embed, fb_skip, nb_skip = self.embed3(embed, fullband_skip_in=fb_skip, narrband_skip_in=nb_skip) # (nb, nf, nt, hid_dim)
                embed = self.embed_proj(embed) # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 2, 1, 3) # (nbatch, nt, nf, nch)
                embed = embed.reshape(nbatch, embed.shape[1], -1)
                embed = self.t_embed_proj(embed)
            elif self.model[0] == 'nbc':
                if self.use_cls:
                    cls_token = self.cls_token.expand(nbatch, -1, -1)
                    embed = torch.cat((embed, cls_token), dim=1)
                embed = self.embed.forward(embed, add_same_one)  # (nbatch, npatch, dembed)
            elif self.model[0] == 'spatialnet':
                pass                
            elif (self.model[0] == 'resnet') | (self.model[0] == 'res2net') | (self.model[0] == 'densenet'):
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.cnn(embed)  # (nbatch, nch', nf, nt)  
                embed = self.cnn_proj(embed)  # (nbatch, dembed, nf, nt) 
                embed = embed.reshape(nbatch, embed.shape[1], -1).permute(0, 2, 1)  #  (nbatch, npatch, dembed)
            elif self.model[0] == 'unet':
                embed = embed.reshape(nbatch, npatch, dpatch, nch)
                embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, nch)
                embed = embed.permute(0, 3, 1, 2)  # (nbatch, nch, nf, nt)
                embed = self.cnn(embed)  # (nbatch, 1, nf, nt)  
                embed = self.cnn_proj(embed)  # (nbatch, dpatch, npatch_f, npatch_t) 
                embed = embed.reshape(nbatch, embed.shape[1], -1).permute(0, 2, 1)  #  (nbatch, npatch, dpatch)
            else:
                pass

        if 'hidden_states' not in locals():
            hidden_states = [embed]

        return embed, hidden_states

class EmbedDecoder(nn.Module):

    def __init__(self, sig_shape, patch_shape, dembed, model=['', 'fc'], use_cls=False):
        super(EmbedDecoder, self).__init__()
        self.model = model
        self.use_cls = use_cls
       
        nf, nt, nreim, nmic = sig_shape
        nch = nreim*nmic
        dpatch = patch_shape[0] * patch_shape[1]
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        npatch = npatch_shape[0] * npatch_shape[1]

        self.dpatch = dpatch
        dembed_out = dpatch*nreim*nmic
        
        if (model[0] == 'transformer') | (model[0] == 'conformer'):
            if self.use_cls:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, dembed))
                trunc_normal_(self.cls_token, std=.02)
                nseq = npatch + 1
            else:
                nseq = npatch
            mhsa_nlayer = 1
        if model[0] == 'transformer':
            self.embed = TransformerEncoder(d_model=dembed, nlayer=mhsa_nlayer, nhead=4, d_ff=dembed*4, nseq=nseq, dropout=0.1)
        elif model[0] == 'conformer':
            self.embed = ConformerEncoder(encoder_dim=dembed, num_layers=mhsa_nlayer, num_attention_heads=4, feed_forward_expansion_factor=4)
        elif model[0] == '':
            pass

        if model[1] == 'fc':
            dff = dembed_out * 3
            self.proj = nn.Sequential(
                nn.Linear(dembed, dff), 
                nn.ReLU(), 
                nn.Linear(dff, dembed_out)
                )
        elif model[1] == 'cnn':
            conv_chs = 64
            self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=patch_shape)
            self.proj = nn.Sequential(
                 nn.Conv2d(int(dembed/npatch), conv_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            	nn.BatchNorm2d(conv_chs),
            	nn.ReLU(inplace=True),
                nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            	nn.BatchNorm2d(conv_chs),
            	nn.ReLU(inplace=True),
                nn.Conv2d(conv_chs, conv_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            	nn.BatchNorm2d(conv_chs),
            	nn.ReLU(inplace=True),
                nn.Conv2d(conv_chs, nch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            	nn.BatchNorm2d(nch),
            	nn.ReLU(inplace=True),
                nn.Conv2d(nch, dembed_out, kernel_size=patch_shape, stride=patch_shape, padding=0, bias=False),
                )

    def forward(self, embed, add_same_one=False):
        nbatch = embed.shape[0]

        if (self.model[0] == 'transformer') | (self.model[0] == 'conformer'):
            if self.use_cls:
                cls_token = self.cls_token.expand(nbatch, -1, -1)
                embed = torch.cat((embed, cls_token), dim=1)
            embed = self.embed(embed, add_same_one)  # (nbatch, npatch, dembed)

        elif self.model[0] == '':
            pass

        if self.model[1] == 'fc':
            embed = self.proj(embed)  # (nbatch, npatch, dpatch)/(nbatch, npatch, dpatch*nch)

        elif self.model[1] == 'cnn':
            npatch = embed.shape[1]
            embed = embed.reshape(nbatch, npatch, self.dpatch, -1) # (nbatch, npatch, dpatch, x）
            embed = self.patch_recover.forward(embed)  # (nbatch, nf, nt, x)
            embed = embed.permute(0, 3, 1, 2)  # (nbatch, x, nf, nt)
            embed = self.proj(embed)  # (nbatch, dembed_out/npatch, ..., ...)
            embed = embed.reshape(nbatch, embed.shape[1], npatch).permute(0, 2, 1)  # (nbatch, npatch, dembed_out)
            
        elif self.model[0] == '':
            pass

        return embed

 
class SplitEmbedEncoder(nn.Module):
    def __init__(self, sig_shape, patch_shape, dembed, model, mode, use_cls, device):
        super().__init__()
        self.doa_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dembed, model=model, mode=mode, use_cls=use_cls, device=device, hierarchical=False)
        self.room_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dembed, model=model, mode=mode, use_cls=use_cls, device=device, hierarchical=False)
        
    def forward(self, embed, add_same_one=False):
        embed_doa, hs_doa = self.doa_encoder(embed, add_same_one)
        embed_room, hs_room = self.room_encoder(embed, add_same_one)
        
        embed_combined = torch.cat([embed_doa, embed_room], dim=-1)
        
        hidden_states = []
        for h_doa, h_room in zip(hs_doa, hs_room):
            hidden_states.append(torch.cat([h_doa, h_room], dim=-1))
            
        return embed_combined, hidden_states

class SARSSL(nn.Module): 

    def __init__(self, sig_shape=[256,256,2,2], patch_shape=(256,1), patch_mode='T', nmasked_patch=128*1, pretrain=True, pretrain_mode='masked', use_cls=False, downstream_token='all', downstream_head='mlp', downstream_embed='spec_spat', downstream_dlabel=1, device='cpu', pretrain_frozen_encoder=False, contrastive_learning=False, num_utterances=300, contrastive_learning_data_aug=False, obt=False, hsic=False, feature_sim_loss=False, diff_feat_selection=False, diff_feat_selection_lambda=0.01, feature_sim_loss_weight=1.0, feature_sim_sigma=1.0, feature_sim_temperature=0.3, hierarchical_feature_sim_room_temperature=0.2, hierarchical_feature_sim_room_sigma=1.0, hierarchical_feature_sim_doa_temperature=0.4, hierarchical_feature_sim_doa_sigma=0.4, spectral_feature_sim_loss=False, spectral_feature_sim_weight=1.0, spectral_feature_sim_sigma=1.0, spectral_feature_sim_temperature=0.3, nepoch=30, spec_dembed=512, spat_dembed=256, spectral_enc_input='all', debug=False, feature_stats=None, spectral_feature_stats=None, hierarchical=False, rec_loss=True, **kwargs):#vis=False):
        # patch_shape: size of each patch, (nf, nt)
        # patch_shape=(16,16), maskedpatch_mode='TF',
        # patch_shape=(256,1), maskedpatch_mode='T'
        super(SARSSL, self).__init__()

        nf, nt, nreim, nmic = sig_shape 
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        dpatch = patch_shape[0] * patch_shape[1]
        if nmasked_patch != (npatch_shape[0]*npatch_shape[1]//2):
            nmasked_patch = npatch_shape[0]*npatch_shape[1]//2
            assert nmasked_patch<npatch_shape[0]*npatch_shape[1], 'number of patches is out of range~'
            print('number of masks patches', nmasked_patch ,'/',npatch_shape[0]*npatch_shape[1])
        if patch_shape[1]!=1:
            f_first = True
        else:
            f_first = False

        self.pretrain = pretrain
        self.pretrain_frozen_encoder = pretrain_frozen_encoder
        self.contrastive_learning = contrastive_learning
        self.contrastive_learning_data_aug = contrastive_learning_data_aug
        self.num_utterances = num_utterances
        self.device = device
        self.use_cls = use_cls
        self.obt = obt
        self.hsic = hsic
        self.nepoch = nepoch
        self.pretrain_mode = pretrain_mode
        self.spectral_enc_input = spectral_enc_input
        self.debug = debug
        self.feature_sim_loss = feature_sim_loss
        self.diff_feat_selection = diff_feat_selection
        self.diff_feat_selection_lambda = diff_feat_selection_lambda
        self.feature_sim_loss_weight = feature_sim_loss_weight
        self.feature_sim_sigma = feature_sim_sigma
        self.feature_sim_temperature = feature_sim_temperature
        self.hierarchical_feature_sim_room_temperature = hierarchical_feature_sim_room_temperature
        self.hierarchical_feature_sim_room_sigma = hierarchical_feature_sim_room_sigma
        self.hierarchical_feature_sim_doa_temperature = hierarchical_feature_sim_doa_temperature
        self.hierarchical_feature_sim_doa_sigma = hierarchical_feature_sim_doa_sigma
        self.spectral_feature_sim_loss = spectral_feature_sim_loss
        self.spectral_feature_sim_weight = spectral_feature_sim_weight
        self.spectral_feature_sim_sigma = spectral_feature_sim_sigma
        self.spectral_feature_sim_temperature = spectral_feature_sim_temperature
        self.hierarchical = hierarchical
        self.hierarchical_w_stop_grad = kwargs.get('hierarchical_w_stop_grad', False)
        self.hierarchical_w_ch_att = kwargs.get('hierarchical_w_ch_att', False)
        self.hierarchical_stop_grad_branch = kwargs.get('hierarchical_stop_grad_branch', 'room')
        self.split_spatial_encoder = kwargs.get('split_spatial_encoder', False)
        self.hierarchical_adv = kwargs.get('hierarchical_adv', False)
        self.adv_weight = kwargs.get('adv_weight', 0.1)
        self.rec_loss = rec_loss
        self.factorized_sac = kwargs.get('factorized_sac', False)
        self.use_pcgrad = kwargs.get('use_pcgrad', False)
        self.normalization_mode = kwargs.get('normalization_mode', 'z_norm')
        self.oracle_supcon = kwargs.get('oracle_supcon', False)
        self.use_cross_attention = kwargs.get('use_cross_attention', False)
        self.family_specific_bn = kwargs.get('family_specific_bn', False)
        self.use_pfa_features = kwargs.get('use_pfa_features', False)
        self.use_pfa_features_w_hueristic = kwargs.get('use_pfa_features_w_hueristic', False)
        self.use_pfa_features_k7 = kwargs.get('use_pfa_features_k7', False)
        
        # Fixed normalization parameters
        self.feature_stats = feature_stats if feature_stats is not None else {} # Dictionary containing 'p1' and 'p99' for each feature
        self.spectral_feature_stats = spectral_feature_stats if spectral_feature_stats is not None else {}

        self.patch_split = at_module.PatchSplit(patch_shape=patch_shape, f_first=f_first)
        self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=patch_shape, f_first=f_first)

        # spec_dembed = 256*2
        # spat_dembed = 256
        self.in_ver = 'separate' # separate mask for spatial and spectral encoders
        # self.in_ver = 'same' # same mask for spatial and spectral encoders
        # self.in_ver = 'single_ch_each_patch' 

        if f_first:
            ## MC-Conformer (CNN_Fre_First+Conformer)
            spec_model = ['cnn_f_first', 'conformer']
            spat_model = ['cnn_f_first', 'conformer']
            print('fre first!')
        else:
            ## MC-Conformer (CNN+Conformer)
            spec_model = ['cnn', 'conformer']
            spat_model = ['cnn', 'conformer']
            ## Conformer
            # spec_model = ['fc', 'conformer']
            # spat_model = ['fc', 'conformer']
            ## CNN + Transformer
            # spec_model = ['cnn', 'transformer']
            # spat_model = ['cnn', 'transformer']
            ## Transformer
            # spec_model = ['fc', 'transformer']
            # spat_model = ['fc', 'transformer']

            ## CRNN
            # spec_model = ['crnn']
            # spat_model = ['crnn']
            ## CRNN-Simple
            # spec_model = ['crnn-sim']
            # spat_model = ['crnn-sim']
            ## TCRNN 
            # spec_model = ['tcrnn']
            # spat_model = ['tcrnn']
            # spec_encoder_dembed = 256
            # spat_encoder_dembed = int(256/2)

            ## ?? need to modify
            ## NBC 
            # spec_model = ['cnn', 'conformer']
            # spat_model = ['nbc']
            ## FN
            # spec_model = ['cnn', 'conformer']
            # spat_model = ['fn']
            ## SpatialNet
            # spec_model = ['cnn', 'conformer']
            # spat_model = ['spatialnet']

            ## ?? need to modify
            ## ResNet
            # spec_model = ['resnet']
            # spat_model = ['resnet']
            ## Res2Net
            # spec_model = ['res2net']
            # spat_model = ['res2net']
            ## DenseNet
            # spec_model = ['densenet']
            # spat_model = ['densenet']
            ## UNnet
            # spec_model = ['']
            # spat_model = ['unet']

        # Ablation study - only spatial encoder
        # spec_model = ['fc', '']
        # spec_encoder_dembed = 256*2 

        self.embed_use4ds = downstream_embed

        if (self.in_ver == 'separate') | (self.in_ver == 'same'):
            if self.spectral_enc_input == 'mono':
                 self.spec_encoder = EmbedEncoder(sig_shape=[nf, nt, nreim, 1], patch_shape=patch_shape, dembed=spec_dembed, model=spec_model, 
                                            mode='spec', use_cls=use_cls, device=self.device)
            else:
                 self.spec_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spec_dembed, model=spec_model, 
                                            mode='spec', use_cls=use_cls, device=self.device)
            if self.split_spatial_encoder:
                self.spat_encoder = SplitEmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spat_dembed, model=spat_model, 
                                                mode='spat', use_cls=use_cls, device=self.device)
            else:
                self.spat_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spat_dembed, model=spat_model, 
                                                mode='spat', use_cls=use_cls, device=self.device, hierarchical=self.hierarchical, hierarchical_w_stop_grad=self.hierarchical_w_stop_grad, hierarchical_w_ch_att=self.hierarchical_w_ch_att, hierarchical_stop_grad_branch=self.hierarchical_stop_grad_branch)

        elif self.in_ver == 'single_ch_each_patch':
            self.spec_encoder = EmbedEncoder(sig_shape=[nf*nmic, nt, nreim, 1], patch_shape=patch_shape, dembed=int(spec_dembed/nmic), model=spec_model,
                                            mode='spec', use_cls=use_cls, device=self.device)
            self.spat_encoder = EmbedEncoder(sig_shape=[nf*nmic, nt, nreim, 1], patch_shape=patch_shape, dembed=int(spat_dembed/nmic),  model=spat_model, 
                                            mode='spat', use_cls=use_cls, device=self.device)

        if self.pretrain:
            self.patch_mask = at_module.PatchMask(patch_mode=patch_mode, nmasked_patch=nmasked_patch, npatch_shape=npatch_shape, device=self.device)
            
            if use_cls:
                spec_dembed = spec_dembed + 1 
                spat_dembed = spat_dembed + 1         
            
            if self.hierarchical or self.split_spatial_encoder:
                 dec_dembed = spec_dembed + spat_dembed * 2
            else:
                 dec_dembed = spec_dembed + spat_dembed

            dec_model = ['', 'fc']
            # dec_model = ['conformer', 'fc']
            self.decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dec_dembed, model=dec_model, use_cls=False)
            
            if self.contrastive_learning or self.contrastive_learning_data_aug:
                # Projection head for Contrastive Loss (Spatial)
                self.cl_proj = nn.Sequential(
                    nn.Linear(spat_dembed, spat_dembed),
                    nn.BatchNorm1d(spat_dembed),
                    nn.ReLU(),
                    nn.Linear(spat_dembed, spat_dembed)
                )
                # Projection head for Contrastive Loss (Spectral)
                self.cl_proj_spec = nn.Sequential(
                    nn.Linear(spec_dembed, spec_dembed),
                    nn.BatchNorm1d(spec_dembed),
                    nn.ReLU(),
                    nn.Linear(spec_dembed, spec_dembed)
                )

            if self.feature_sim_loss:
                # Projection head for Feature Similarity Loss
                # Projects spatial embedding to a latent space for similarity comparison
                if getattr(self, 'use_cross_attention', False):
                    if self.oracle_supcon:
                        self.family_indices = {'tdoa': ['TDOA'], 'room': ['T60', 'DRR', 'C50', 'ABS']}
                    else:
                        self.family_indices = {'tdoa': ['tdoa'], 'room': ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max']}
                    self.num_active_families = len(self.family_indices)
                    self.family_queries = nn.Parameter(torch.empty(self.num_active_families, spat_dembed))
                    nn.init.normal_(self.family_queries, std=0.02)
                    self.cross_attention = nn.MultiheadAttention(embed_dim=spat_dembed, num_heads=8, batch_first=True)
                    
                    if getattr(self, 'family_specific_bn', False):
                        self.feat_sim_proj_lin1 = nn.Linear(spat_dembed, spat_dembed)
                        self.feat_sim_proj_bns = nn.ModuleList([nn.BatchNorm1d(spat_dembed) for _ in range(self.num_active_families)])
                        self.feat_sim_proj_relu = nn.ReLU()
                        self.feat_sim_proj_lin2 = nn.Linear(spat_dembed, 128)
                    else:
                        self.feat_sim_proj = nn.Sequential(
                            nn.Linear(spat_dembed, spat_dembed),
                            nn.BatchNorm1d(spat_dembed),
                            nn.ReLU(),
                            nn.Linear(spat_dembed, 128)
                        )
                elif self.hierarchical or self.split_spatial_encoder:
                    # Two projection heads, each taking the full embedding dimension (since we duplicated output)
                    split_dim = spat_dembed
                    self.feat_sim_proj_doa = nn.Sequential(
                        nn.Linear(split_dim, split_dim),
                        nn.BatchNorm1d(split_dim),
                        nn.ReLU(),
                        nn.Linear(split_dim, 128)
                    )
                    self.feat_sim_proj_room = nn.Sequential(
                        nn.Linear(split_dim, split_dim),
                        nn.BatchNorm1d(split_dim),
                        nn.ReLU(),
                        nn.Linear(split_dim, 128)
                    )
                else:
                    self.feat_sim_proj = nn.Sequential(
                        nn.Linear(spat_dembed, spat_dembed),
                        nn.BatchNorm1d(spat_dembed),
                        nn.ReLU(),
                        nn.Linear(spat_dembed, 128) # Smaller dim for similarity
                    )

            if self.spectral_feature_sim_loss:
                # Same as above but for spectral features
                self.feat_sim_proj_spec = nn.Sequential(
                    nn.Linear(spec_dembed, spec_dembed),
                    nn.BatchNorm1d(spec_dembed),
                    nn.ReLU(),
                    nn.Linear(spec_dembed, 128)
                )
                
            if self.feature_sim_loss and (self.diff_feat_selection or kwargs.get('fixed_feature_weights', False)):
                # Define candidate spatial features
                if self.oracle_supcon:
                    self.candidate_features = ['TDOA', 'T60', 'DRR', 'C50', 'ABS']
                    self.anchored_features = ['TDOA']
                else:
                    self.candidate_features = [
                        'coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max', 
                        'iacc_max', 'decay_slope_median', 'decay_slope_p10', 'decay_slope_p90',
                        'tdoa', 'ild_mean'
                    ]
                    self.anchored_features = ['tdoa']
                # Learnable weights initialized to 1.0
                self.spatial_feature_weights = nn.Parameter(torch.ones(len(self.candidate_features)))

            if self.hierarchical_adv and (self.hierarchical or self.split_spatial_encoder) and self.adv_weight > 0:
                from common.adversarial import GradientReversalLayer, TDOAAdversaryHead
                self.grl = GradientReversalLayer(lambda_adv=1.0)
                self.tdoa_adversary_head = TDOAAdversaryHead(in_features=spat_dembed)

        elif pretrain_frozen_encoder:
            self.patch_mask = at_module.PatchMask(patch_mode=patch_mode, nmasked_patch=nmasked_patch, npatch_shape=npatch_shape, device=self.device)
            
            if use_cls:
                spec_dembed = spec_dembed + 1
                spec_dembed = spat_dembed + 1 
            dec_dembed = spec_dembed + spat_dembed

            de_model = ['', 'fc']
            self.spec_spat_decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dec_dembed, model=de_model, use_cls=False)
            self.spec_decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spec_dembed, model=de_model, use_cls=False)
            self.spat_decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=spec_dembed, model=de_model, use_cls=False)

        else:
            # specify downstream embedding
            real_spat_dembed = spat_dembed
            if self.embed_use4ds == 'spec_spat':
                dembed_ds = spec_dembed + real_spat_dembed
            elif self.embed_use4ds == 'spec':
                dembed_ds = spec_dembed+0
            elif self.embed_use4ds == 'spat':
                dembed_ds =  real_spat_dembed +0
            elif self.embed_use4ds == 'noinfo':
                dembed_ds = spec_dembed+0

            # specify pred head for fine-tuning
            if downstream_head == 'mlp':
                if downstream_dlabel == 1:
                    self.mlp_head = nn.Sequential(
                        nn.LayerNorm(dembed_ds), 
                        nn.Linear(dembed_ds, downstream_dlabel)
                        )
                else:
                    self.joint_head = nn.Sequential(
                    nn.LayerNorm(dembed_ds), 
                    nn.Linear(dembed_ds, dembed_ds),
                    nn.ReLU(),
                    nn.Linear(dembed_ds, downstream_dlabel)
                    )
            # elif downstream_head == 'crnn_in':
            #     self.pred_head = CausCRNN_predhead(cnn_in_dim=nmic*nreim+int(dembed_ds/256), cnn_dim = 64, rnn_in_dim = 256, rnn_hid_dim = 256, fc_in_dim = 256, fc_out_dim = 1, res_flag = False, embed_add = 'in')
            # elif downstream_head == 'crnn_med':
            #     self.pred_head = CausCRNN_predhead(cnn_in_dim=nmic*nreim, cnn_dim = 64, rnn_in_dim = 256+dembed_ds, rnn_hid_dim = 256, fc_in_dim = 256, fc_out_dim = 1, res_flag = False, embed_add = 'med')
            # elif downstream_head == 'crnn_out':
            #     self.pred_head = CausCRNN_predhead(cnn_in_dim=nmic*nreim, cnn_dim = 64, rnn_in_dim = 256, rnn_hid_dim = 256, fc_in_dim = 256+dembed_ds, fc_out_dim = 1, res_flag = False, embed_add = 'out')

            self.downstream_head = downstream_head
            self.downstream_dlabel = downstream_dlabel
            self.ds_token = downstream_token

    def forward(self, x, indices=None, epoch=None, gen_metrics=False, precalc_spec_feats=None, gt_spatial_features=None):
        nbatch, nmic, nf, nt, nreim = x.shape
        
        if self.pretrain:
            ## Patch (frame) spliting
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)

            ## Mask generation
            if self.pretrain_mode == 'autoenc':
                npatch = vec_patch.shape[1]
                # No masking: pass full input to both encoders
                vec_patch_flat = vec_patch.reshape(nbatch, npatch, -1) # (nbatch, npatch, dpatch*nreim*nmic)

                if self.spectral_enc_input == 'mono':
                     # Take only the first channel for spectral encoder
                     dpatch = vec_patch.shape[2]
                     nch = 2 # nreim
                     vec_patch_flat_spec = vec_patch[:, :, :, :, 0].reshape(nbatch, npatch, -1) # (nbatch, npatch, dpatch*nreim)
                     embed_spec = self.spec_encoder.forward(vec_patch_flat_spec)[0]
                else:
                     embed_spec = self.spec_encoder.forward(vec_patch_flat)[0]
                embed_spat = self.spat_encoder.forward(vec_patch_flat, add_same_one=False)[0]
                
                # Decoding
                embed = torch.cat([embed_spec, embed_spat], dim=2)
                vec_patch_pred = self.decoder.forward(embed)
                
                # Reshape prediction
                dpatch = vec_patch.shape[2]
                vec_patch_pred = vec_patch_pred.reshape(nbatch, npatch, dpatch, 2, nmic)
                
                # Reconstruction Loss (MSE on full tensor)
                if self.rec_loss:
                    loss_gen = torch.mean((vec_patch_pred - vec_patch)**2)
                else:
                    loss_gen = torch.tensor(0.0, device=self.device)
                
                diff = torch.tensor(0.0, device=self.device) # No diff metric relevant here yet
                
                # Visualization data
                data_vis = {}
                data_vis['mask'] = torch.zeros_like(vec_patch) # No mask
                data_vis['pred'] = self.patch_recover(vec_patch_pred)
                data_vis['tar'] = self.patch_recover(vec_patch)
                data_vis['embed_spec'] = embed_spec
                data_vis['embed_spat'] = embed_spat

            else:
                mask_dense, mask_patch_dense, mask_ch_dense, mask_patch_idx, _ = self.patch_mask(data_shape=vec_patch.shape)  
                # mask_dense/mask_patch_dense/mask_ch_dense: (nbatch, npatch, dpatch, nmic), mask_patch_idx: (nbatch, nmasked_patch)

                ## Single-channel masking 
                npatch = vec_patch.shape[1]
                mask_dense_expand = mask_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1).to(vec_patch.device)
                mask_patch_dense_expand = mask_patch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1).to(vec_patch.device)
                mask_ch_dense_expand = mask_ch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1).to(vec_patch.device)
                vec_patch_mask = vec_patch * mask_dense_expand  # (nbatch, npatch, dpatch, 2, nmic)

                ## Embedding encoding
                if self.in_ver=='separate':
                    ## spectral encoder
                    vec_patch_mask_reshape_spec = vec_patch * (1-mask_patch_dense_expand) * mask_ch_dense_expand + vec_patch * mask_patch_dense_expand * (1-mask_ch_dense_expand) # (nbatch, npatch, d_patch, 2, nmic)
                    vec_patch_mask_reshape_spec = vec_patch_mask_reshape_spec.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*2*nmic)
                    embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape_spec)[0]  # (nbatch, npatch, dembed)
                    
                    ## spatial encoder
                    # decoupling 1 (f-wise random)
                    # random_f_value = torch.rand(nbatch, vec_patch.shape[2]).to(vec_patch.device) # (nb, nf)
                    # random_f_value = random_f_value[:, np.newaxis, :, np.newaxis, np.newaxis].expand_as(vec_patch)
                    # vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand * random_f_value

                    # decoupling 2 (tf-wise random)
                    # random_tf_value = torch.rand(nbatch, vec_patch.shape[1], vec_patch.shape[2]).to(vec_patch.device) # (nb, nt, nf)
                    # random_tf_value = random_tf_value[:, :, :, np.newaxis, np.newaxis].expand_as(vec_patch)
                    # vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand * random_tf_value

                    # decoupling 3 (local-tf-wise random)
                    # ntf_each_region = [16, 256] # (nt_each_region, nf_each_region)
                    # random_tf_region = torch.rand((nbatch, int(nt/ntf_each_region[0]), int(nf/ntf_each_region[1]))).to(vec_patch.device)
                    # random_tf_value = torch.repeat_interleave(torch.repeat_interleave(random_tf_region, ntf_each_region[0], dim=1), ntf_each_region[1], dim=2) # (nb, nt, nf)
                    # random_tf_value = random_tf_value[:, :, :, np.newaxis, np.newaxis].expand_as(vec_patch)
                    # vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand * random_tf_value
                    
                    vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand  # (nbatch, npatch, d_patch, 2, nreim)
                    vec_patch_mask_reshape_spat = vec_patch_mask_reshape_spat.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*nch)
                    embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape_spat, add_same_one=False)[0]  # (nbatch, npatch, dembed)
                    
                elif self.in_ver == 'single_ch_each_patch': # each patch only has single-channel signal
                    vec_patch_mask_reshape = torch.cat([vec_patch_mask[:,:,:,:,0], vec_patch_mask[:,:,:,:,1]], dim=1) # (nbatch, npatch*nmic, dpatch, nreim)
                    vec_patch_mask_reshape = vec_patch_mask_reshape.reshape(nbatch, npatch*nmic, -1)  # (nbatch, npatch*nmic, dpatch*nreim)
                    embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape)[0]  # (nbatch, npatch*nmic, dembed/nmic)
                    embed_spec = torch.cat([embed_spec[:, 0:npatch, :], embed_spec[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                    embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape)[0]  # (nbatch, npatch*nmic, dembed/nmic)
                    embed_spat = torch.cat([embed_spat[:, 0:npatch, :], embed_spat[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)

                elif self.in_ver == 'same': # without additional spectral/spatial masking operation 
                    vec_patch_mask_reshape = vec_patch_mask.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*nch)
                    embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape)[0]
                    embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape, add_same_one=False)[0]  # (nbatch, npatch, dembed)
            
                ## Embedding decoding
                embed = torch.cat([embed_spec, embed_spat], dim=2)
                vec_patch_pred = self.decoder.forward(embed)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, 2*dembed)

                ## Loss calculation for masked part
                tar_ch_patches = torch.sum(vec_patch * (1 - mask_ch_dense_expand), dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
                tar_anotherch_patches = torch.sum(vec_patch * mask_ch_dense_expand, dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
                
                dpatch = vec_patch.shape[2]
                vec_patch_pred = vec_patch_pred.reshape(nbatch, npatch, dpatch, 2, nmic)  # (nbatch, npatch, dpatch, 2, nmic)
                pred_patches = torch.sum(vec_patch_pred * (1 - mask_ch_dense_expand), dim=-1)  # (nbatch, npatch, dpatch, 2)
        
                if self.rec_loss:
                    loss, diff = self.gen_loss(pred_patches=pred_patches, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches)
                    loss_gen = loss
                else:
                    loss_gen = torch.tensor(0.0, device=self.device)
                    diff = torch.tensor(0.0, device=self.device)

                # Additional visualized data
                mask = mask_dense.clone().detach()
                tar = vec_patch.clone().detach()
                pred = vec_patch_pred.clone().detach()
                data_vis = {}
                data_vis['mask'], data_vis['pred'], data_vis['tar'] = self.vis_results(mask_patches=mask, pred_patches=pred, tar_patches=tar)
                data_vis['embed_spec'] = embed_spec.detach() if hasattr(embed_spec, 'detach') else embed_spec
                data_vis['embed_spat'] = embed_spat.detach() if hasattr(embed_spat, 'detach') else embed_spat
            
            loss_cl_spat = torch.tensor(0.0, device=self.device)
            loss_cl_spec = torch.tensor(0.0, device=self.device)
            obt_loss = torch.tensor(0.0, device=self.device)
            if self.debug:
                 debug_stats = {} # Initialize early for accumulation

            if (self.contrastive_learning or self.contrastive_learning_data_aug) and indices is not None:
                # Spatial CL
                z_spat = torch.mean(embed_spat, dim=1) # (nbatch, dembed)
                z_proj_spat = self.cl_proj(z_spat)
                
                if self.contrastive_learning:
                    label_spat = indices // self.num_utterances
                elif self.contrastive_learning_data_aug:
                    base_indices = (indices // 8) * 8
                    offset = indices % 8
                    mask = offset < 4
                    label_spat = torch.where(mask, indices, base_indices)
                
                loss_cl_spat, cl_stats_spat = self.cl_loss(z_proj_spat, label_spat)
                if self.debug and cl_stats_spat:
                     for k, v in cl_stats_spat.items():
                         debug_stats[f'cl_spat_{k}'] = v
                
                # Spectral CL
                z_spec = torch.mean(embed_spec, dim=1) # (nbatch, dembed)
                z_proj_spec = self.cl_proj_spec(z_spec)

                if self.contrastive_learning:
                    label_spec = indices % self.num_utterances
                elif self.contrastive_learning_data_aug:
                    # Reuse previously computed base_indices, offset, mask
                    label_spec = torch.where(mask, base_indices, indices)

                loss_cl_spec, cl_stats_spec = self.cl_loss(z_proj_spec, label_spec)
                if self.debug and cl_stats_spec:
                     for k, v in cl_stats_spec.items():
                         debug_stats[f'cl_spec_{k}'] = v
                
            if self.obt or self.hsic:
                if epoch is not None and epoch > (self.nepoch / 3):
                    # OBT Orthogonality Loss
                    # Push spatial encoder to be orthogonal to spectral encoder
                    # Minimize cross-covariance
                    # Flatten to (N, D)
                    z_spat_v = torch.mean(embed_spat, dim=1) # (nbatch, dembed)
                    z_spec_v = torch.mean(embed_spec, dim=1) # (nbatch, dembed)
                    
                    if self.hsic:
                        # Minimize HSIC (independence)
                        obt_loss = at_module.hsic(z_spat_v, z_spec_v)
                    elif self.hierarchical or self.split_spatial_encoder:
                        # Center
                        dim = z_spat_v.shape[-1]//2
                        z_spat_doa_v = z_spat_v[:, :dim] - z_spat_v[:, :dim].mean(dim=0)
                        z_spat_room_v = z_spat_v[:, dim:] - z_spat_v[:, dim:].mean(dim=0)
                        
                        # Covariance
                        cov = (z_spat_doa_v.T @ z_spat_room_v) / (nbatch - 1)
                        
                        # Loss = sum of squared elements
                        obt_loss = cov.pow(2).sum()
                    else:
                        # Center
                        z_spat_v = z_spat_v - z_spat_v.mean(dim=0)
                        z_spec_v = z_spec_v - z_spec_v.mean(dim=0)
                        
                        # Covariance
                        cov = (z_spat_v.T @ z_spec_v) / (nbatch - 1)
                        
                        # Loss = sum of squared elements
                        obt_loss = cov.pow(2).sum()
                    
            # Final Loss Calculation
            # Organize losses to sum to 1 with equal weights
            losses = []
            if self.rec_loss:
                losses.append(loss_gen)
            else:
                 loss_gen = torch.tensor(0.0, device=self.device)

            if (self.contrastive_learning or self.contrastive_learning_data_aug) and indices is not None:
                losses.append(loss_cl_spat)
                losses.append(loss_cl_spec)
            
            if (self.obt or self.hsic) and epoch is not None and epoch > (self.nepoch / 3):
                 losses.append(obt_loss)
            
            
            if self.feature_sim_loss:
                # x is [nbatch, nmic, nf, nt, nreim]
                # spatial_features_from_stft expects [B, 2, F, T] complex
                # Assuming nmic=2. 
                x_complex = torch.view_as_complex(x) # [B, nmic, nf, nt]

                with torch.no_grad():
                    feats = at_module.spatial_features_from_stft(x_complex, sample_rate=16000)

                if self.oracle_supcon:
                    if gt_spatial_features is not None:
                        feats.update(gt_spatial_features)
                    else:
                        for k in ['TDOA', 'T60', 'DRR', 'C50', 'ABS']:
                            feats[k] = torch.zeros(nbatch, device=x.device)
                
                if getattr(self, 'use_cross_attention', False):
                    loss_feat_sim, feat_sim_metrics = self.sac_loss_cross_attention(embed_spat, feats, eps=1e-8, return_metrics=True)
                else:
                    loss_feat_sim, feat_sim_metrics = self.feature_similarity_loss_hierarchical(embed_spat, feats, return_metrics=True) if (self.hierarchical or self.split_spatial_encoder) else self.feature_similarity_loss(embed_spat, feats, return_metrics=True)
                loss_feat_sim = loss_feat_sim * self.feature_sim_loss_weight
                
                if self.diff_feat_selection:
                    # L1 Regularization Penalty (exclude anchored features)
                    anchored_feats = getattr(self, 'anchored_features', [])
                    learnable_weights = torch.stack([w for i, w in enumerate(self.spatial_feature_weights) if self.candidate_features[i] not in anchored_feats])
                    l1_penalty = self.diff_feat_selection_lambda * torch.sum(torch.abs(learnable_weights))
                    loss_feat_sim = loss_feat_sim + l1_penalty
                    if self.debug:
                        debug_stats['l1_penalty'] = l1_penalty.item()
                
                losses.append(loss_feat_sim)
                
                if gen_metrics:
                    data_vis['metrics'] = feat_sim_metrics

                if feat_sim_metrics is not None:
                     if 'diagnostics' in feat_sim_metrics:
                         data_vis['diagnostics'] = feat_sim_metrics['diagnostics']
                     elif 'doa' in feat_sim_metrics and 'room' in feat_sim_metrics:
                         doa_diag = feat_sim_metrics['doa'].get('diagnostics', {}) if feat_sim_metrics['doa'] else {}
                         room_diag = feat_sim_metrics['room'].get('diagnostics', {}) if feat_sim_metrics['room'] else {}
                         # prefix keys
                         doa_diag = {f'DOA_{k}': v for k, v in doa_diag.items()}
                         room_diag = {f'ROOM_{k}': v for k, v in room_diag.items()}
                         data_vis['diagnostics'] = {**doa_diag, **room_diag}
                         
                     if getattr(self, 'factorized_sac', False):
                         if 'factorized_losses' in feat_sim_metrics:
                             data_vis['factorized_losses'] = feat_sim_metrics['factorized_losses']
                         elif 'doa' in feat_sim_metrics and 'room' in feat_sim_metrics:
                             doa_losses = feat_sim_metrics['doa'].get('factorized_losses', {}) if feat_sim_metrics['doa'] else {}
                             room_losses = feat_sim_metrics['room'].get('factorized_losses', {}) if feat_sim_metrics['room'] else {}
                             data_vis['factorized_losses'] = {**doa_losses, **room_losses}

                if self.debug:
                    debug_stats['loss_feat_sim'] = loss_feat_sim.detach()
            
            if self.spectral_feature_sim_loss:
                x_complex = torch.view_as_complex(x) # [B, nmic, nf, nt]
                with torch.no_grad():
                    if precalc_spec_feats is not None:
                        feats_spec = precalc_spec_feats
                    else:
                        feats_spec = at_module.spectral_features_from_stft(x_complex, sample_rate=16000)
                
                loss_spec_feat_sim, spec_feat_sim_metrics = self.spectral_feature_similarity_loss(embed_spec, feats_spec, return_metrics=gen_metrics)
                loss_spec_feat_sim = loss_spec_feat_sim * self.spectral_feature_sim_weight
                losses.append(loss_spec_feat_sim)

                if gen_metrics:
                    data_vis['spec_metrics'] = spec_feat_sim_metrics

                if self.debug:
                    debug_stats['loss_spec_feat_sim'] = loss_spec_feat_sim.detach()

            if self.hierarchical_adv and hasattr(self, 'tdoa_adversary_head') and self.adv_weight > 0:
                z_spat_mean = embed_spat.mean(dim=1)
                z_room = z_spat_mean[:, embed_spat.shape[-1]//2:]
                
                z_room_grl = self.grl(z_room)
                predicted_tdoa = self.tdoa_adversary_head(z_room_grl)
                
                if 'feats' not in locals():
                    x_complex = torch.view_as_complex(x)
                    with torch.no_grad():
                        feats = at_module.spatial_features_from_stft(x_complex, sample_rate=16000)
                
                target_tdoa = feats['tdoa'] * 10000 # scale to ~1
                loss_adv = F.mse_loss(predicted_tdoa, target_tdoa)
                losses.append(loss_adv * self.adv_weight)
                
                if self.debug:
                    debug_stats['loss_adv'] = loss_adv.detach()

            loss = sum(losses) / len(losses)

            if self.debug:
                 # debug_stats already exists and has CL info
                 debug_stats['loss_gen'] = loss_gen.detach()
                 
                 # Embeddings stats
                 if 'embed_spat' in locals():
                     debug_stats['embed_spat_norm'] = torch.norm(embed_spat, dim=-1).mean().detach()
                     debug_stats['embed_spat_var'] = torch.var(embed_spat, dim=-1).mean().detach()
                 
                 if 'embed_spec' in locals():
                     debug_stats['embed_spec_norm'] = torch.norm(embed_spec, dim=-1).mean().detach()
                     debug_stats['embed_spec_var'] = torch.var(embed_spec, dim=-1).mean().detach()

                 if 'z_spat' in locals():
                     debug_stats['z_spat_norm'] = torch.norm(z_spat, dim=-1).mean().detach()
                     debug_stats['z_spat_var'] = torch.var(z_spat, dim=-1).mean().detach()

                 if 'z_spec' in locals():
                     debug_stats['z_spec_norm'] = torch.norm(z_spec, dim=-1).mean().detach()
                     debug_stats['z_spec_var'] = torch.var(z_spec, dim=-1).mean().detach()
                     
                 if (self.contrastive_learning or self.contrastive_learning_data_aug) and indices is not None:
                     debug_stats['loss_cl_spat'] = loss_cl_spat.detach()
                     debug_stats['loss_cl_spec'] = loss_cl_spec.detach()
                     
                 if self.obt:
                     debug_stats['obt_loss'] = obt_loss.detach()
                     
                 data_vis['debug_stats'] = debug_stats
            
            if 'loss_feat_sim' not in locals():
                loss_feat_sim = torch.tensor(0.0, device=self.device)
            if 'loss_spec_feat_sim' not in locals():
                loss_spec_feat_sim = torch.tensor(0.0, device=self.device)
            if 'loss_adv' not in locals():
                loss_adv = torch.tensor(0.0, device=self.device)

            def _tensorize_dict(d, device):
                if isinstance(d, dict):
                    return {k: _tensorize_dict(v, device) for k, v in d.items()}
                elif isinstance(d, (float, int)):
                    return torch.tensor(float(d), device=device)
                return d

            data_vis = _tensorize_dict(data_vis, self.device)
            return loss, diff, data_vis, loss_cl_spat, loss_cl_spec, obt_loss, loss_gen, loss_feat_sim, loss_spec_feat_sim, loss_adv


        elif self.pretrain_frozen_encoder:
            ## Patch (frame) spliting
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)

            ## Mask generation
            mask_dense, mask_patch_dense, mask_ch_dense, mask_patch_idx, _ = self.patch_mask(data_shape=vec_patch.shape)  
            # mask_dense/mask_patch_dense/mask_ch_dense: (nbatch, npatch, dpatch, nmic), mask_patch_idx: (nbatch, nmasked_patch)

            ## Single-channel masking 
            npatch = vec_patch.shape[1]
            mask_dense_expand = mask_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1).to(vec_patch.device)
            mask_patch_dense_expand = mask_patch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1).to(vec_patch.device)
            mask_ch_dense_expand = mask_ch_dense[:, :, :, np.newaxis, :].expand(-1, -1, -1, 2, -1).to(vec_patch.device)
            vec_patch_mask = vec_patch * mask_dense_expand  # (nbatch, npatch, dpatch, 2, nmic)

            ## Embedding encoding
            if self.in_ver=='separate':
                ## spectral information
                # vec_patch_mask_reshape_spec = vec_patch * (1-mask_patch_dense_expand) * mask_ch_dense_expand + vec_patch * mask_patch_dense_expand * (1-mask_ch_dense_expand) # (nbatch, npatch, d_patch, 2, nmic)
                vec_patch_mask_reshape_spec = vec_patch * (1-mask_patch_dense_expand) * mask_ch_dense_expand # only unmasked channel
                # vec_patch_mask_reshape_spec = vec_patch * mask_patch_dense_expand * (1-mask_ch_dense_expand) # only masked channel
                vec_patch_mask_reshape_spec = vec_patch_mask_reshape_spec.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*2*nmic)
                embed_spec = self.spec_encoder.forward(vec_patch_mask_reshape_spec)[0]  # (nbatch, npatch, dembed)

                ## spatial information
                vec_patch_mask_reshape_spat = vec_patch * mask_patch_dense_expand  # (nbatch, npatch, d_patch, 2, nreim)
                vec_patch_mask_reshape_spat = vec_patch_mask_reshape_spat.reshape(nbatch, npatch, -1)  # (nbatch, npatch, dpatch*nch)
                embed_spat = self.spat_encoder.forward(vec_patch_mask_reshape_spat, add_same_one=False)[0]  # (nbatch, npatch, dembed)

            ## Embedding decoding
            embed = torch.cat([embed_spec, embed_spat], dim=2)

            ## Loss calculation for masked part
            dpatch = vec_patch.shape[2]
            tar_ch_patches = torch.sum(vec_patch * (1 - mask_ch_dense_expand), dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
            tar_anotherch_patches = torch.sum(vec_patch * mask_ch_dense_expand, dim=-1).clone().detach()  # (nbatch, npatch, d_patch, 2)
            
            ## use spectral encoder
            # vec_patch_pred_spec = self.spec_decoder.forward(embed_spec)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, dembed)
            # pred_patches_spec = vec_patch_pred_spec.reshape(nbatch, npatch, dpatch, 2)  # (nbatch, npatch, dpatch, 2)
            # loss = self.gen_loss_spec(pred_patches=pred_patches_spec, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches, tar_maskch=False)
            # vec_patch_pred = pred_patches_spec
 
            ## use spatial encoder
            # vec_patch_pred_spat = self.spat_decoder.forward(embed_spat)  # (nbatch, npatch, dpatch*nch+dembed) / (nbatch, npatch, dembed)
            # pred_patches_spat = vec_patch_pred_spat.reshape(nbatch, npatch, dpatch, 2)  # (nbatch, npatch, dpatch, 2)
            # loss = self.gen_loss_spec(pred_patches=pred_patches_spat, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches, tar_maskch=False)
            # vec_patch_pred =  pred_patches_spat
            
            ## use spatial & spectral encoders
            vec_patch_pred = self.spec_spat_decoder.forward(embed)  # (nbatch, npatch, dpatch*nch)
            vec_patch_pred = vec_patch_pred.reshape(nbatch, npatch, dpatch, 2, nmic)  # (nbatch, npatch, dpatch, 2, nmic)
            pred_patches = torch.sum(vec_patch_pred * (1 - mask_ch_dense_expand), dim=-1)  # (nbatch, npatch, dpatch, 2)
            if self.rec_loss:
                loss = self.gen_loss_spec(pred_patches=pred_patches, tar_patches=tar_ch_patches, mask_idx=mask_patch_idx, tar_unmaskch_patches=tar_anotherch_patches, tar_maskch=True)
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            ## Additional visualized data
            mask = mask_dense.clone().detach()
            tar = vec_patch.clone().detach()
            pred = vec_patch_pred.clone().detach()
            data_vis = {}
            data_vis['mask'], data_vis['pred'], data_vis['tar'] = self.vis_results(mask_patches=mask, pred_patches=pred, tar_patches=tar)
            
            data_vis['mask'], data_vis['pred'], data_vis['tar'] = self.vis_results(mask_patches=mask, pred_patches=pred, tar_patches=tar)
            
            return loss, loss*0.0, data_vis, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        else:
            ## Patch (frame) spliting
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)

            ## Embedding encoding
            npatch = vec_patch.shape[1]
 
            if (self.in_ver == 'separate') | (self.in_ver == 'same'):
                vec_patch_reshape = vec_patch.reshape(nbatch, npatch, -1) # (nbatch, npatch, dpatch*nch) 
                
                if self.spectral_enc_input == 'mono':
                    vec_patch_spec = vec_patch[:, :, :, :, 0].reshape(nbatch, npatch, -1)
                    embed_spec, hs_spec = self.spec_encoder.forward(vec_patch_spec)
                else:
                    embed_spec, hs_spec = self.spec_encoder.forward(vec_patch_reshape)  # (nbatch, npatch, dembed)
                
                embed_spat, hs_spat = self.spat_encoder.forward(vec_patch_reshape, add_same_one=False)  # (nbatch, npatch, dembed)

            elif self.in_ver == 'single_ch_each_patch':
                vec_patch_reshape = torch.cat([vec_patch[:,:,:,:,0], vec_patch[:,:,:,:,1]], dim=1) # (nbatch, npatch*nmic, dpatch, nreim)
                vec_patch_reshape = vec_patch_reshape.reshape(nbatch, npatch*nmic, -1)  # (nbatch, npatch*nmic, dpatch*nreim)
                embed_spec, hs_spec = self.spec_encoder.forward(vec_patch_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spec = torch.cat([embed_spec[:, 0:npatch, :], embed_spec[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                hs_spec = [torch.cat([h[:, 0:npatch, :], h[:, npatch:npatch*2, :]], dim=2) for h in hs_spec]
                
                embed_spat, hs_spat = self.spat_encoder.forward(vec_patch_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spat = torch.cat([embed_spat[:, 0:npatch, :], embed_spat[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                hs_spat = [torch.cat([h[:, 0:npatch, :], h[:, npatch:npatch*2, :]], dim=2) for h in hs_spat]

            ## Downstream processing
            if self.hierarchical or self.split_spatial_encoder:
                dim = embed_spat.shape[-1] // 2
                if getattr(self, 'task', None) == 'TDOA':
                    embed_spat = embed_spat[:, :, :dim]
                    if isinstance(hs_spat, list):
                        hs_spat = [h[:, :, :dim] for h in hs_spat]
                else:
                    embed_spat = embed_spat[:, :, dim:]
                    if isinstance(hs_spat, list):
                        hs_spat = [h[:, :, dim:] for h in hs_spat]

            hidden_states = []
            if self.use_cls:
                if self.ds_token == 'cls':
                    embed = torch.cat([embed_spec[:, -1:, :], embed_spat[:, -1:, :]], dim=-1)
                    hidden_states = [torch.cat([hs1[:, -1:, :], hs2[:, -1:, :]], dim=-1) for hs1, hs2 in zip(hs_spec, hs_spat)]
                elif self.ds_token == 'all':
                    embed = torch.cat([embed_spec[:, :-1, :], embed_spat[:, :-1, :]], dim=-1)
                    hidden_states = [torch.cat([hs1[:, :-1, :], hs2[:, :-1, :]], dim=-1) for hs1, hs2 in zip(hs_spec, hs_spat)]
            else:
                if self.embed_use4ds == 'spec_spat':
                    embed = torch.cat([embed_spec, embed_spat], dim=2)
                    hidden_states = [torch.cat([hs1, hs2], dim=2) for hs1, hs2 in zip(hs_spec, hs_spat)]
                elif self.embed_use4ds == 'spec':
                    embed = embed_spec+0.0
                    hidden_states = hs_spec
                elif self.embed_use4ds == 'spat':
                    embed = embed_spat+0.0
                    hidden_states = hs_spat
                elif self.embed_use4ds == 'noinfo':
                    embed = torch.zeros_like(embed_spec, device=self.device).detach()
                    hidden_states = [torch.zeros_like(h, device=self.device).detach() for h in hs_spec]

            pred = torch.mean(embed, dim=1) # (nbatch, dembed*n)
            if self.downstream_head == 'mlp':
                if self.downstream_dlabel == 1:
                    pred = self.mlp_head(pred)
                else:
                    pred = self.joint_head(pred)

            elif (self.downstream_head == 'crnn_in') | (self.downstream_head == 'crnn_med'):  
                dpatch = vec_patch.shape[2]
                embed = embed.reshape(nbatch, npatch, dpatch, -1) # (nbatch, npatch, dembed*n) → (nbatch, npatch, dpatch, dembed*n/dpatch)
                embed = self.patch_recover(embed).permute(0, 3, 1, 2) # (nbatch, n, nf, nt)
                pred = self.pred_head(x, embed)

            elif self.downstream_head == 'crnn_out': 
                pred = self.pred_head(x, pred)
            return pred, torch.mean(embed, dim=1), hidden_states

    def gen_loss(self, pred_patches, tar_patches, mask_idx, tar_unmaskch_patches=None):
        """ Calculate the generative loss from predited masked patches and target patches 
			Args:	pred_patches - (nbatch, npatch, dpatch, 2)
					tar_patches  - (nbatch, npatch, dpatch, 2)
                    mask_idx     - (nbatch, nmasked_patch)
			Return: loss
		"""

        nmasked_patch = mask_idx.shape[-1]
        nbatch, _, dpatch, nreim = tar_patches.shape
        pred = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        tar = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        if tar_unmaskch_patches is not None:
            tar_unmaskch = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()

        for b_idx in range(nbatch):
            pred[b_idx, ...] = pred_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            tar[b_idx, ...] = tar_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            if tar_unmaskch_patches is not None:
                tar_unmaskch[b_idx, ...] = tar_unmaskch_patches[b_idx, mask_idx[b_idx], :, :]

        # Calculate the MSE loss
        loss = torch.mean((pred - tar)**2)
        if tar_unmaskch_patches is not None:
            diff = torch.mean((tar - tar_unmaskch)**2)
         
        return loss if tar_unmaskch_patches is None else loss, diff
    
    def gen_loss_spec(self, pred_patches, tar_patches, mask_idx, tar_unmaskch_patches=None, tar_maskch=True): 
        """ Calculate the generative loss from predited masked patches and target patches 
			Args:	pred_patches - (nbatch, npatch, dpatch, 2)
					tar_patches  - (nbatch, npatch, dpatch, 2)
                    mask_idx     - (nbatch, nmasked_patch)
			Return: loss
		"""
 
        nmasked_patch = mask_idx.shape[-1]
        nbatch, _, dpatch, nreim = tar_patches.shape
        pred = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        tar = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()  # (nbatch, nmaskedpatch, dpatch, 2)
        tar_unmaskch = torch.empty((nbatch, nmasked_patch, dpatch, nreim), device=self.device).float()

        for b_idx in range(nbatch):
            pred[b_idx, ...] = pred_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            tar[b_idx, ...] = tar_patches[b_idx, mask_idx[b_idx], :, :]  # (nbatch, npatch, dpatch, 2)
            tar_unmaskch[b_idx, ...] = tar_unmaskch_patches[b_idx, mask_idx[b_idx], :, :]

        # Calculate the MSE loss
        if tar_maskch:
            loss = torch.mean((pred - tar)**2)
        else:
            loss = torch.mean((pred - tar_unmaskch)**2)
         
        return loss 

    def cl_loss(self, z, labels, neutral_mask=None):
        """ Calculate Supervised Contrastive Loss
            Args: z - (nbatch, dim)
                  labels - (nbatch)
                  neutral_mask - (nbatch, nbatch) Optional mask where 1 indicates pairs to ignore (neutral)
            Returns: loss
        """
        device = z.device
        temperature = 0.07

        # Normalize embeddings
        z = F.normalize(z, dim=1)

        # Calculate similarity matrix
        sim_matrix = torch.matmul(z, z.T) / temperature # (nbatch, nbatch)
        exp_sim_matrix = torch.exp(sim_matrix)

        # get labels (rir index)
        # labels = indices // self.num_utterances # (nbatch)
        
        # Create masks
        self_mask = torch.eye(z.shape[0], device=device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        neg_mask = 1 - pos_mask
        
        # P(i) set size (number of positives excluding self)
        # We need to adjust p_i if neutral matches were counted as positives (unlikely given logic, but safe to check)
        # Actually p_i is sum of pos_mask. If a neutral pair was in pos_mask, we'd remove it later.
        # But for now p_i assumes all positives are valid.
        
        if neutral_mask is not None:
             # Ensure neutral mask has 0 on diagonal
             neutral_mask = neutral_mask.float() * (1 - self_mask)
             exp_sim_matrix = exp_sim_matrix * (1 - neutral_mask)


        # Denominator: sum over A(i) (all other samples)
        # sum_{a \in A(i)} exp(sim) = sum_{all} exp(sim) - exp(self_sim)
        denom_sum = exp_sim_matrix.sum(dim=1) - torch.diag(exp_sim_matrix) # (nbatch)

        # Numerator: exp(sim(z_i, z_p)) for p in P(i)
        # We need log(exp(sim_ip) / denom) = sim_ip/tau - log(denom)
        
        # logic:
        # L = sum_{i} -1/|P(i)| sum_{p in P(i)} log ( exp(sim_ip)/denom )
        #   = sum_{i} -1/|P(i)| sum_{p in P(i)} ( sim_ip - log(denom) )
        
        # Extract sim_ip for positive pairs (excluding self)
        # mask out self matches from pos_mask for selection
        pos_mask_no_self = pos_mask - self_mask
        
        if neutral_mask is not None:
             # Ensure neutral pairs are not treated as positives
             pos_mask_no_self = pos_mask_no_self * (1 - neutral_mask)

        # Recalculate p_i based on effective positives
        p_i = pos_mask_no_self.sum(dim=1)
        # Avoid division by zero if there are no positives
        p_i = torch.max(p_i, torch.ones_like(p_i))

        # We can implement this efficiently:
        # log_prob = sim_matrix - torch.log(denom_sum).unsqueeze(1)
        # loss_i = - (log_prob * pos_mask_no_self).sum(dim=1) / p_i
        # loss = loss_i.mean()
        
        log_prob = sim_matrix - torch.log(denom_sum).unsqueeze(1) # (nbatch, nbatch)
        
        # if True: # Debug printing
        #      with torch.no_grad():
        #          sim_mean = sim_matrix.mean().item()
        #          sim_std = sim_matrix.std().item()
        #          pos_count = pos_mask_no_self.sum().item()
        #          denom_mean = denom_sum.mean().item()
        #          print(f"CL Debug: Sim Mean: {sim_mean:.4f}, Sim Std: {sim_std:.4f}, Pos Count: {pos_count}, Denom Mean: {denom_mean:.4f}")
        #          if pos_count == 0:
        #              print("CL Debug: NO POSITIVES!")
        #          # Check if embeddings are collapsed
        #          z_std = z.std(dim=0).mean().item()
        #          print(f"CL Debug: Z std (dim=0): {z_std:.6f}")

        # Sum log probs for positive pairs
        sum_log_prob_pos = (log_prob * pos_mask_no_self).sum(dim=1) # (nbatch)
        
        loss = - (sum_log_prob_pos / p_i).mean()
        
        debug_stats = {}
        if self.debug:
             with torch.no_grad():
                  # Stats
                  sim_mean = sim_matrix.mean().item()
                  sim_std = sim_matrix.std().item()
                  
                  # Positive similarities (excluding self)
                  pos_vals = sim_matrix[pos_mask_no_self.bool()]
                  pos_sim_mean = pos_vals.mean().item() if pos_vals.numel() > 0 else 0.0
                  
                  # Negative similarities (excluding self, which is 0 in neg_mask)
                  # neg_mask excludes diagonal (since pos_mask includes it)
                  neg_vals = sim_matrix[neg_mask.bool()]
                  neg_sim_mean = neg_vals.mean().item() if neg_vals.numel() > 0 else 0.0
                  
                  denom_mean = denom_sum.mean().item()
                  avg_num_pos = p_i.mean().item()

                  debug_stats = {
                      'sim_mean': sim_mean,
                      'sim_std': sim_std,
                      'pos_sim_mean': pos_sim_mean,
                      'neg_sim_mean': neg_sim_mean,
                      'denom_mean': denom_mean,
                      'avg_num_pos': avg_num_pos
                  }

        return loss, debug_stats

    def feature_similarity_loss(self, embed_spat, spatial_features, temperature=None, eps=1e-8, return_metrics=False):
        """
        Calculate Feature Similarity Loss using a cue-weighted contrastive approach.
        
        Args:
            embed_spat: [B, npatch, dembed] or [B, dembed]
            spatial_features: dict of raw features
        """
        if temperature is None:
            temperature = self.feature_sim_temperature

        # 1. Aggregate embeddings: Mean over patches if necessary
        if embed_spat.dim() == 3:
            z = embed_spat.mean(dim=1) # [B, dembed]
        else:
            z = embed_spat
            
        z_proj = self.feat_sim_proj(z) # [B, 128]      
        
        # 2. & 3. Calculate Loss using helper
        # If Differentiable Feature Selection is Enabled OR we are using fixed weights:
        if (hasattr(self, 'diff_feat_selection') and self.diff_feat_selection) or getattr(self, 'fixed_feature_weights', False):
            feature_keys = self.candidate_features
            learned_weights = self.spatial_feature_weights
        else:
            if getattr(self, 'oracle_supcon', False):
                feature_keys = ['TDOA', 'T60', 'DRR', 'C50', 'ABS']
            elif getattr(self, 'use_pfa_features_k7', False):
                feature_keys = [
                    'tdoa', 'gcc_phat_max', 'decay_slope_median', 'early_late_ratio_median',
                    'iacc_max', 'ild_mean', 'mod_low_high_ratio'
                ]
            elif getattr(self, 'use_pfa_features_w_hueristic', False):
                feature_keys = [
                    'coherence_mean', 'ipd_cos_mean', 'ipd_sin_mean', 'early_late_ratio_median',
                    'decay_slope_median', 'hf_lf_ratio', 'ild_mean', 'mod_low_high_ratio',
                    'coherence_low', 'coherence_var', 'decay_slope_p90', 'tdoa', 'gcc_phat_max'
                ]
            elif getattr(self, 'use_pfa_features', False):
                feature_keys = [
                    'coherence_mean', 'ipd_cos_mean', 'ipd_sin_mean', 'early_late_ratio_median',
                    'decay_slope_median', 'hf_lf_ratio', 'ild_mean', 'mod_low_high_ratio',
                    'coherence_low', 'coherence_var', 'decay_slope_p90'
                ]
            else:
                feature_keys = ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max', 'tdoa']
            learned_weights = None

        sigma = self.feature_sim_sigma
        loss, metrics = self._calc_single_feat_sim_loss(z_proj, spatial_features, feature_keys, temperature, sigma, eps, return_metrics, learned_weights)

        return loss, metrics

    def feature_similarity_loss_hierarchical(self, embed_spat, spatial_features, doa_temperature=None, room_temperature=None, doa_sigma=None, room_sigma=None, eps=1e-8, return_metrics=False):
        """
        Hierarchical Feature Similarity Loss.
        Splits embedding into DOA and Room halves and calculates separate losses.
        
        DOA features: ['ild_mean', 'coherence_mean', 'tdoa']
        Room features: ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max', 'hf_lf_ratio']
        """
        if doa_temperature is None:
            doa_temperature = self.hierarchical_feature_sim_doa_temperature
        if room_temperature is None:
            room_temperature = self.hierarchical_feature_sim_room_temperature
        if doa_sigma is None:
            doa_sigma = self.hierarchical_feature_sim_doa_sigma
        if room_sigma is None:
            room_sigma = self.hierarchical_feature_sim_room_sigma

        if getattr(self, 'oracle_supcon', False):
            doa_features = ['TDOA']
            room_features = ['T60', 'DRR', 'C50', 'ABS']
        else:
            doa_features = ['tdoa']
            room_features = ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max']
        # 1. Aggregate and split embeddings
        if embed_spat.dim() == 3:
            z = embed_spat.mean(dim=1) # [B, dembed]
        else:
            z = embed_spat
            
        dim = z.shape[-1]
        z_doa = z[:, :dim//2]   # First half for DOA
        z_room = z[:, dim//2:]  # Second half for Room
        
        # 2. Project
        z_proj_doa = self.feat_sim_proj_doa(z_doa)   # [B, 128]
        z_proj_room = self.feat_sim_proj_room(z_room) # [B, 128]

        # 3. Calculate Loss for DOA
        loss_doa, metrics_doa = self._calc_single_feat_sim_loss(
            z_proj_doa, spatial_features, doa_features, doa_temperature, doa_sigma, eps, return_metrics)
        
        # 4. Calculate Loss for Room
        loss_room, metrics_room = self._calc_single_feat_sim_loss(
            z_proj_room, spatial_features, room_features, room_temperature, room_sigma, eps, return_metrics)
        
        loss = (loss_doa + loss_room) / 2.0
        
        metrics = None
        if return_metrics:
            metrics = {
                'doa': metrics_doa,
                'room': metrics_room
            }
        return loss, metrics

    def sac_loss_cross_attention(self, embed_spat, spatial_features, eps=1e-8, return_metrics=False):
        device = embed_spat.device
        B = embed_spat.shape[0]
        
        Q = self.family_queries.unsqueeze(0).expand(B, -1, -1)
        attn_output, _ = self.cross_attention(query=Q, key=embed_spat, value=embed_spat)
        
        if getattr(self, 'family_specific_bn', False):
            x = self.feat_sim_proj_lin1(attn_output) # [B, num_families, spat_dembed]
            x_normed = []
            for i in range(self.num_active_families):
                x_normed.append(self.feat_sim_proj_bns[i](x[:, i, :]))
            x = torch.stack(x_normed, dim=1)
            x = self.feat_sim_proj_relu(x)
            Z_families = self.feat_sim_proj_lin2(x)
        else:
            attn_output_flat = attn_output.reshape(B * self.num_active_families, -1)
            z_flat = self.feat_sim_proj(attn_output_flat)
            Z_families = z_flat.reshape(B, self.num_active_families, -1)
        
        total_loss = 0.0
        family_names = list(self.family_indices.keys())
        
        metrics = {'factorized_losses': {}, 'diagnostics': {}}
        
        import math
        for k in range(self.num_active_families):
            fam_name = family_names[k]
            feature_keys = self.family_indices[fam_name]
            
            c_family_list = []
            for key in feature_keys:
                 val = spatial_features[key]
                 if key in self.feature_stats:
                     stats = self.feature_stats[key]
                     val_norm = (val - stats['mean']) / (3 * stats['std'] + 1e-6)
                     val_norm = torch.clamp(val_norm, -1.0, 1.0)
                 else:
                     val_norm = val
                 c_family_list.append(val_norm.unsqueeze(1))
                 
            c_family = torch.cat(c_family_list, dim=1) # [B, num_features_in_family]
            z_family = Z_families[:, k, :]
            
            cue_dist = torch.cdist(c_family, c_family, p=2)
            
            if self.feature_stats is not None and 'family_medians' in self.feature_stats:
                median_dist = self.feature_stats['family_medians'].get(fam_name, None)
                if median_dist is not None:
                    local_sigma = median_dist / math.sqrt(math.log(2.0))
                else:
                    local_sigma = self.feature_sim_sigma * math.sqrt(len(feature_keys))
            else:
                local_sigma = self.feature_sim_sigma * math.sqrt(len(feature_keys))
                
            w = torch.exp(-(cue_dist / local_sigma) ** 2)
            z_norm = F.normalize(z_family, dim=-1)
            sim = torch.matmul(z_norm, z_norm.T) / self.feature_sim_temperature
            
            diag_mask = torch.eye(B, device=device, dtype=torch.bool)
            off_diag = ~diag_mask
            
            exp_sim = torch.exp(sim) * off_diag.float()
            den = exp_sim.sum(dim=1, keepdim=True) + eps
            log_prob = sim - torch.log(den)
            
            w_masked = w * off_diag.float()
            w_sum = w_masked.sum(dim=1, keepdim=True) + eps
            w_norm = w_masked / w_sum
            
            loss_per_sample = -(w_norm * log_prob).sum(dim=1)
            loss_family = loss_per_sample.mean()
            
            metrics['factorized_losses'][f'loss_{fam_name}'] = loss_family
            
            # Diagnostics: Uniformity and Alignment_Top10 (Reusing sim for zero computational overhead)
            cos_sim = sim * self.feature_sim_temperature
            sq_dist = 2.0 - 2.0 * cos_sim
            sq_dist_flat = sq_dist[off_diag]
            uniformity = torch.log(torch.exp(-2.0 * sq_dist_flat).mean()).item()
            
            w_flat = w[off_diag]
            topk_k = max(1, int(0.1 * w_flat.numel()))
            _, topk_idx = torch.topk(w_flat, topk_k)
            ali_top10 = cos_sim[off_diag][topk_idx].mean().item()
            
            # Add effective rank:
            z_centered = z_family - z_family.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (B - 1)
            eigenvalues = torch.linalg.eigvalsh(cov.float())
            eigenvalues = torch.relu(eigenvalues) + 1e-8
            p = eigenvalues / eigenvalues.sum()
            effective_rank = torch.exp(-torch.sum(p * torch.log(p))).item()
            
            metrics['diagnostics'][fam_name] = {
                'Uniformity': uniformity,
                'Alignment': ali_top10,
                'EffectiveRank': effective_rank
            }
            
            total_loss += loss_family
            
        loss = total_loss / self.num_active_families
        return loss, metrics

    def spectral_feature_similarity_loss(self, embed_spec, spectral_features_dict, eps=1e-8, return_metrics=False):
        """
        Calculate Spectral Feature Similarity Loss on a frame-by-frame basis.
        
        Args:
            embed_spec: [B, T, dembed]
            spectral_features_dict: dict of 'feature_matrix' [B, 16, T] and 'voiced_prob' [B, T]
        """
        device = embed_spec.device
        temperature = self.spectral_feature_sim_temperature
        sigma = self.spectral_feature_sim_sigma
        
        B, T, dembed = embed_spec.shape
        
        # 1. Prepare target features
        feature_matrix = spectral_features_dict['feature_matrix'] # [B, 16, T]
        voiced_prob = spectral_features_dict['voiced_prob'] # [B, T]
        
        num_features = feature_matrix.shape[1]
        feature_norm = torch.zeros_like(feature_matrix)
        
        feature_names = [
            'F0', 'Energy', 'ZCR', 'Spectral_Centroid', 'Spectral_Spread', 
            'Spectral_Skewness', 'Spectral_Kurtosis', 'Spectral_Rolloff', 
            'Spectral_Flux', 'Spectral_Flatness', 'MFCC_1', 'MFCC_2', 
            'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6'
        ]
        
        if return_metrics:
             raw_values = {name: [] for name in feature_names}
             norm_values = {name: [] for name in feature_names}
        
        # Normalize features
        if 'mean' in self.spectral_feature_stats and 'std' in self.spectral_feature_stats:
             means = self.spectral_feature_stats['mean'] # [16]
             stds = self.spectral_feature_stats['std'] # [16]
             for i in range(num_features):
                  val = feature_matrix[:, i, :]
                  mean = means[i]
                  std = stds[i]
                  v_norm = (val - mean) / (3 * std + 1e-6)
                  feature_norm[:, i, :] = torch.clamp(v_norm, -1.0, 1.0)
        else:
             feature_norm = feature_matrix

        # 2. Project embeddings
        # Reshape for projection
        z_flat = embed_spec.reshape(B * T, dembed)
        z_proj_flat = self.feat_sim_proj_spec(z_flat) # [B*T, 128]
        z_proj = z_proj_flat.view(B, T, 128)
        
        z_norm = F.normalize(z_proj, dim=-1) # [B, T, 128]
        
        diag = torch.eye(B, device=device, dtype=torch.bool)
        mask = ~diag
        
        loss_frames = []
        if return_metrics:
            metrics_frames = {
                'raw': raw_values,
                'norm': norm_values,
                'sim_weights': [],
                'sim_weights_norm': [],
                'cue_dist': [],
                'sim': []
            }
        
        # 3. Calculate frame-by-frame loss
        for t in range(T):
             # Features at time t
             f_t = feature_norm[:, :, t] # [B, 16]
             v_p_t = voiced_prob[:, t] # [B]
             
             # Calculate pairwise squared differences [B, B, 16]
             f_t_exp1 = f_t.unsqueeze(1).expand(B, B, num_features)
             f_t_exp2 = f_t.unsqueeze(0).expand(B, B, num_features)
             sq_diff = (f_t_exp1 - f_t_exp2) ** 2
             
             # Soft-mask the F0 (index 0) using joint voiced probabilities
             joint_v_prob = v_p_t.unsqueeze(1) * v_p_t.unsqueeze(0) # [B, B]
             sq_diff[:, :, 0] = sq_diff[:, :, 0] * joint_v_prob
             
             # Sum of squared differences
             cue_dist_sq = sq_diff.sum(dim=-1) # [B, B]
             cue_dist = torch.sqrt(cue_dist_sq + eps)
             
             w = torch.exp(- (cue_dist / sigma) ** 2)
             
             # Embedding similarity
             z_t = z_norm[:, t, :] # [B, 128]
             sim = torch.matmul(z_t, z_t.T) / temperature # [B, B]
             
             exp_sim = torch.exp(sim) * mask
             den = exp_sim.sum(dim=1, keepdim=True) + eps
             log_prob = sim - torch.log(den)
             
             w_mask = w * mask
             w_sum = w_mask.sum(dim=1, keepdim=True) + eps
             w_norm = w_mask / w_sum
             
             loss_t = - (w_norm * log_prob).sum(dim=1)
             loss_frames.append(loss_t.mean())
             
             if return_metrics:
                 for i, name in enumerate(feature_names):
                     metrics_frames['raw'][name].append(f_t[:, i].detach())
                     metrics_frames['norm'][name].append(feature_norm[:, i, t].detach())
                 metrics_frames['sim_weights'].append(w.detach())
                 metrics_frames['sim_weights_norm'].append(w_norm.detach())
                 metrics_frames['cue_dist'].append(cue_dist.detach())
                 metrics_frames['sim'].append(sim.detach())
                 
        total_loss = torch.stack(loss_frames).mean()
        
        metrics = None
        if return_metrics:
            # Stack the lists into tensors across time dimension (if desired) or keep as lists
            # We'll just return the dictionary of lists directly, the plotting code can handle lists of tensors.
            metrics = metrics_frames
            
        return total_loss, metrics

    def get_active_spatial_feature_weights(self):
        """Calculate and return active spatial feature weights."""
        if hasattr(self, 'spatial_feature_weights'):
            if getattr(self, 'fixed_feature_weights', False):
                # When weights are fixed, they are loaded as-is from JSON (already passed through sigmoid/softmax basically)
                w_sig = self.spatial_feature_weights
            else:
                w_sig = torch.sigmoid(self.spatial_feature_weights)
            
            anchored_feats = getattr(self, 'anchored_features', [])
            
            w_active = []
            for i, feat in enumerate(self.candidate_features):
                if feat in anchored_feats:
                    # Anchored features have a fixed weight of 1
                    w_active.append(torch.tensor(1.0, device=self.spatial_feature_weights.device))
                else:
                    w_active.append(w_sig[i])
            return torch.stack(w_active)
        return None

    def _calc_single_feat_sim_loss(self, z_proj, spatial_features, feature_keys, temperature, sigma, eps, return_metrics, learned_weights=None):
        """Helper to calculate cue-weighted contrastive loss for a single set of features."""
        
        kv_pairs = []
        raw_values = {}
        norm_values = {}
        
        if learned_weights is not None:
            #  w_sig = torch.sigmoid(learned_weights)
            w_active = self.get_active_spatial_feature_weights()
            if w_active is None:
                K = len(feature_keys)
                tau = 6.0
                w_active = K * torch.softmax(learned_weights / tau, dim=0)
        
        for i, k in enumerate(feature_keys):
            val = spatial_features[k] # [B]
            if return_metrics:
                raw_values[k] = val.detach()
             
            if k in self.feature_stats:
                if getattr(self, 'normalization_mode', 'z_norm') == 'cdf':
                    val_rank = torch.argsort(torch.argsort(val)).float() / (val.size(0) - 1.0 + 1e-6)
                    val_norm = val_rank * 2.0 - 1.0
                else:
                    stats = self.feature_stats[k]
                    mean = stats['mean']
                    std = stats['std']
                    val_norm = (val - mean) / (3 * std + 1e-6)
                    val_norm = torch.clamp(val_norm, -1.0, 1.0)
            else:
                val_norm = val
                
            if learned_weights is not None:
                val_norm = val_norm * w_active[i]
                 
            kv_pairs.append(val_norm.unsqueeze(1))
             
            if return_metrics:
                norm_values[k] = val_norm.detach()
        
        device = z_proj.device
        B = z_proj.shape[0]

        if getattr(self, 'factorized_sac', False):
            z_norm = F.normalize(z_proj, dim=-1)
            sim = torch.matmul(z_norm, z_norm.T) / temperature
            diag = torch.eye(B, device=device, dtype=torch.bool)
            mask = ~diag
            
            exp_sim = torch.exp(sim) * mask
            den = exp_sim.sum(dim=1, keepdim=True) + eps # [B, 1]
            log_prob = sim - torch.log(den) # [B, B]
            
            total_loss = 0.0
            factorized_losses = {}
            for i, k in enumerate(feature_keys):
                val_norm = kv_pairs[i] # [B, 1]
                cue_dist_k = torch.cdist(val_norm, val_norm, p=2)
                w_k = torch.exp(- (cue_dist_k / sigma) ** 2)
                
                w_mask_k = w_k * mask
                w_sum_k = w_mask_k.sum(dim=1, keepdim=True) + eps
                w_norm_k = w_mask_k / w_sum_k # [B, B]
                
                loss_per_sample_k = - (w_norm_k * log_prob).sum(dim=1)
                loss_k = loss_per_sample_k.mean()
                
                if learned_weights is not None:
                    loss_k = loss_k * w_active[i]
                
                factorized_losses[f'loss_{k}'] = loss_k
                total_loss += loss_k
                
            loss = total_loss / len(feature_keys)
            
            # Pack individual losses into metrics
            
            metrics = None
            if return_metrics:
                metrics = {
                    'raw': raw_values,
                    'norm': norm_values,
                    'sim_weights': None,
                    'sim_weights_norm': None,
                    'cue_dist': None,
                    'sim': sim.detach(),
                    'factorized_losses': factorized_losses
                }
            else:
                metrics = {'factorized_losses': factorized_losses}
            
            return loss, metrics
        else:
            cues = torch.cat(kv_pairs, dim=1) # [B, num_features]
            
            z_norm = F.normalize(z_proj, dim=-1)
            sim = torch.matmul(z_norm, z_norm.T) / temperature
            cue_dist = torch.cdist(cues, cues, p=2)
            w = torch.exp(- (cue_dist / sigma) ** 2)

            diag = torch.eye(B, device=device, dtype=torch.bool)
            mask = ~diag

            # 1. Prepare Denominator for Softmax (sum over k != i)
            # We use masking to ensure we don't sum the diagonal
            exp_sim = torch.exp(sim) * mask
            den = exp_sim.sum(dim=1, keepdim=True) + eps # [B, 1]
            
            # 2. Calculate Log-Probability: log( exp(s_ij) / sum(exp(s_ik)) )
            # log(a/b) = log(a) - log(b) = s_ij - log(den)
            log_prob = sim - torch.log(den) # [B, B]

            # 3. Normalize the Weights (sum over j != i should be 1)
            w_mask = w * mask
            w_sum = w_mask.sum(dim=1, keepdim=True) + eps
            w_norm = w_mask / w_sum # [B, B]

            # 4. Calculate SupCon Loss: Weighted Sum of Log-Probs
            # L_i = - sum_j ( w_norm_ij * log_prob_ij )
            loss_per_sample = - (w_norm * log_prob).sum(dim=1)
            loss = loss_per_sample.mean()
            
            # Diagnostics (Zero computational overhead)
            cos_sim = sim * temperature
            sq_dist = 2.0 - 2.0 * cos_sim
            sq_dist_flat = sq_dist[mask]
            uniformity = torch.log(torch.exp(-2.0 * sq_dist_flat).mean()).item()
            
            w_flat = w[mask]
            topk_k = max(1, int(0.1 * w_flat.numel()))
            _, topk_idx = torch.topk(w_flat, topk_k)
            ali_top10 = cos_sim[mask][topk_idx].mean().item()
            
            # Add effective rank:
            z_centered = z_proj - z_proj.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (z_proj.size(0) - 1)
            eigenvalues = torch.linalg.eigvalsh(cov.float())
            eigenvalues = torch.relu(eigenvalues) + 1e-8
            p = eigenvalues / eigenvalues.sum()
            effective_rank = torch.exp(-torch.sum(p * torch.log(p))).item()
            
            diagnostics = {'Global': {'Uniformity': uniformity, 'Alignment': ali_top10, 'EffectiveRank': effective_rank}}
        
        metrics = {'diagnostics': diagnostics}
        if return_metrics:
            metrics.update({
                'raw': raw_values,
                'norm': norm_values,
                'sim_weights': w.detach(),
                'sim_weights_norm': w_norm.detach(),
                'cue_dist': cue_dist.detach(),
                'sim': sim.detach()
            })
            
        return loss, metrics

    def vis_results(self, mask_patches, pred_patches, tar_patches):
        """ Visualized constructed results
			Args:	mask_patches 		- (nbatch, npatch, dpatch, nmic)
					pred_patches 		- (nbatch, npatch, dpatch, 2, nmic)
					tar_patches  		- (nbatch, npatch, dpatch, 2, nmic)
			Return: mask_patches_fold 	- (nbatch, nf, nt, 2, nmic)
					pred_patches_fold 	- (nbatch, nf, nt, 2, nmic)
					tar_patches_fold  	- (nbatch, nf, nt, 2, nmic)
		"""

        mask_patches_fold = self.patch_recover(mask_patches)
        pred_patches_fold = self.patch_recover(pred_patches)
        tar_patches_fold = self.patch_recover(tar_patches)

        return mask_patches_fold, pred_patches_fold, tar_patches_fold


class SARSSL_MultiCH(nn.Module): 
    def __init__(self, sig_shape, nmic_pair, task, device): 
        super(SARSSL_MultiCH, self).__init__()

        self.model_sch = SARSSL(sig_shape=sig_shape, pretrain=False, device=device, downstream_token='all', downstream_head='', downstream_embed='spat', downstream_dlabel=1)
        spec_dembed = 256*2
        spat_dembed = 256
        dembed_ds = spat_dembed
        if task == 'TDOA':
            factor = nmic_pair
        else:
            factor = 1
        self.head_mch = nn.Sequential(
                    nn.LayerNorm(dembed_ds*nmic_pair), 
                    nn.Linear(dembed_ds*nmic_pair, dembed_ds*nmic_pair),
                    nn.ReLU(),
                    nn.Linear(dembed_ds*nmic_pair, factor)
                    )
        self.nmic_pair = nmic_pair
                 
    def forward(self, x):
        nbatch_nmicpair, nmic, nf, nt, nreim = x.shape
        nbatch = nbatch_nmicpair//self.nmic_pair
        embed_sch, _ = self.model_sch(x) # (nb*nmic_pair, nembed)
        nembed = embed_sch.shape[-1]
        embed_sch = embed_sch.reshape(-1, self.nmic_pair*nembed)
        pred = self.head_mch(embed_sch)

        return pred, embed_sch


class MCConformer(nn.Module): 
    """ 
    patch_shape: size of each patch, (nf, nt)
    """
    def __init__(self, 
                 sig_shape=[256,256,2,2], 
                 patch_shape=(256,1), 
                 spec_model = ['cnn', 'conformer'],
                 spat_model = ['cnn', 'conformer'],
                 dembed={'spec': 512, 'spat': 256},
                 dec_model = ['', 'fc'], 
                 device='cpu'
                 ):
        ## MC-Conformer (CNN_Fre_First+Conformer)
        # spec_model = ['cnn_f_first', 'conformer']
        # spat_model = ['cnn_f_first', 'conformer']
        ## MC-Conformer (CNN+Conformer)
        # spec_model = ['cnn', 'conformer']
        # spat_model = ['cnn', 'conformer']
        ## Conformer
        # spec_model = ['fc', 'conformer']
        # spat_model = ['fc', 'conformer']
        ## CNN + Transformer
        # spec_model = ['cnn', 'transformer']
        # spat_model = ['cnn', 'transformer']
        ## Transformer
        # spec_model = ['fc', 'transformer']
        # spat_model = ['fc', 'transformer']
        # dec_model = ['', 'fc']
        # dec_model = ['conformer', 'fc']
        super(MCConformer, self).__init__()
        nf, nt, nreim, nmic = sig_shape
        npatch_shape = [int(nf / patch_shape[0]), int(nt / patch_shape[1])]
        dpatch = patch_shape[0] * patch_shape[1]
        self.dembed = dembed

        if patch_shape[1]!=1:
            f_first = True
        else:
            f_first = False

        self.patch_split = at_module.PatchSplit(patch_shape=patch_shape, f_first=f_first)
        self.patch_recover = at_module.PatchRecover(output_shape=(nf, nt), patch_shape=patch_shape, f_first=f_first)

        if f_first:
            if 'cnn' in spec_model[0]:
                spec_model[0] = 'cnn_f_first'
            if 'cnn' in spat_model[1]:
                spec_model[1] = 'cnn_f_first'
            
            print('fre first!')
        if dembed['spec'] > 0:
            self.spec_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dembed['spec'], model=spec_model, 
                                            mode='spec', use_cls=False, device=device)
        if dembed['spat'] > 0:
            self.spat_encoder = EmbedEncoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dembed['spat'], model=spat_model, 
                                        mode='spat', use_cls=False, device=device)


        dec_dembed = dembed['spec'] + dembed['spat']
        self.decoder = EmbedDecoder(sig_shape=sig_shape, patch_shape=patch_shape, dembed=dec_dembed, model=dec_model, use_cls=False)
        
    def forward(self, x):
        
        data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
        
        ## Patch (frame) spliting
        vec_patch = self.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)
        nbatch, npatch, dpatch, _, nmic = vec_patch.shape 
        vec_patch_reshape = vec_patch.reshape(nbatch, npatch, -1) # (nbatch, npatch, dpatch*nch) 
        
        ## Embedding encoding
        if (self.dembed['spec'] > 0) & (self.dembed['spat'] == 0):
            embed = self.spec_encoder.forward(vec_patch_reshape)[0]  # (nbatch, npatch, dembed_spec)
        elif (self.dembed['spec'] == 0) & (self.dembed['spat'] > 0):
            embed = self.spat_encoder.forward(vec_patch_reshape, add_same_one=False)[0]  # (nbatch, npatch, dembed_spat)
        elif (self.dembed['spec'] > 0) & (self.dembed['spat'] > 0):
            embed_spec = self.spec_encoder.forward(vec_patch_reshape)[0]  # (nbatch, npatch, dembed_spec)
            embed_spat = self.spat_encoder.forward(vec_patch_reshape, add_same_one=False)[0]  # (nbatch, npatch, dembed_spat)
            embed = torch.cat([embed_spec, embed_spat], dim=2) # (nbatch, npatch, dembed_spec+dembed_spat)

        ## Embedding decoding
        vec_patch_pred = self.decoder.forward(embed)  # (nbatch, npatch, dpatch*nch) 
        vec_patch_pred = vec_patch_pred.reshape(nbatch, npatch, dpatch, 2, nmic)  # (nbatch, npatch, dpatch, 2, nmic)
        
        ## Patch recovering
        data_pred = self.patch_recover(vec_patch_pred)  # (nbatch, nf, nt, nreim, nmic)
        
        return data_pred 


if __name__ == "__main__":
    import torch
    from common.utils import get_nparams

    nb = 100
    nf = 256
    nt = 256
    nmic = 2
    nreim = 2
    input = torch.randn((nb, nmic, nf, nt, nreim))

    net = SARSSL(pretrain=True)
    ouput = net(input)
    nparam, nparam_sum = get_nparams(net, ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head'])
    print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 2)) for key in nparam.keys()])

    net = SARSSL(pretrain=False)
    nparam, nparam_sum = get_nparams(net, ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head'])
    print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 3)) for key in nparam.keys()])

    net = MCConformer(
                sig_shape=[nf, nt, nreim, nmic], 
                patch_shape=(nf, 1), 
                spec_model = ['cnn', 'conformer'],
                spat_model = ['cnn', 'conformer'],
                dembed={'spec': 512, 'spat': 256},
                )
    ouput = net(input)
    nparam, nparam_sum = get_nparams(net, ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head'])
    print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 2)) for key in nparam.keys()])


    net = SARSSL_MultiCH(
                    sig_shape=(256,256,2,2), 
                    nmic_pair=3, 
                    task='TDOA', 
                    device='cpu')
    input = torch.rand((nb*3, 2, nf, nt, nreim))
    output = net(input)


