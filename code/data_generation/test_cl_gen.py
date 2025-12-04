import os
import numpy as np
import soundfile as sf
from gen_simu import GenerateRandomMicSig
import shutil

def create_dummy_data(src_dir, noi_dir):
    os.makedirs(src_dir + '/tr/spk1', exist_ok=True)
    os.makedirs(src_dir + '/tr/spk2', exist_ok=True)
    os.makedirs(noi_dir, exist_ok=True)

    # Create dummy wav files
    sr = 16000
    dummy_audio = np.random.uniform(-1, 1, sr * 5)
    sf.write(src_dir + '/tr/spk1/utt1.wav', dummy_audio, sr)
    sf.write(src_dir + '/tr/spk1/utt2.wav', dummy_audio, sr)
    sf.write(src_dir + '/tr/spk2/utt1.wav', dummy_audio, sr)
    
    sf.write(noi_dir + '/noise.wav', dummy_audio, sr)

def verify_outputs(save_to, num_rooms, num_utts):
    stage = 'pretrain'
    info_file = os.path.join(save_to, stage, 'all_info.npz')
    if not os.path.exists(info_file):
        print(f"FAILED: {info_file} not found")
        return False
    
    data = np.load(info_file, allow_pickle=True)
    sa_cfgs = data['cfgs']
    
    expected_len = num_rooms * num_utts
    if len(sa_cfgs) != expected_len:
        print(f"FAILED: Expected {expected_len} configs, got {len(sa_cfgs)}")
        return False
    
    # Check spatial acoustics (should be same within room block)
    for i in range(num_rooms):
        base_idx = i * num_utts
        base_cfg = sa_cfgs[base_idx]
        for j in range(1, num_utts):
            curr_cfg = sa_cfgs[base_idx + j]
            # Compare room size as a proxy for equality
            if not np.allclose(base_cfg['room_sz'], curr_cfg['room_sz']):
                 print(f"FAILED: Room {i} utterance {j} has different room size")
                 return False
    
    # Check individual files for source index (should be same across rooms for same utt index)
    # We need to check the _info.npz files for each sample
    for j in range(num_utts):
        # Check if src_idx is same for this utt index across all rooms
        src_indices = []
        for i in range(num_rooms):
            idx = i * num_utts + j
            sample_info_file = os.path.join(save_to, stage, f'{idx}_info.npz')
            if not os.path.exists(sample_info_file):
                print(f"FAILED: {sample_info_file} not found")
                return False
            sample_data = np.load(sample_info_file, allow_pickle=True)
            src_indices.append(sample_data['src_idx'])
        
        # Verify all src_indices are the same
        if len(set(src_indices)) != 1:
             print(f"FAILED: Utt index {j} has different src_indices across rooms: {src_indices}")
             # Note: This might fail if src_idx logic in utils_simu_rir_sig.py is not working as expected
             # or if my understanding of "same utterance" implies same src_idx.
             # Yes, I implemented src_idx = idx % num_utts, so it should be exactly the same.
             return False
        else:
             print(f"PASSED: Utt index {j} has consistent src_idx {src_indices[0]}")

    print("PASSED: All checks passed")
    return True

if __name__ == "__main__":
    base_dir = '/tmp/test_cl'
    src_dir = base_dir + '/src'
    noi_dir = base_dir + '/noi'
    save_to = base_dir + '/out'
    
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    create_dummy_data(src_dir, noi_dir)
    
    try:
        GenerateRandomMicSig(
            mode='sig',
            stage='pretrain',
            src_dir=src_dir,
            noi_dir=noi_dir,
            save_to=save_to,
            use_gpu=False,
            contrastive_learning=True,
            num_rooms_for_cl=2,
            num_utterances_for_cl=2,
            data_num=10, # Should be ignored
            gpus=[] # No GPU
        )
        
        verify_outputs(save_to, 2, 2)
        
    except Exception as e:
        print(f"FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
