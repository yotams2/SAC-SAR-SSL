import os
import glob
import uuid
import tqdm
import itertools
import importlib
import numpy as np
import scipy.signal
import soundfile as sf
import pandas as pd
from argparse import ArgumentParser
import multiprocessing as mp
from functools import partial

def beta2t60(beta, room_sz):
    """Calculate T60 using Sabine's formula given uniform reflection coefficient beta."""
    alpha = 1.0 - beta**2
    V = room_sz[0] * room_sz[1] * room_sz[2]
    Sa = 2 * (room_sz[0]*room_sz[1] + room_sz[0]*room_sz[2] + room_sz[1]*room_sz[2]) * alpha
    if Sa == 0:
        return 0.0
    return 0.161 * V / Sa

def get_room_classes():
    classes = []
    
    # Keep track of unique room shapes (ignoring rotation)
    unique_rooms = set()
    
    # Small Rooms
    widths = np.arange(1.5, 3.51, 1.0)
    depths = np.arange(2.5, 4.51, 1.0)
    heights = np.arange(2.5, 3.01, 0.5)
    for w, d, h in itertools.product(widths, depths, heights):
        shape = (min(w, d), max(w, d), h)
        if shape not in unique_rooms:
            unique_rooms.add(shape)
            classes.append(('Small', (float(w), float(d), float(h))))
            
    # Large Rooms
    widths = np.arange(6.0, 13.01, 1.0)
    depths = np.arange(6.0, 12.01, 2.0)
    heights = np.arange(2.5, 3.51, 1.0)
    for w, d, h in itertools.product(widths, depths, heights):
        shape = (min(w, d), max(w, d), h)
        if shape not in unique_rooms:
            unique_rooms.add(shape)
            classes.append(('Large', (float(w), float(d), float(h))))
            
    # Halls
    widths = np.arange(1.0, 3.01, 1.0)
    depths = np.arange(7.0, 13.01, 1.0)
    heights = np.arange(2.5, 3.51, 1.0)
    for w, d, h in itertools.product(widths, depths, heights):
        shape = (min(w, d), max(w, d), h)
        if shape not in unique_rooms:
            unique_rooms.add(shape)
            classes.append(('Hall', (float(w), float(d), float(h))))
        
    return classes

def init_worker(gpus):
    # Initialize GPU for this worker
    process_id = int(mp.current_process().name.split('-')[-1]) - 1
    gpu_id = gpus[process_id % len(gpus)]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import gpuRIR
    importlib.reload(gpuRIR)
    global gpuRIR_module
    gpuRIR_module = gpuRIR
    
    # Import NoiseSignal from utils_noise locally to avoid top-level issues
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils_noise import NoiseSignal
    global NoiseSignal_cls
    NoiseSignal_cls = NoiseSignal

def generate_sample(args, c=343.0, fs=16000):
    (split, sample_idx, cls_id, macro_label, room_sz, src_wav_path, save_dir) = args
    
    np.random.seed() # reset seed in worker
    room_sz = np.array(room_sz)
    
    # Physics params
    beta = np.random.uniform(0.88, 0.90)
    beta_array = np.array([beta]*6)
    
    # Mic array
    aperture = np.random.uniform(0.03, 0.20)
    min_dist_wall = 0.5
    margin = min_dist_wall + aperture / 2
    
    # Dynamic margin to prevent crash in narrow halls
    margin_x = min(margin, room_sz[0] / 2.0 - 0.01)
    margin_y = min(margin, room_sz[1] / 2.0 - 0.01)
    margin_z = min(min_dist_wall, room_sz[2] / 2.0 - 0.01)
    
    cx = np.random.uniform(margin_x, room_sz[0] - margin_x)
    cy = np.random.uniform(margin_y, room_sz[1] - margin_y)
    cz = np.random.uniform(margin_z, room_sz[2] - margin_z)
    
    theta = np.random.uniform(0, 2*np.pi)
    mic_pos = np.array([
        [cx + (aperture/2)*np.cos(theta), cy + (aperture/2)*np.sin(theta), cz],
        [cx - (aperture/2)*np.cos(theta), cy - (aperture/2)*np.sin(theta), cz]
    ])
    
    # Source position
    margin_x_src = min(min_dist_wall, room_sz[0] / 2.0 - 0.01)
    margin_y_src = min(min_dist_wall, room_sz[1] / 2.0 - 0.01)
    margin_z_src = min(min_dist_wall, room_sz[2] / 2.0 - 0.01)
    
    while True:
        sx = np.random.uniform(margin_x_src, room_sz[0] - margin_x_src)
        sy = np.random.uniform(margin_y_src, room_sz[1] - margin_y_src)
        sz = np.random.uniform(margin_z_src, room_sz[2] - margin_z_src)
        src_pos = np.array([sx, sy, sz])
        
        if np.linalg.norm(src_pos - np.array([cx, cy, cz])) >= 0.5:
            break
            
    # Simulate RIR
    T60 = beta2t60(beta, room_sz)
    Tdiff = gpuRIR_module.att2t_SabineEstimator(12, T60)
    Tmax = gpuRIR_module.att2t_SabineEstimator(40, T60)
    if T60 < 0.15: Tdiff = Tmax
    nb_img = gpuRIR_module.t2n(Tdiff, room_sz)
    
    rir = gpuRIR_module.simulateRIR(
        room_sz=room_sz, 
        beta=beta_array, 
        pos_src=np.array([src_pos]), 
        pos_rcv=mic_pos, 
        nb_img=nb_img, 
        Tmax=Tmax, 
        fs=fs, 
        Tdiff=Tdiff, 
        c=c
    )
    rir = rir[0].T # shape (n_samples, 2)
    
    # Load source wav
    src_sig, src_fs = sf.read(src_wav_path)
    if src_fs != fs:
        src_sig = scipy.signal.resample_poly(src_sig, up=fs, down=src_fs)
    if len(src_sig.shape) > 1:
        src_sig = src_sig[:, 0]
        
    # Limit source audio to exactly 4 seconds
    target_samples = int(4.0 * fs)
    while len(src_sig) < target_samples:
        src_sig = np.concatenate((src_sig, src_sig), axis=0)
    if len(src_sig) > target_samples:
        st = np.random.randint(0, len(src_sig) - target_samples + 1)
        src_sig = src_sig[st:st+target_samples]
        
    # Convolve
    mic_sig_l = scipy.signal.fftconvolve(src_sig, rir[:, 0], mode='full')
    mic_sig_r = scipy.signal.fftconvolve(src_sig, rir[:, 1], mode='full')
    mic_sig_clean = np.stack([mic_sig_l, mic_sig_r], axis=1)
    
    # Noise
    snr = np.random.uniform(15, 30)
    # Noise generation length should match the reverberated signal
    T_dur = len(mic_sig_clean) / fs
    noidataset = NoiseSignal_cls(T=T_dur, fs=fs, nmic=2, noise_type='diffuse_white', c=c)
    noise_sig = noidataset.generate_random_noise(mic_pos=mic_pos)
    
    if len(noise_sig) < len(mic_sig_clean):
        noise_sig = np.pad(noise_sig, ((0, len(mic_sig_clean) - len(noise_sig)), (0, 0)), mode='wrap')
    noise_sig = noise_sig[:len(mic_sig_clean)]
    
    mic_sig = noidataset.add_noise(mic_sig_clean, noise_sig, snr, mic_sig_dp=None)
    
    # Ensure final output is strictly 4 seconds (64,000 samples)
    # The 'full' convolution added the RIR tail length to the array
    mic_sig = mic_sig[:target_samples]
    
    max_val = np.max(np.abs(mic_sig))
    if max_val > 0.99:
        mic_sig = mic_sig * (0.99 / max_val)
        
    room_instance_id = uuid.uuid4().hex
    filename = f"{cls_id:03d}_{room_instance_id}.wav"
    out_path = os.path.join(save_dir, filename)
    
    sf.write(out_path, mic_sig, fs)
    
    return {
        'File_Path': os.path.abspath(out_path),
        'Macro_Class': macro_label,
        'Micro_Class_ID': cls_id,
        'Room_Instance_ID': room_instance_id,
        'Split': split,
        'Source_Wav': os.path.abspath(src_wav_path)
    }

def main():
    parser = ArgumentParser(description='Generate RevRIR binaural dataset')
    parser.add_argument('--train_num', type=int, default=8000)
    parser.add_argument('--val_num', type=int, default=2000)
    parser.add_argument('--test_num', type=int, default=2000)
    parser.add_argument('--src_dir_train', type=str, default='../../../data/SrcSig/wsj0/tr', help='Path to train clean speech directory')
    parser.add_argument('--src_dir_val', type=str, default='../../../data/SrcSig/wsj0/dt', help='Path to val clean speech directory')
    parser.add_argument('--src_dir_test', type=str, default='../../../data/SrcSig/wsj0/et', help='Path to test clean speech directory')
    parser.add_argument('--save_dir', type=str, default='../../data_room_classification')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of GPUs to use')
    args = parser.parse_args()

    classes = get_room_classes()
    print(f"Total defined room classes: {len(classes)}")

    splits = {
        'train': (args.train_num, args.src_dir_train),
        'val': (args.val_num, args.src_dir_val),
        'test': (args.test_num, args.src_dir_test)
    }

    for split_name, (num_samples, src_dir) in splits.items():
        if num_samples == 0:
            continue
            
        print(f"\nProcessing {split_name} split...")
        split_save_dir = os.path.join(args.save_dir, split_name)
        os.makedirs(split_save_dir, exist_ok=True)
        
        # Get all wav files in src_dir (recursive)
        wav_files = glob.glob(os.path.join(src_dir, '**', '*.wav'), recursive=True)
        if len(wav_files) == 0:
            print(f"Warning: No wav files found in {src_dir}. Skipping.")
            continue
            
        print(f"Found {len(wav_files)} source files for {split_name}.")
        
        metadata_path = os.path.join(args.save_dir, f'metadata_{split_name}.csv')
        existing_df = None
        existing_count = 0
        if os.path.exists(metadata_path):
            existing_df = pd.read_csv(metadata_path)
            existing_count = len(existing_df)
            print(f"Found existing {metadata_path} with {existing_count} samples.")
            
        samples_to_generate = num_samples - existing_count
        if samples_to_generate <= 0:
            print(f"Already have {existing_count} samples for {split_name} (target {num_samples}). Skipping generation.")
            continue
            
        print(f"Generating {samples_to_generate} additional samples to reach target {num_samples}.")

        # Prepare job arguments
        job_args = []
        for i in range(samples_to_generate):
            cls_id = np.random.randint(0, len(classes))
            macro_label, room_sz = classes[cls_id]
            src_wav_path = np.random.choice(wav_files)
            
            job_args.append((
                split_name,
                existing_count + i,
                cls_id,
                macro_label,
                room_sz,
                src_wav_path,
                split_save_dir
            ))
            
        # Multiprocessing
        pool = mp.Pool(processes=len(args.gpus), initializer=init_worker, initargs=(args.gpus,))
        
        results = []
        with tqdm.tqdm(total=samples_to_generate, desc=f"Generating {split_name}") as pbar:
            for res in pool.imap_unordered(generate_sample, job_args, chunksize=10):
                results.append(res)
                pbar.update()
                
        pool.close()
        pool.join()
        
        # Save metadata
        df = pd.DataFrame(results)
        if existing_df is not None:
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.sort_values(by=['Micro_Class_ID', 'Room_Instance_ID'], inplace=True)
        df.to_csv(metadata_path, index=False)
        print(f"Saved {len(df)} samples metadata to {metadata_path}")

if __name__ == '__main__':
    # Fix mp start method for CUDA
    mp.set_start_method('spawn', force=True)
    main()
