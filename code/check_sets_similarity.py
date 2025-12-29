
import os
import argparse
import numpy as np
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import from data_generation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_generation.utils_src import WSJ0Dataset, explore_corpus
except ImportError:
    # If running from SAR-SSL/code, this should work. 
    # If not, try appending the parent directory or handle it.
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_generation'))
    from utils_src import WSJ0Dataset, explore_corpus

def load_data_info(data_dir, override_src_dir=None):
    """
    Load all_info.npz and reconstruct the utterance/speaker info 
    by replicating the RNG logic from gen_simu.py.
    """
    info_path = os.path.join(data_dir, 'all_info.npz')
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return None

    try:
        data = np.load(info_path, allow_pickle=True)
        args = data['args'].item()  # Extract dictionary from 0-d array
        cfgs = data['cfgs']
    except Exception as e:
        print(f"Error loading {info_path}: {e}")
        return None

    # Extract relevant args
    fs = args.get('fs', 16000)
    T = args.get('T', 4.112)
    num_source_range = args.get('num_source_range', (1, 1))
    src_dir = args.get('src_dir', '')
    stage = args.get('stage', 'pretrain')
    data_num = args.get('data_num', 1024)
    seed_base = int(args.get('seed', 1)) 
    
    if override_src_dir:
        src_dir = override_src_dir
        print(f"Using overridden source directory: {src_dir}")
    else:
        # Try to resolve relative to SAR-SSL/code if the stored path was relative
        # The stored path might be relative to where gen_simu.py was run.
        # Common pattern in this repo seems to be ../../../data or ../../data
        
        # Check if the stored path exists as is
        if not os.path.exists(src_dir):
             # Try resolving relative to CWD
             potential_path = os.path.abspath(os.path.join(os.getcwd(), src_dir))
             if os.path.exists(potential_path):
                 src_dir = potential_path
             else:
                 # Try common fallback: ../../data/SrcSig/wsj0 (relative to code/)
                 fallback_path = os.path.abspath(os.path.join(os.getcwd(), '../../data/SrcSig/wsj0'))
                 if os.path.exists(fallback_path):
                     print(f"Warning: Stored src_dir {src_dir} not found. Found fallback at {fallback_path}")
                     src_dir = fallback_path
                 else:
                     # Attempt to find SrcSig in nearby directories
                     # This is specific to the user's observed path failure
                     fallback_path_2 = os.path.abspath(os.path.join(os.getcwd(), '../../../data/SrcSig/wsj0'))
                     if os.path.exists(fallback_path_2):
                         src_dir = fallback_path_2
                     
    
    # Re-construct seed based on stage if it wasn't captured correctly in args (args usually captures the passed arguments, but gen_simu modifies 'seed' variable locally).
    
    # Re-construct seed based on stage if it wasn't captured correctly in args (args usually captures the passed arguments, but gen_simu modifies 'seed' variable locally).
    # Inspecting gen_simu.py: 'args = locals().copy()' is called AFTER seed is set based on stage. So args['seed'] should be the correct base seed.
    
    contrastive_learning = args.get('contrastive_learning', False)
    num_utterances_for_cl = args.get('num_utterances_for_cl', 300)
    contrastive_learning_data_aug = args.get('contrastive_learning_data_aug', False)

    # Adjust src_dir path to be relative to where the script is run or absolute
    # The saved src_dir might be relative like "../../../data/SrcSig/wsj0"
    # We need to make sure we can find it.
    # We will assume the script is run from SAR-SSL/code and the relative paths in args hold true, 
    # OR we try to resolve it.
    
    # If src_dir is relative, we might need to adjust it relative to the original execution path.
    # However, we don't know the original execution PWD easily.
    # Let's try to use the src_dir as is, and warn if fails.
    
    # WSJ0Dataset expects the root path of WSJ0, but gen_simu modifies it:
    # if stage == 'pretrain': src_dir = src_dir + '/tr' ...
    # BUT this modification happens AFTER args capture in gen_simu.py?
    # Let's check gen_simu.py lines 242 vs 356.
    # Line 242: args = locals().copy()
    # Line 356: modification of src_dir happens LATER.
    # So args['src_dir'] is the ROOT dir (e.g. .../wsj0).
    
    # We need to replicate the suffix addition.
    real_src_dir = src_dir
    src_dir_for_exclusion = None
    
    if 'pretrain' in stage:
        real_src_dir = os.path.join(src_dir, 'tr')
    elif 'preval' in stage or 'val' in stage:
        real_src_dir = os.path.join(src_dir, 'dt')
    elif 'pretest' in stage or 'test' in stage:
        real_src_dir = os.path.join(src_dir, 'et')
    elif 'train' in stage:
        real_src_dir = os.path.join(src_dir, 'tr')
        
    if 'pretest_vis' in stage:
       src_dir_for_exclusion = os.path.join(src_dir, 'tr')
       # Check if exclusion dir exists, if not try relative to CWD if potential_path used below
        
    # Check if directory exists
    if not os.path.exists(real_src_dir):
        # Try to resolve relative to SAR-SSL/code if the stored path was relative
        # Assumption: The user is running this script from SAR-SSL/code
        potential_path = os.path.abspath(os.path.join(os.getcwd(), real_src_dir))
        if os.path.exists(potential_path):
            real_src_dir = potential_path
            # Also update exclusion path if needed
            if src_dir_for_exclusion:
                src_dir_for_exclusion = os.path.join(potential_path, 'tr') # Assuming exclusion is from tr
        else:
             print(f"Warning: Source directory {real_src_dir} (derived from args) not found. Utterance reconstruction might fail.")
             print(f"Consider providing --src-dir argument.")

    exclude_spks = None
    if src_dir_for_exclusion and os.path.exists(src_dir_for_exclusion):
         print(f"DEBUG: Triggering exclusion logic for {stage}. Scanning {src_dir_for_exclusion}")
         try:
             # explore_corpus might fail if imports are messed up, handle gracefully or assume it works
             tr_corpus, _ = explore_corpus(src_dir_for_exclusion, 'wav')
             exclude_spks = list(tr_corpus.keys())
             print(f"DEBUG: Found {len(exclude_spks)} speakers to exclude.")
         except Exception as e:
             print(f"Error determining exclusion list: {e}")

    # Instantiate dataset
    try:
        if 'WSJ0Dataset' in str(WSJ0Dataset):
            # WSJ0Dataset args: path, T, fs, num_source
            srcdataset = WSJ0Dataset(
                path=real_src_dir,
                T=T,
                fs=fs,
                num_source=max(num_source_range),
                exclude_spk_ids=exclude_spks
            )
        else:
            print("Unknown dataset class.")
            return None
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return None

    # iterate to find utterances
    utterances = set()
    speakers = set()
    rooms = []

    # Logic from gen_simu.py GenerateRandomMicSig
    
    if contrastive_learning_data_aug:
        gen_range = range(data_num // 8)
    else:
        gen_range = range(data_num)
        
    for idx in gen_range:
        # Replicate Seed logic
        if contrastive_learning:
            src_idx_seed = idx % num_utterances_for_cl
            np.random.seed(seed=seed_base + src_idx_seed)
        else:
            np.random.seed(seed=seed_base + idx)

        if contrastive_learning:
             src_idx = idx % num_utterances_for_cl
        elif contrastive_learning_data_aug:
             src_idx = np.random.randint(0, len(srcdataset))
        else:
             src_idx = np.random.randint(0, len(srcdataset))
             
        # Replicating __getitem__ logic to get file paths:
        if src_idx < 0: src_idx = len(srcdataset.spkIDs) + src_idx
        elif src_idx >= len(srcdataset.spkIDs): src_idx = src_idx % len(srcdataset.spkIDs)
        
        current_spkID = srcdataset.spkIDs[src_idx]
        spkID_list = [current_spkID]
        idx_list = [src_idx]
        
        # Handling multiple sources (WSJ0Dataset lines 102-105)
        while len(set(spkID_list)) < srcdataset.sum_source:
             idx_othersources = np.random.randint(0, len(srcdataset.spkIDs))
             spkID_list.append(srcdataset.spkIDs[idx_othersources])
             idx_list.append(idx_othersources)
             
        # For each source, it picks an utterance (WSJ0Dataset lines 110-125)
        for source_idx in range(srcdataset.sum_source):
            # spkID = spkID_list[source_idx] # redundant
            spkWAVs = srcdataset.spkWAVs[idx_list[source_idx]]
            utt_paths = list(spkWAVs.values())
            
            # The Critical Random Choice
            utt_rng_idx = np.random.randint(0, len(utt_paths))
            
            # This is the utterance!
            utt_path = utt_paths[utt_rng_idx]
            
            utterances.add(utt_path)
            speakers.add(spkID_list[source_idx])

        # Room info
        # From sa_cfgs[idx] (or sa_cfgs[idx*8] if data_aug?)
        # In contrastive_learning_data_aug, we loop `gen_range = range(data_num // 8)`. 
        # We use `sa_cfgs[idx]`. Wait.
        # In gen_simu.py lines 1023: `sa_cfg = sa_cfgs[idx]` 
        # BUT if contrastive_learning_data_aug, `idx` goes from 0 to data_num // 8.
        # But `sa_cfgs` was generated in lines 301-323.
        # `sa_cfgs` has length `data_num // 8`. 
        # So `sa_cfgs[idx]` is correct.
        
        # Room signature: Room Dimensions + T60 + Mic Array Config?
        # Let's just use room dimensions for "identical rooms" check roughly.
        cfg = cfgs[idx]
        room_sig = (
            tuple(np.round(cfg['room_sz'], 3)), 
            round(cfg['T60_sabine'], 3),
            # tuple(np.round(cfg['mic_pos'].flatten(), 3)) # Include mic pos?
        )
        rooms.append(room_sig)

    return {
        'utterances': utterances,
        'speakers': speakers,
        'rooms': set(rooms),
        'args': args
    }

def main():
    parser = argparse.ArgumentParser(description="Check similarity between two SAR-SSL datasets")
    parser.add_argument("dir1", type=str, help="First dataset directory (e.g. .../pretrain)")
    parser.add_argument("dir2", type=str, help="Second dataset directory (e.g. .../pretest)")
    parser.add_argument("--src-dir", type=str, default=None, help="Override source directory (e.g. ../../data/SrcSig/wsj0)")
    args = parser.parse_args()

    print(f"Analyzing {args.dir1}...")
    info1 = load_data_info(args.dir1, override_src_dir=args.src_dir)
    if info1 is None: return

    print(f"Analyzing {args.dir2}...")
    info2 = load_data_info(args.dir2, override_src_dir=args.src_dir)
    if info2 is None: return

    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)

    # 1. Utterances
    u1 = info1['utterances']
    u2 = info2['utterances']
    common_u = u1.intersection(u2)
    print(f"Utterances:")
    print(f"  Set 1 Unique: {len(u1 - u2)}")
    print(f"  Set 2 Unique: {len(u2 - u1)}")
    print(f"  Common:       {len(common_u)}")
    if len(common_u) > 0:
        print(f"  (Common Example: {list(common_u)[0]})")

    # 2. Speakers
    s1 = info1['speakers']
    s2 = info2['speakers']
    common_s = s1.intersection(s2)
    print(f"\nSpeakers:")
    print(f"  Set 1 Unique: {len(s1 - s2)}")
    print(f"  Set 2 Unique: {len(s2 - s1)}")
    print(f"  Common:       {len(common_s)}")

    # 3. Rooms
    r1 = info1['rooms']
    r2 = info2['rooms']
    common_r = r1.intersection(r2)
    print(f"\nIdentical Rooms (Dim + T60):")
    print(f"  Set 1 Unique: {len(r1 - r2)}")
    print(f"  Set 2 Unique: {len(r2 - r1)}")
    print(f"  Common:       {len(common_r)}")
    
    # 4. Args comparison
    print(f"\nConfiguration:")
    print(f"  Set 1 Stage: {info1['args'].get('stage')}")
    print(f"  Set 2 Stage: {info2['args'].get('stage')}")

if __name__ == "__main__":
    main()
