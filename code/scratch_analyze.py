import numpy as np
import os
import matplotlib.pyplot as plt

def analyze():
    # 1. Analyze sigma search results
    search_res_path = '/storage/yotam/SAR-SSL/SAR-SSL/metrics/sigma_search/search_results_multibatch.npy'
    if os.path.exists(search_res_path):
        res = np.load(search_res_path, allow_pickle=True).item()
        print("=== Sigma Search Recommendations ===")
        for bs, bs_res in res.items():
            print(f"Batch Size {bs}:")
            for k, v in bs_res['recommendations'].items():
                print(f"  {k}: {v:.4f}")
    else:
        print(f"File not found: {search_res_path}")

    # 2. Analyze the tau sweep trajectory
    tau_res_path = '/storage/yotam/SAR-SSL/SAR-SSL/metrics/tau_search/tau_sweep_results.npy'
    # we need to find the latest dir
    tau_dirs = sorted([d for d in os.listdir('/storage/yotam/SAR-SSL/SAR-SSL/metrics/tau_search') if os.path.isdir(os.path.join('/storage/yotam/SAR-SSL/SAR-SSL/metrics/tau_search', d))])
    if tau_dirs:
        latest_tau_dir = tau_dirs[-1]
        tau_res_path = f'/storage/yotam/SAR-SSL/SAR-SSL/metrics/tau_search/{latest_tau_dir}/tau_sweep_results.npy'
        if os.path.exists(tau_res_path):
            tau_res = np.load(tau_res_path, allow_pickle=True).item()
            print(f"\n=== Tau Sweep Results ({latest_tau_dir}) ===")
            for tau, data in tau_res.items():
                final_uni = data['uniformity'][-1] if data['uniformity'] else 'N/A'
                final_ali = data['alignment'][-1] if data['alignment'] else 'N/A'
                print(f"Tau {tau}: Final Uniformity = {final_uni}, Final Alignment = {final_ali}")

if __name__ == '__main__':
    analyze()
