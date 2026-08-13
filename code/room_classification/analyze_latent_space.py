import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    parser = argparse.ArgumentParser(description="Analyze latent space of spatial embeddings")
    parser.add_argument('--debug_dir', type=str, required=True, help="Path to the debug output directory containing test_embeds.npy and test_physics.csv")
    args = parser.parse_args()

    # Load data
    embeds_path = os.path.join(args.debug_dir, 'test_embeds.npy')
    physics_path = os.path.join(args.debug_dir, 'test_physics.csv')

    if not os.path.exists(embeds_path) or not os.path.exists(physics_path):
        print("Error: Could not find required files in debug_dir.")
        return

    test_embeds = np.load(embeds_path)
    physics_df = pd.read_csv(physics_path)

    print("--- Latent Space Analysis ---")
    print(f"Loaded {test_embeds.shape[0]} embeddings of dimension {test_embeds.shape[1]}")

    # Metric 1: Macro-Class Silhouette Score
    macro_labels = physics_df['Macro_Class'].values
    sil_score = silhouette_score(test_embeds, macro_labels)
    print(f"\n1. Macro-Class Silhouette Score: {sil_score:.4f}")

    # Metric 2: Continuous Probing (Ridge Regression)
    print("\n2. Continuous Probing (Ridge Regression 80/20 Split):")
    # Prepare targets
    targets = physics_df[['Width', 'Depth', 'Height', 'Volume']].values
    
    X_train, X_test, y_train, y_test = train_test_split(test_embeds, targets, test_size=0.2, random_state=42)
    
    target_names = ['Width', 'Depth', 'Height', 'Volume']
    for i, name in enumerate(target_names):
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train[:, i])
        preds = reg.predict(X_test)
        
        r2 = r2_score(y_test[:, i], preds)
        mae = mean_absolute_error(y_test[:, i], preds)
        print(f"   {name:6s} -> R^2: {r2:.4f}, MAE: {mae:.4f}")

    # Metric 3: Linear Centered Kernel Alignment (CKA)
    print("\n3. Linear Centered Kernel Alignment (CKA):")
    
    # Scale X and Y
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(test_embeds)
    
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(targets)

    # Compute Gram matrices
    K = np.dot(X_scaled, X_scaled.T)
    L = np.dot(Y_scaled, Y_scaled.T)

    # Centering
    K_mean_row = K.mean(axis=0, keepdims=True)
    K_mean_col = K.mean(axis=1, keepdims=True)
    K_mean = K.mean()
    K_c = K - K_mean_row - K_mean_col + K_mean

    L_mean_row = L.mean(axis=0, keepdims=True)
    L_mean_col = L.mean(axis=1, keepdims=True)
    L_mean = L.mean()
    L_c = L - L_mean_row - L_mean_col + L_mean

    # Calculate CKA
    # tr(Kc * Lc) can be calculated efficiently as sum of element-wise product since matrices are symmetric
    trace_Kc_Lc = np.sum(K_c * L_c)
    trace_Kc_Kc = np.sum(K_c * K_c)
    trace_Lc_Lc = np.sum(L_c * L_c)
    
    cka = trace_Kc_Lc / np.sqrt(trace_Kc_Kc * trace_Lc_Lc)
    print(f"   Linear CKA Score: {cka:.4f}")

if __name__ == '__main__':
    main()
