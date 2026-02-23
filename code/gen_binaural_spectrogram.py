
import os
import argparse
import soundfile as sf
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def generate_binaural_spectrogram(input_file, output_file):
    """
    Generates a PNG image of the STFT of a binaural recording.
    The image shows two spectrograms (one for each channel) with a diagonal offset,
    without axes or numbers.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    try:
        data, fs = sf.read(input_file)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return

    if data.ndim == 1:
        print("Warning: Input file is mono. Duplicating channel for visualization.")
        data = np.stack([data, data], axis=1)
    elif data.shape[1] < 2:
        print("Warning: Input file has less than 2 channels. Using channel 0 for both.")
        data = np.stack([data[:, 0], data[:, 0]], axis=1)

    # STFT parameters (standard for speech/audio)
    nperseg = 512
    noverlap = 256
    
    # Compute STFT for both channels
    f1, t1, Zxx1 = scipy.signal.stft(data[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    f2, t2, Zxx2 = scipy.signal.stft(data[:, 1], fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Convert to magnitude and log scale (dB)
    mag1 = np.abs(Zxx1) + 1e-8
    mag2 = np.abs(Zxx2) + 1e-8
    
    Sxx1 = 20 * np.log10(mag1)
    Sxx2 = 20 * np.log10(mag2)

    # Clipping for better contrast
    Sxx1 = np.clip(Sxx1, -80, None)
    Sxx2 = np.clip(Sxx2, -80, None)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    
    # Define axes positions for diagonal offset
    # Back one should be slightly above the front one
    # Smaller offset
    
    # Width and height of the spectrograms
    w, h = 0.65, 0.55
    
    # Front spectrogram (Channel 1) - Lower and Right
    # x, y
    f_x, f_y = 0.25, 0.25
    
    # Back spectrogram (Channel 2) - Higher and Left (behind)
    # offset
    off_x, off_y = 0.04, 0.06
    
    b_x, b_y = f_x - off_x, f_y + off_y

    # Back spectrogram (Channel 2)
    ax_back = fig.add_axes([b_x, b_y, w, h]) 
    
    # Front spectrogram (Channel 1)
    ax_front = fig.add_axes([f_x, f_y, w, h])

    # Plot on back axes
    # Use 'viridis' for lighter colors (yellow/green/blue) and no black
    im_back = ax_back.pcolormesh(t2, f2, Sxx2, shading='gouraud', cmap='viridis')
    ax_back.set_xticks([])
    ax_back.set_yticks([])

    # Plot on front axes
    im_front = ax_front.pcolormesh(t1, f1, Sxx1, shading='gouraud', cmap='viridis')
    
    # Make sure the front plot covers the back one properly.
    ax_front.set_facecolor('white') 
    ax_front.set_xticks([])
    ax_front.set_yticks([])

    # Keep frames (spines) visible, but can customize if needed
    # Default is visible, creating a box
    for ax in [ax_back, ax_front]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5) # Make frame slightly thicker
            spine.set_edgecolor('black')

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Spectrogram image saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binaural spectrogram image.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .png file")
    args = parser.parse_args()

    generate_binaural_spectrogram(args.input, args.output)
