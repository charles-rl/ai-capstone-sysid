import numpy as np

def extract_features(trajectory):
    # trajectory shape: (600, 4) ->[theta, omega, cos, sin]
    theta = trajectory[:, 0]
    omega = trajectory[:, 1]
    
    features =[
        np.std(theta), np.max(theta), np.min(theta), # Stats
        np.std(omega), np.max(np.abs(omega)),
        np.mean(theta[-50:]),                        # Steady state
        ((omega[:-1] * omega[1:]) < 0).sum(),        # Zero crossings
        np.argmax(np.abs(np.fft.rfft(theta)))        # Dominant frequency
    ]
    return np.array(features)

# Apply this to your whole dataset -> X_train shape: (50000, 8)

