import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift

# Parametry anteny
Nr = 12  # Zwiększenie liczby elementów anteny
d = 0.5  # Odległość między elementami anteny (w długościach fali)
theta_soi = 0 / 360 * 2 * np.pi  # Ustawienie SOI na 0 stopni (główna wstęga)
nulls_deg = np.linspace(-60, 60, 11)  # Więcej nullów (5 po każdej stronie)
nulls_rad = np.asarray(nulls_deg) / 360 * np.pi  # Konwersja na radiany

# Start out with conventional beamformer pointed at theta_soi
w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1, 1)

# Loop through nulls
for null_rad in nulls_rad:
    # weights equal to steering vector in target null direction
    w_null = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(null_rad)).reshape(-1, 1)

    # scaling_factor (complex scalar) for w at nulled direction
    scaling_factor = w_null.conj().T @ w / (w_null.conj().T @ w_null)

    # Update weights to include the null, but scale the influence of nulls
    # Further reduce the influence of nulls to make side lobes smaller
    scaling_factor *= 0.2  # Zmniejszamy wpływ nulli na główną wiązkę jeszcze bardziej
    w = w - w_null @ scaling_factor  # sidelobe-canceler equation

# Plot beam pattern
N_fft = 1024
w = np.conj(w)  # or else our answer will be negative/inverted
w_padded = np.concatenate((w.squeeze(), np.zeros(N_fft - Nr)))  # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2)  # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB)  # normalize to 0 dB at peak
theta_bins = np.linspace(-np.pi, np.pi, N_fft)  # Map the FFT bins to angles in radians

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB)
# Add dots where nulls and SOI are
for null_rad in nulls_rad:
    ax.plot([null_rad], [0], 'or')
ax.plot([theta_soi], [0], 'og')  # Zielony punkt dla SOI
ax.set_theta_zero_location('N')  # 0 stopni na górze
ax.set_theta_direction(-1)  # Kierunek zgodny z ruchem wskazówek zegara
ax.set_thetagrids(np.arange(0, 360, 30))  # Oznaczenia co 30 stopni
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_ylim([-40, 1])  # Skala amplitudy od -40 dB do 0 dB
plt.show()
