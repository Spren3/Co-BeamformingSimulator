import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# Parametry anteny
Nr = 12  # Liczba elementów anteny
d = 0.5  # Odległość między elementami anteny (w długościach fali)
theta_soi = 0 / 360 * np.pi  # SOI na 0 stopni (główna wstęga)
nulls_deg = np.linspace(-60, 60, 11)  # Nulle (5 po każdej stronie)
nulls_rad = np.asarray(nulls_deg) / 360 * np.pi  # Konwersja na radiany

# Funkcja do obliczania wzoru promieniowania
def calculate_beam_pattern(Nr, d, theta_soi, nulls_rad):
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1, 1)
    for null_rad in nulls_rad:
        w_null = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(null_rad)).reshape(-1, 1)
        scaling_factor = w_null.conj().T @ w / (w_null.conj().T @ w_null)
        scaling_factor *= 0.2  # Redukcja wpływu nulli
        w = w - w_null @ scaling_factor

    N_fft = 1024
    w = np.conj(w)
    w_padded = np.concatenate((w.squeeze(), np.zeros(N_fft - Nr)))
    w_fft_dB = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2)
    w_fft_dB -= 8
    theta_bins = np.linspace(-np.pi, np.pi, N_fft)
    return theta_bins, w_fft_dB

# Funkcja do obracania wiązki
def rotate_beam_pattern(theta_bins, w_fft_dB, rotation_deg):
    rotation_rad = rotation_deg / 360  * np.pi
    theta_bins_rotated = (theta_bins + rotation_rad) % (2 * np.pi)
    return theta_bins_rotated, w_fft_dB

# Funkcja do obliczania mocy w danym kierunku
def calculate_power_at_angle(theta_bins, w_fft_dB, target_angle_deg):
    target_angle_rad = target_angle_deg / 360 * np.pi
    idx = np.argmin(np.abs(theta_bins - target_angle_rad))
    return w_fft_dB[idx]

# Funkcja do rysowania wzoru promieniowania
def plot_beam_pattern(theta_bins, w_fft_dB, theta_soi_deg, nulls_deg):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB, label="Wzór promieniowania")
    # for null_deg in nulls_deg:
    #     ax.plot([null_deg], [10], 'or')  # Czerwona kropka na wysokości 10 dB
    ax.plot([theta_soi_deg / 360 * np.pi], [np.max(w_fft_dB)], 'og', label="Główna wiązka")  # Zielona kropka dla głównej wiązki
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_ylim([-30, 15])  # Zakres zysku
    ax.legend(loc="upper right")
    plt.show()

# Przykład użycia
theta_soi_deg = 0  # Kąt głównej wiązki (np. STA1)
theta_bins, w_fft_dB = calculate_beam_pattern(Nr, d, theta_soi, nulls_rad)
theta_bins_rotated, w_fft_dB_rotated = rotate_beam_pattern(theta_bins, w_fft_dB, theta_soi_deg)

# STA2 odbiera moc i kąt
sta2_angle_deg = 270  # Kąt STA2 względem anteny
sta2_power = calculate_power_at_angle(theta_bins_rotated, w_fft_dB_rotated, sta2_angle_deg)

# Wyświetlenie wyników
# print(f"Moc odbierana przez STA2: {sta2_power:.2f} dB")
# print(f"Kąt STA2 względem anteny: {sta2_angle_deg} stopni")

# Rysowanie wykresu
plot_beam_pattern(theta_bins_rotated, w_fft_dB_rotated, theta_soi_deg, nulls_deg)

max_db = np.max(w_fft_dB)
max_index = np.argmax(w_fft_dB)
angle_at_max = np.degrees(theta_bins[max_index])
print(f"Maksymalna wartość: {max_db:.1f} dB przy kącie {angle_at_max:.1f}°")
# for angle, db in zip(np.degrees(theta_bins), w_fft_dB):
#     print(f"Kąt: {angle:.1f}°, dB: {db:.1f}")
