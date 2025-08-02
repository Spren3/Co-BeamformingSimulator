import numpy as np
import matplotlib.pyplot as plt
from math import log10, pow 
from beam_pattern import calculate_beam_pattern, calculate_power_at_angle, rotate_beam_pattern, plot_beam_pattern_cartesian
room = np.zeros((100,100))
ap1=np.array([5,20])
sta1=np.array([100,20])
sta2=np.array([15,40])
ap2=np.array([5,30])
sta3=np.array([15,20])
# aps = np.array([ap1, ap2])
stas = np.array([sta1, sta2]) #zmienne zeby zgrupowac stacje i AP do wyswietlania na wykresie
objects=np.array([ap1,ap2,sta1,sta2])
theta_bins, w_fft_dB=calculate_beam_pattern(12,0.5,0/360*np.pi,np.asarray(np.linspace(-60, 60, 11)) / 360* np.pi)
def angle_between(ref_point, target_point):
    """
        Oblicza kąt między punktem odniesienia (ref_point) a punktem docelowym (target_point)
    w układzie współrzędnych kartezjańskich przekształconym do biegunowego.
    
    Args:
        ref_point: np.array([x, y]) - punkt odniesienia, który będzie traktowany jako (0,0)
        target_point: np.array([x, y]) - punkt docelowy
        
    Returns:
        float: Kąt w stopniach (0-360) między punktami
    """
    isOnRight=False
    # Obliczamy różnice współrzędnych (przesunięcie punktu docelowego względem punktu odniesienia)
    dx = target_point[0] - ref_point[0]
    dy = target_point[1] - ref_point[1]
    if dx<0:
        isOnRight=True
    # Obliczamy kąt w radianach używając funkcji arctan2, która uwzględnia ćwiartki układu
    angle_rad = np.atan2(dy, dx)
    
    # Konwersja na stopnie i normalizacja do przedziału 0-360
    angle_deg = np.degrees(angle_rad)
    
    # Konwersja ujemnych kątów do przedziału 0-360
    # if angle_deg < 0:
    #     angle_deg += 360
        
    return angle_deg % 360

def angle_between_points_from_perspective(ap_coords, sta1_coords, sta2_coords=None):
    """
    Oblicza kąt między punktami STA z perspektywy punktu AP.
    
    Args:
        ap_coords: Tuple (x, y) - współrzędne punktu AP (punkt odniesienia)
        sta1_coords: Tuple (x, y) - współrzędne pierwszego punktu STA
        sta2_coords: Tuple (x, y) lub None - współrzędne drugiego punktu STA (opcjonalnie)
        
    Returns:
        float: Kąt w stopniach (0-360) jeśli podano tylko jeden punkt STA
        float: Kąt między punktami STA1 i STA2 z perspektywy AP jeśli podano dwa punkty STA
    """
    # Obliczamy kąt dla pierwszego punktu STA względem AP
    angle_sta1 = angle_between(ap_coords, sta1_coords)
    
    # Jeśli nie podano drugiego punktu STA, zwracamy tylko kąt dla pierwszego
    if sta2_coords is None:
        return angle_sta1
    
    # Obliczamy kąt dla drugiego punktu STA względem AP
    angle_sta2 = angle_between(ap_coords, sta2_coords)
    
    # Obliczamy różnicę kątów (kąt między punktami STA1 i STA2 z perspektywy AP)
    angle_diff = abs(angle_sta1 - angle_sta2)
    
    # Wybieramy mniejszy kąt (zawsze chcemy kąt < 180°)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
        
    return angle_diff

def get_beam_angles_and_delta(AP1, STA1, AP2, STA2):
    """
    Oblicza kąty wiązek nadawczych oraz różnicę przesunięcia (delta).
    
    Zwraca:
        α, β, γ, δ
    """
    alpha = angle_between(AP1, STA1)  # (1) f(AP1, STA1) -> α
    beta  = angle_between(AP2, STA1)  # (2a) f(AP2, STA1) -> β
    gamma = angle_between(AP2, STA2)  # (2b) f(AP2, STA2) -> γ

    delta = (gamma - beta) % 360      # Δ = γ - β z zachowaniem zakresu [0,360)

    return alpha, beta, gamma, delta
# k,kat=angle_between(sta2,ap2)
# sta2_antenna_power=calculate_power_at_angle(theta_bins,w_fft_dB,kat)
# print("kat w stopniach miedzy sta2 i ap2 to: ", kat, "a moc odbierana: ", sta2_antenna_power)
# def pickOtherAP(STA,AP):
#     min=100
#     # min=np.linalg.norm(STA-AP)
#     closestAP=[]
#     for i in range(len(aps)):
#         if np.array_equal(AP, aps[i]):
#             continue
#         d=np.linalg.norm(STA-aps[i])
#         if d<min:
#             min=d
#             closestAP=aps[i]
#     return d,closestAP
# dist,closest=pickOtherAP(sta3,ap1)
# print("najblizsza stacja do sta1 to: ",closest, "w odleglosci :",dist)
##parametry (mozna powiedziec ze globalne)
f= 2.4 #GHz
Tx_PWR = 20 # w dBm
noise=0.0000000004
Bp=10 # breaking point w metrach
room_size = 100  # w metrach

class calculations:
    def __init__(self,aps):
        self.aps=aps

    def pickOtherAP(self,STA,AP):
        min=100
        # min=np.linalg.norm(STA-AP)
        closestAP=[]
        for i in range(len(self.aps)):
            if np.array_equal(AP, self.aps[i]):
                continue
            d=np.linalg.norm(STA-self.aps[i])
            if d<min:
                min=d
                closestAP=self.aps[i]
        return d,closestAP

    def path_loss(self,d,f):
        P=35*log10(d/Bp)
        path_loss=40.05 + 20*log10((min(d,Bp)*f)/2.4)
        if (d>Bp):
            path_loss+=P
        return path_loss
    # def interf(self,d):
    #     print("txpwr to :",Tx_PWR,"a path loss: ", self.path_loss(d,f))
    #     interference=Tx_PWR-self.path_loss(d,f)
    #     interference_mW=pow(10,(interference/10))
    #     return interference_mW
    def interf(self, STA, AP, all_active_transmissions=None):
        total_interference_mW = 0  # Inicjalizacja bez szumu
        
        if all_active_transmissions is None:
            # Standardowa interferencja od innych AP bez beamformingu
            for interfering_AP in self.aps:
                if np.array_equal(interfering_AP, AP):
                    continue  # Pomijamy nadający AP
                    
                d_interferer = np.linalg.norm(STA-interfering_AP)
                interference_mW = pow(10, (Tx_PWR-self.path_loss(d_interferer, f))/10)
                total_interference_mW += interference_mW
        else:
            # Interferencja z uwzględnieniem beamformingu
            for tx_info in all_active_transmissions:
                tx_ap = tx_info['ap']
                if np.array_equal(tx_ap, AP):
                    continue  # Pomijamy nadający AP
                
                tx_beam_pattern = tx_info['beam_pattern']
                rx_sta = tx_info['sta']
                # Obliczamy odległość do interferera
                d_interferer = np.linalg.norm(STA-tx_ap)
                # Obliczamy kąt między stacją a interfererem
                # angle_interferer = angle_between(tx_ap, STA)
                angle_interferer=angle_between_points_from_perspective(tx_ap,STA,rx_sta)
                print("pozycje sta i ap int: ",STA,tx_ap,"kąt miedzy STA, a int: ",angle_interferer)

                
                # Obliczamy bazową interferencję
                base_interference = Tx_PWR - self.path_loss(d_interferer, f)
                
                # Obliczamy wzmocnienie z wiązki w kierunku stacji
                theta_bins, w_fft_dB = tx_beam_pattern
                interference_gain = calculate_power_at_angle(theta_bins, w_fft_dB, angle_interferer)
                
                # Łączna interferencja z uwzględnieniem beamformingu
                total_interference = base_interference + interference_gain
                print("path loss: ",self.path_loss(d_interferer,f),"beam int pwr: ",interference_gain,"kat sta od int: ",angle_interferer)
                interference_mW = pow(10, total_interference/10)
                
                total_interference_mW += interference_mW
        
        return total_interference_mW
    def SINR(self,STA,AP,antenna_gain=0,all_active_transmissions=None):
        d=np.linalg.norm(STA-AP)
        d2, clAP=self.pickOtherAP(STA, AP)
        print("Kontrolnie wypisanie wszystkiego: ",clAP,d2,self.path_loss(d,f),10*log10(self.interf(STA, AP, all_active_transmissions))) ##zakomentowane
        sinr=(Tx_PWR+antenna_gain)-(self.path_loss(d,f)+10*np.log10(self.interf(STA, AP, all_active_transmissions)+noise))
        return sinr
    
    def SSB(self,AP_pos, beam_angle, STA_pos, TX_PWR_AP=Tx_PWR, theta_bins=theta_bins, w_fft_dB=w_fft_dB, f=f, Bp=Bp):
        """
        Oblicza siłę sygnału (Signal Strength under Beamforming, SSB).
        
        Args:
            AP_pos: np.array([x, y]) - pozycja AP
            beam_angle: float - kąt nadawania AP (w stopniach, 0-360)
            STA_pos: np.array([x, y]) - pozycja stacji
            TX_PWR_AP: float - moc nadawcza AP (dBm)
            theta_bins, w_fft_dB: charakterystyka wiązki
            f: częstotliwość (GHz)
            Bp: breaking point (metry)
            
        Returns:
            float: SSB w dBm
        """
        # Odległość
        d = np.linalg.norm(STA_pos - AP_pos)
        # Path loss
        path_loss_val = calculations([AP_pos]).path_loss(d, f)
        # Kąt do STA względem AP
        angle_to_sta = angle_between(AP_pos, STA_pos)
        # Różnica kąta względem kierunku wiązki
        rel_angle = (angle_to_sta - beam_angle) % 360
        # Wzmocnienie anteny w tym kierunku
        antenna_gain = calculate_power_at_angle(theta_bins, w_fft_dB, rel_angle)
        print("antena gain: ",antenna_gain)
        # SSB = TX_PWR_AP - path_loss + antenna_gain
        ssb = TX_PWR_AP - path_loss_val + antenna_gain
        return ssb
    
    def SINR_beamforming(self,AP, AP_angle, STA, interfering_APs, interfering_angles, theta_bins, w_fft_dB, noise=noise):
        """
        Oblicza SINR z wykorzystaniem SSB.
        AP: pozycja naszego AP
        AP_angle: kąt nadawania naszego AP (w stopniach)
        STA: pozycja stacji
        interfering_APs: lista pozycji innych AP
        interfering_angles: lista kątów nadawania innych AP
        theta_bins, w_fft_dB: charakterystyka wiązki
        noise: moc szumu (mW)
        """
        print("kąt między AP a STA: ", AP_angle)
        signal = self.SSB(AP, AP_angle, STA, theta_bins=theta_bins, w_fft_dB=w_fft_dB)
        interference = 0
        for ap_int, angle_int in zip(interfering_APs, interfering_angles):
            interference += pow(10, self.SSB(ap_int, angle_int, STA, theta_bins=theta_bins, w_fft_dB=w_fft_dB)/10)
            print("poziom","kat interf: ",angle_int)
        noise_mW = noise
        SINR_dB = signal - 10 * np.log10(interference + noise_mW)
        return SINR_dB



# test_scen=calculations(aps)
# print("kontrolnie odleglosc ", np.linalg.norm(sta3-ap1), "i dzielenie ", np.linalg.norm(sta3-ap1)/Bp, "calosc ", 35*log10(np.linalg.norm(sta3-ap1)/Bp))
# print(35 * log10(10.0))
# print("Dla sta1 i ap1 mamy: ",test_scen.SINR(sta3,ap1),"sa one w odleglosci: ",np.linalg.norm(sta3-ap1))
## zakomentowane, bo badamy szczegolny przypadek
# sinr_values={}
# for i in range(15,100,1):
#     sta3=np.array([i,20])
#     temp=SINR(sta3,ap1)
#     sinr_values[i]=temp

# for x, sinr in sinr_values.items(): #zwykle wypisywanie X i SINR
#     print(f"x={x}, SINR={sinr:.2f} dB")

mcs_table_ac_ax = [
    (4, 0, "BPSK", "1/2", 8.6),
    (7, 1, "QPSK", "1/2", 17.2),
    (10, 2, "QPSK", "3/4", 25.8),
    (13, 3, "16-QAM", "1/2", 34.4),
    (16, 4, "16-QAM", "3/4", 51.6),
    (19, 5, "64-QAM", "2/3", 68.8),
    (22, 6, "64-QAM", "3/4", 77.4),
    (25, 7, "64-QAM", "5/6", 86),
    (28, 8, "256-QAM", "3/4", 103.2),
    (31, 9, "256-QAM", "5/6", 114.7),
    (34, 10, "1024-QAM", "3/4", 129),
    (37, 11, "1024-QAM", "5/6", 143.4),
]

def sinr_to_mcs(sinr):
    for sinr_threshold, mcs_index, mod, coding, rate in mcs_table_ac_ax:
        if sinr < sinr_threshold:
            return mcs_index - 1, rate
    return mcs_table_ac_ax[-1][1], mcs_table_ac_ax[-1][4]

# results_throughput = [sinr_to_mcs(sinr) for sinr in sinr_values.values()]

# for sinr, (mcs, rate) in zip(sinr_values.values(), results_throughput):
#     print(f"SINR: {sinr:.2f} dB -> MCS: {mcs}, Rate: {rate} Mbps")
def first_scenario_2AP_STA_moving(AP1,AP2,distance):
    sinr_with_interference = {}
    sinr_without_interference = {}
    throughput_with_interference = {}
    throughput_without_interference = {}

    APS=np.array([AP1,AP2])
    firstscenario=calculations(APS)

    for i in range(AP1[0]+10,distance,1):
        sta = np.array([i, AP2[1]])
        angle=angle_between(sta,AP1)[1]
        d=np.linalg.norm(sta-AP1)
        # SINR z interferencją
        theta_bins_rotated, w_fft_dB_rotated = rotate_beam_pattern(theta_bins, w_fft_dB, angle)
        antenna_gain=calculate_power_at_angle(theta_bins_rotated,w_fft_dB_rotated,angle)
        print("kat miedzy sta, a AP1: ",angle, "moc kierowana to: ", antenna_gain)
        sinr_with_interference[i] = firstscenario.SINR(sta, AP1)
        rate_with_interference = sinr_to_mcs(sinr_with_interference[i])[1]
        throughput_with_interference[i] = rate_with_interference
        
        ##SINR bez interferencji (zakładając interferencję jako 0)
        sinr_without_interference[i] = Tx_PWR +antenna_gain - (firstscenario.path_loss(d,f)+10*np.log10(noise))
        rate_without_interference = sinr_to_mcs(sinr_without_interference[i])[1]
        throughput_without_interference[i] = rate_without_interference

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sinr_with_interference.keys(), sinr_with_interference.values(), label="SINR z interferencją", color='b', marker='o', linestyle='-')
    ax1.plot(sinr_without_interference.keys(), sinr_without_interference.values(), label="SINR bez interferencji", color='g', marker='x', linestyle='--')

    ax1.set_xlabel("Pozycja STA (metry)")
    ax1.set_ylabel("SINR (dB)")
    ax1.set_title("SINR z i bez interferencji w zależności od pozycji STA")
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Wykres przepustowości (throughtput)
    ax2 = ax1.twinx()  # Druga oś Y
    ax2.plot(throughput_with_interference.keys(), throughput_with_interference.values(), label="Przepustowość z interferencją", color='r', linestyle='-.')
    ax2.plot(throughput_without_interference.keys(), throughput_without_interference.values(), label="Przepustowość bez interferencji", color='orange', linestyle=':')

    ax2.set_ylabel("Przepustowość (Mbps)")
    ax2.legend(loc='upper right')
    plt.show()

# first_scenario_2AP_STA_moving(np.array([20,10]),np.array([20,20]),120) # uruchomienie 1 scen i plot wykresów 
def second_scenario_AP_moving(distance):
    sinr_with_interference = {}
    sinr_without_interference = {}
    throughput_with_interference = {}
    throughput_without_interference = {}
    AP2 = np.array([35, 20])
    aps=np.array([ap1,AP2])
    sta=np.array([15,20])
    for i in range(sta[0]+20,sta[0]+distance,1):
        AP2[0]=i
        aps = np.array([ap1, AP2])
        scecond_scen = calculations(aps)
        print("wspolrzedne AP2 dla ",i,"metra to: ",AP2)
        angle=angle_between(sta,ap1)[1]
        d=np.linalg.norm(sta-ap1)
        # SINR z interferencją
        theta_bins_rotated, w_fft_dB_rotated = rotate_beam_pattern(theta_bins, w_fft_dB, angle)
        antenna_gain=calculate_power_at_angle(theta_bins_rotated,w_fft_dB_rotated,angle)
        print("kat miedzy sta, a AP1: ",angle, "moc kierowana to: ", antenna_gain)
        sinr_with_interference[i] = scecond_scen.SINR(sta, ap1,antenna_gain)
        rate_with_interference = sinr_to_mcs(sinr_with_interference[i])[1]
        throughput_with_interference[i] = rate_with_interference
        
        ##SINR bez interferencji (zakładając interferencję jako 0)
        sinr_without_interference[i] = Tx_PWR +antenna_gain - (scecond_scen.path_loss(d,f)+10*np.log10(noise))
        rate_without_interference = sinr_to_mcs(sinr_without_interference[i])[1]
        throughput_without_interference[i] = rate_without_interference

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sinr_with_interference.keys(), sinr_with_interference.values(), label="SINR z interferencją", color='b', marker='o', linestyle='-')
    ax1.plot(sinr_without_interference.keys(), sinr_without_interference.values(), label="SINR bez interferencji", color='g', marker='x', linestyle='--')

    ax1.set_xlabel("Pozycja INT (metry)")
    ax1.set_ylabel("SINR (dB)")
    ax1.set_title("SINR z i bez interferencji w zależności od pozycji AP (INT)")
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Wykres przepustowości (throughtput)
    ax2 = ax1.twinx()  # Druga oś Y
    ax2.plot(throughput_with_interference.keys(), throughput_with_interference.values(), label="Przepustowość z interferencją", color='r', linestyle='-.')
    ax2.plot(throughput_without_interference.keys(), throughput_without_interference.values(), label="Przepustowość bez interferencji", color='orange', linestyle=':')

    ax2.set_ylabel("Przepustowość (Mbps)")
    ax2.legend(loc='upper right')
    plt.show()

# second_scenario_AP_moving(200)

def third_scenario_AP_STA_around():
    from math import sin, cos, radians
    radius=20
    AP=ap1
    ap4=np.array([300,20])
    sta4=np.array([5,30])
    sta_positions = []
    for angle in range(0, 360, 1):  # co 30 stopni
        # Konwersja na radiany
        rad = radians(angle)
        third_scen=calculations(aps=np.array([AP,ap4]))
        # Obliczenie współrzędnych x,y
        x = AP[0] + radius * sin(rad)
        y = AP[1] + radius * cos(rad)
        sta_positions.append(np.array([x, y]))
    results={}
    for i in range(0,360,6):
        angle=i
        theta_bins_rotated, w_fft_dB_rotated = rotate_beam_pattern(theta_bins, w_fft_dB, 0)
        antenna_gain=calculate_power_at_angle(theta_bins_rotated,w_fft_dB_rotated,angle)
        print("kat miedzy sta, a AP1: ",angle, "moc kierowana to: ", antenna_gain)
        # Obliczenia SINR dla każdej pozycji
        print("kat: ",angle,"gain: ",antenna_gain)
        sinr = third_scen.SINR(sta4, AP,antenna_gain)
        results[angle] = sinr

    angles = list(results.keys())
    sinr_values = list(results.values())
    
    # Przesuwamy wartości o jedną pozycję w prawo
    sinr_values = sinr_values[1:] + sinr_values[:1]
    throughput_values = [sinr_to_mcs(sinr)[1] for sinr in sinr_values]
    angles_rad = np.array(angles) * np.pi / 180

    # Przeskalowanie wartości SINR do zakresu wzoru promieniowania (-30, 15)
    min_beam = -30
    max_beam = 15
    beam_range = max_beam - min_beam
    # min_sinr = min(sinr_values)
    # max_sinr = max(sinr_values)
    # sinr_range = max_sinr - min_sinr
    min_throughput = min(throughput_values)
    max_throughput = max(throughput_values)
    throughput_range = max_throughput - min_throughput
    scaled_throughput = [(x - min_throughput) * beam_range / throughput_range + min_beam for x in throughput_values]

    # Funkcja do skalowania wartości
    # scaled_sinr = [(x - min_sinr) * beam_range / sinr_range + min_beam for x in sinr_values]

    # Tworzenie wykresu
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Wykres wiązki
    ax.plot(theta_bins, w_fft_dB, 'r-', label="Wzór promieniowania")
    ax.plot([0/ 360 * np.pi], [np.max(w_fft_dB)], 'og', label="Główna wiązka")

    # Wykres SINR
    ax.plot(angles_rad, scaled_throughput, 'b-o', label="SINR")

    # Ustawienia wykresu
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_ylim([min_beam, max_beam])

    # Dodanie dwóch legend - jednej dla wartości oryginalnych SINR
    ax.legend(loc='upper right')

    # Dodanie adnotacji z rzeczywistymi wartościami SINR
    # for i, (angle, sinr) in enumerate(zip(angles_rad, sinr_values)):
    #     if i % 2 == 0:  # dodajemy etykiety co drugi punkt dla czytelności
    #         ax.annotate(f'{sinr:.1f}dB', 
    #                 (angle, scaled_sinr[i]),
    #                 xytext=(10, 10),
    #                 textcoords='offset points')
    for i, (angle, thr) in enumerate(zip(angles_rad, throughput_values)):
        if i % 2 == 0:  # dodajemy etykiety co drugi punkt dla czytelności
            ax.annotate(f'{thr:.1f} Mbps', 
                    (angle, scaled_throughput[i]),
                    xytext=(10, 10),
                    textcoords='offset points')

    plt.title("Charakterystyka promieniowania anteny i SINR")
    plt.show()

# third_scenario_AP_STA_around()

def fourth_scenario_2AP_2STA(d):
    ap3=np.array([37,30])
    ap4=np.array([43,30])
    sta4=np.array([40,30])
    sta5=np.array([40,30])
    aps=np.array([ap3,ap4])
    stas=np.array([sta4,sta5])
    scen4=calculations(aps)
    sinr_omni = {}
    sinr_beam = {}
    sinr_beam2 = {}
    throughput_omni = {}
    throughput_beam = {}
    throughput_beam2 = {}
    theta_bins1, w_fft_dB1 = calculate_beam_pattern(12, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)
    theta_bins2, w_fft_dB2 = calculate_beam_pattern(12, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)
    plt.figure(figsize=(8, 8))
    plt.scatter(aps[:, 0], aps[:, 1], c='red', label='AP', marker='x', s=100)
    plt.scatter(stas[:, 0], stas[:, 1], c='blue', label='STA', marker='o', s=100)
    plt.xlim(0, room_size-45)
    plt.ylim(0, room_size-45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Rozmieszczenie AP i STA')
    plt.xlabel('X (metry)')
    plt.ylabel('Y (metry)')
    plt.legend()
    plt.show()
    for i in range(1,d):
        sta5[1] = sta5[1]+1
        sta4[1]=sta4[1]-1
        # print(type(sta5[1]))
        # print(type(sta4[1]))
        print("---------Kolejna runda dla odleglosci: ",i,"----------")
        angle1=angle_between(ap3,sta4)
        angle1int=angle_between(sta4,ap4)
        angle2=angle_between(ap4,sta5)
        angle2int=angle_between(sta5,ap3)
        d=np.linalg.norm(sta4-ap3)
        # SINR 
        theta_bins1_rotated, w_fft_dB1_rotated = rotate_beam_pattern(theta_bins1, w_fft_dB1, angle1)
        theta_bins2_rotated, w_fft_dB2_rotated = rotate_beam_pattern(theta_bins2, w_fft_dB2, angle2)
    
        # Obliczanie wzmocnienia dla docelowych stacji
        antenna_gain1 = calculate_power_at_angle(theta_bins1_rotated, w_fft_dB1_rotated, angle1)
        antenna_gain2 = calculate_power_at_angle(theta_bins2_rotated, w_fft_dB2_rotated, angle2)
        all_active_transmissions = [
            {
                'ap': ap3,
                'sta': sta4,
                'beam_pattern': (theta_bins1_rotated, w_fft_dB1_rotated)
            },
            {
                'ap': ap4,
                'sta': sta5,
                'beam_pattern': (theta_bins2_rotated, w_fft_dB2_rotated)
            }
        ]
        print("kat miedzy sta, a AP: ",angle2, "moc kierowana to: ", antenna_gain2, "pozycja sta to: ",sta5, "odleglosc miedzy sta ",i)
        sinr_omni[i] = Tx_PWR - (scen4.path_loss(d,f)+10*np.log10(noise))
        rate_omni = sinr_to_mcs(sinr_omni[i])[1]
        throughput_omni[i] = rate_omni
        
        ##SINR z antenna gain
        sinr_beam[i] = scen4.SINR(sta4,ap3, antenna_gain2,all_active_transmissions)
        rate_beam = sinr_to_mcs(sinr_beam[i])[1]
        throughput_beam[i] = rate_beam

        # sinr_beam2[i] = scen4.SINR(sta4,ap3,antenna_gain1,all_active_transmissions)
        # rate_beam=sinr_to_mcs(sinr_beam2[i])[1]
        # throughput_beam2[i] = rate_beam
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sinr_omni.keys(), sinr_omni.values(), label="SINR dookolny", color='b', marker='o', linestyle='-')
    ax1.plot(sinr_beam.keys(), sinr_beam.values(), label="SINR wiazka", color='g', marker='x', linestyle='--')

    ax1.set_xlabel("Odległość między STAs/2")
    ax1.set_ylabel("SINR (dB)")
    ax1.set_title("SINR omni i beam w zależności od odległości między STA")
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Wykres przepustowości (throughtput)
    ax2 = ax1.twinx()  # Druga oś Y
    ax2.plot(throughput_omni.keys(), throughput_omni.values(), label="Przepustowość omni", color='r', linestyle='-.')
    ax2.plot(throughput_beam.keys(), throughput_beam.values(), label="Przepustowość beam", color='orange', linestyle=':')

    ax2.set_ylabel("Przepustowość (Mbps)")
    ax2.legend(loc='upper right')
    plt.show()

# fourth_scenario_2AP_2STA(12)

def scenario_2AP_2STA_beamforming(d):
    """
    Scenariusz dwóch AP i dwóch STA z wykorzystaniem get_beam_angles_and_delta oraz SINR_beamforming/SSB.
    """
    ap3 = np.array([37, 30])
    ap4 = np.array([43, 30])
    sta4 = np.array([40, 30])
    sta5 = np.array([40, 30])
    aps = np.array([ap3, ap4])
    stas = np.array([sta4, sta5])
    scen = calculations(aps)
    sinr_beamforming = {}
    throughput_beamforming = {}
    sinr_omni = {}
    throughput_omni = {}

    theta_bins1, w_fft_dB1 = calculate_beam_pattern(8, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)
    theta_bins2, w_fft_dB2 = calculate_beam_pattern(8, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)

    # for i in range(1, d):
    #     sta4[1] = sta4[1] - 1
    #     sta5[1] = sta5[1] + 1
    #     print("pozycje stacji: ", sta4, "i ", sta5)
    #     # Wyznacz kąty wiązek i deltę
    #     alpha, beta, gamma, delta = get_beam_angles_and_delta(ap3, sta4, ap4, sta5)

    #     # Oblicz SINR_beamforming dla STA4 (od ap3, zakładając ap4 jako interferer)
    #     interfering_APs = [ap4]
    #     interfering_angles = [gamma]  # gamma to kąt wiązki ap4->sta5

    #     # Używamy theta_bins1/w_fft_dB1 dla ap3, theta_bins2/w_fft_dB2 dla ap4
    #     sinr = scen.SINR_beamforming(
    #         ap3, alpha, sta4,
    #         interfering_APs, interfering_angles,
    #         theta_bins1, w_fft_dB1
    #     )
    #     sinr_beamforming[i] = sinr
    #     throughput_beamforming[i] = sinr_to_mcs(sinr)[1]
    #     d= np.linalg.norm(sta4 - ap3)
    #     sinr_omni[i] = Tx_PWR - (scen.path_loss(d,f)+10*np.log10(noise))
    #     rate_omni = sinr_to_mcs(sinr_omni[i])[1]
    #     throughput_omni[i] = rate_omni

    #     print(f"Iteracja {i}: alpha={alpha:.1f}, gamma={gamma:.1f}, delta={delta:.1f} ,SINR={sinr:.2f} dB, throughput={throughput_beamforming[i]:.1f} Mbps")
    # ###### -------- Wykres porównawczy SINR i throughput dla beamforming i omni --------
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.plot(sinr_omni.keys(), sinr_omni.values(), label="SINR dookolny", color='b', marker='o', linestyle='-')
    # ax1.plot(sinr_beamforming.keys(), sinr_beamforming.values(), label="SINR beamforming", color='g', marker='x', linestyle='--')
    # ax1.set_xlabel("Odległość między STAs/2")
    # ax1.set_ylabel("SINR (dB)")
    # ax1.set_title("SINR beamforming w zależności od odległości między STA")
    # ax1.grid(True)
    # ax1.legend(loc='upper left')
    # ax1.set_ylim(bottom=0)


    # ax2 = ax1.twinx()
    # ax2.plot(throughput_omni.keys(), throughput_omni.values(), label="Przepustowość omni", color='r', linestyle='-.')
    # ax2.plot(throughput_beamforming.keys(), throughput_beamforming.values(), label="Przepustowość beamforming", color='orange', linestyle=':')
    # ax2.set_ylabel("Przepustowość (Mbps)")
    # ax2.legend(loc='upper right')
    # ax2.set_ylim(bottom=0)
    # plt.show()
    
# --- WYKRESY SŁUPKOWE DLA STAŁYCH POZYCJI sta4=[40,25], sta5=[40,35] ---
    sta4_fixed = np.array([37, 20])
    sta5_fixed = np.array([43, 40])
    aps = np.array([ap3, ap4])
    scen2 = calculations(aps)

    alpha1, beta1, gamma1, delta1 = get_beam_angles_and_delta(ap3, sta4_fixed, ap4, sta5_fixed)
    print(f"Dla stałych miejsc Alpha1: {alpha1}, Beta1: {beta1}, Gamma1: {gamma1}, Delta1: {delta1}")
    # 1. Tylko jeden AP nadaje (omni)
    d_ap3_sta4 = np.linalg.norm(sta4_fixed - ap3)
    d_ap4_sta5 = np.linalg.norm(sta5_fixed - ap4)
    # sinr_omni_ap3 = scen2.SSB(ap3, 0, sta4_fixed, theta_bins=theta_bins1, w_fft_dB=w_fft_dB1)
    # sinr_omni_ap4 = scen2.SSB(ap4, 0, sta5_fixed, theta_bins=theta_bins2, w_fft_dB=w_fft_dB2)
    sinr_omni_ap3 = Tx_PWR - (scen2.path_loss(d_ap3_sta4, f) + 10 * np.log10(noise))
    sinr_omni_ap4 = Tx_PWR - (scen2.path_loss(d_ap4_sta5, f) + 10 * np.log10(noise))
    thr_omni_ap3 = sinr_to_mcs(sinr_omni_ap3)[1]
    thr_omni_ap4 = sinr_to_mcs(sinr_omni_ap4)[1]

    all_active_transmissions = [
            {
                'ap': ap3,
                'sta': sta4,
                'beam_pattern': (theta_bins1, w_fft_dB1)
            },
            {
                'ap': ap4,
                'sta': sta5,
                'beam_pattern': (theta_bins2, w_fft_dB2)
            }
        ]

    # 2. Oba AP nadają (omni)
    interf_ap3 = pow(10, (Tx_PWR - scen2.path_loss(np.linalg.norm(sta4_fixed - ap4), f)) / 10)
    interf_ap4 = pow(10, (Tx_PWR - scen2.path_loss(np.linalg.norm(sta5_fixed - ap3), f)) / 10)
    sinr_omni_both_ap3 = Tx_PWR - (scen2.path_loss(d_ap3_sta4, f) + 10 * np.log10(interf_ap3 + noise))
    sinr_omni_both_ap4 = Tx_PWR - (scen2.path_loss(d_ap4_sta5, f) + 10 * np.log10(interf_ap4 + noise))
    thr_omni_both_ap3 = sinr_to_mcs(sinr_omni_both_ap3)[1]
    thr_omni_both_ap4 = sinr_to_mcs(sinr_omni_both_ap4)[1]

    # 3. Tylko jeden AP nadaje (beamforming)
    angle_ap3_sta4 = angle_between(ap3, sta4_fixed)
    angle_ap4_sta5 = angle_between(ap4, sta5_fixed)
    print("kąty dla tylko 1 AP nadającego: ap3 ", angle_ap3_sta4, "ap4: ", angle_ap4_sta5)
    theta_bins1_rot, w_fft_dB1_rot = rotate_beam_pattern(theta_bins1, w_fft_dB1, angle_ap3_sta4)
    theta_bins2_rot, w_fft_dB2_rot = rotate_beam_pattern(theta_bins2, w_fft_dB2, angle_ap4_sta5)
    # plot_beam_pattern_cartesian(theta_bins1_rot, w_fft_dB1_rot)
    gain_ap3 = calculate_power_at_angle(theta_bins1_rot, w_fft_dB1_rot, angle_ap3_sta4)
    gain_ap4 = calculate_power_at_angle(theta_bins2_rot, w_fft_dB2_rot, angle_ap4_sta5)
    print("1 ap beam gain: ap3 ", gain_ap3, "ap4: ", gain_ap4)
    sinr_beam_ap3 = Tx_PWR + gain_ap3 - (scen2.path_loss(d_ap3_sta4, f) + 10 * np.log10(noise))
    sinr_beam_ap4 = Tx_PWR + gain_ap4 - (scen2.path_loss(d_ap4_sta5, f) + 10 * np.log10(noise))
    print("sinr_beam_ap3: ", sinr_beam_ap3, "sinr_beam_ap4: ", sinr_beam_ap4)
    thr_beam_ap3 = sinr_to_mcs(sinr_beam_ap3)[1]
    thr_beam_ap4 = sinr_to_mcs(sinr_beam_ap4)[1]

    # 4. Oba AP nadają (beamforming)
    # Interferencja od drugiego AP (beamforming)
    angle_ap4_sta4 = angle_between(ap4, sta4_fixed)
    gain_ap4_to_sta4 = calculate_power_at_angle(theta_bins2_rot, w_fft_dB2_rot, angle_ap4_sta4)
    interf_beam_ap3 = pow(10, (Tx_PWR + gain_ap4_to_sta4 - scen.path_loss(np.linalg.norm(sta4_fixed - ap4), f)) / 10)
    sinr_beam_both_ap3 = scen.SSB(ap3, angle_ap3_sta4, sta4_fixed, theta_bins=theta_bins1, w_fft_dB=w_fft_dB1) - 10 * np.log10(interf_beam_ap3 + noise)
    thr_beam_both_ap3 = sinr_to_mcs(sinr_beam_both_ap3)[1]

    angle_ap3_sta5 = angle_between(ap3, sta5_fixed)
    gain_ap3_to_sta4 = calculate_power_at_angle(theta_bins1_rot, w_fft_dB1_rot, angle_ap3_sta4)
    interf_beam_ap4 = pow(10, (Tx_PWR + gain_ap4_to_sta4 - scen.path_loss(np.linalg.norm(sta4_fixed - ap4), f)) / 10)
    sinr_beam_both_ap4 = scen.SSB(ap4, angle_ap4_sta5, sta5_fixed, theta_bins=theta_bins2, w_fft_dB=w_fft_dB2) - 10 * np.log10(interf_beam_ap4 + noise)
    print("wszystkie sily wchodzące dla ap4: ", Tx_PWR, gain_ap3_to_sta4, "scen.path_loss: ", scen2.path_loss(np.linalg.norm(sta4_fixed - ap3), f), "interf_beam_ap4: ", np.log10(interf_beam_ap4), "noise: ", noise)
    thr_beam_both_ap4 = sinr_to_mcs(sinr_beam_both_ap4)[1]

    labels = [
        "1 AP omni", "2 AP omni",
        "1 AP beam", "2 AP beam"
    ]
    sinr_sta4 = [
        sinr_omni_ap3, sinr_omni_both_ap3,
        sinr_beam_ap3, sinr_beam_both_ap3
    ]
    sinr_sta5 = [
        sinr_omni_ap4, sinr_omni_both_ap4,
        sinr_beam_ap4, sinr_beam_both_ap4
    ]
    thr_sta4 = [
        thr_omni_ap3, thr_omni_both_ap3,
        thr_beam_ap3, thr_beam_both_ap3
    ]
    thr_sta5 = [
        thr_omni_ap4, thr_omni_both_ap4,
        thr_beam_ap4, thr_beam_both_ap4
    ]

    x = np.arange(len(labels))
    width = 0.35

    # Wykres słupkowy SINR
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, sinr_sta4, width, label='STA4 [37,20]')
    plt.bar(x + width/2, sinr_sta5, width, label='STA5 [43,40]')
    plt.xticks(x, labels)
    plt.ylabel("SINR (dB)")
    plt.title("Porównanie SINR dla różnych wariantów (stałe pozycje STA)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Wykres słupkowy przepustowości
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, thr_sta4, width, label='STA4 [40,25]')
    plt.bar(x + width/2, thr_sta5, width, label='STA5 [40,35]')
    plt.xticks(x, labels)
    plt.ylabel("Przepustowość (Mbps)")
    plt.title("Porównanie przepustowości dla różnych wariantów (stałe pozycje STA)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter([ap3[0], ap4[0]], [ap3[1], ap4[1]], c='red', label='AP', marker='x', s=100)
    plt.scatter([sta4_fixed[0], sta5_fixed[0]], [sta4_fixed[1], sta5_fixed[1]], c='blue', label='STA', marker='o', s=100)
    plt.xlim(0, room_size-45)
    plt.ylim(0, room_size-45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Rozmieszczenie AP i STA')
    plt.xlabel('X (metry)')
    plt.ylabel('Y (metry)')
    plt.legend()
    plt.show()


# Przykład uruchomienia:
scenario_2AP_2STA_beamforming(12)
# Wizualizacja rozmieszczenia
# plt.figure(figsize=(8, 8))
# plt.scatter(aps[:, 0], aps[:, 1], c='red', label='AP', marker='x', s=100)
# plt.scatter(stas[:, 0], stas[:, 1], c='blue', label='STA', marker='o', s=100)
# plt.xlim(0, room_size)
# plt.ylim(0, room_size)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.title('Rozmieszczenie AP i STA')
# plt.xlabel('X (metry)')
# plt.ylabel('Y (metry)')
# plt.legend()
# plt.show()

## wartosci SINR
# fig, ax1=plt.subplots(figsize=(10, 6))
# plt.plot(sinr_values.keys(), sinr_values.values(), marker='o', linestyle='-', color='b', label='SINR')  # Wykres liniowy z markerami
# plt.title("Wykres SINR w zależności od pozycji STA na osi X")
# plt.xlabel("Pozycja STA (metry)")
# plt.ylabel("SINR (dB)")
# plt.grid(True)  
# plt.legend()
# plt.show()

def fifth_scenario_4STA_2AP_line(d1=10, d2_range=None):
    """
    Scenariusz: 4 stacje w linii (STA1--AP1--STA2)-(STA3--AP2--STA4).
    d1 - odległość AP-STA (stała)
    d2_range - lista odległości AP-AP (domyślnie od 2*d1 do 8*d1 co d1)
    Dla każdej strategii (omni/beam do: zewnętrznych, wewnętrznych, lewych, prawych)
    rysuje wykres sumarycznej przepustowości (sumaryczny throughput) vs d2.
    """
    if d2_range is None:
        d2_range = [2*d1, 3*d1, 4*d1, 5*d1, 6*d1, 7*d1, 8*d1]

    # Wyniki dla każdej strategii
    results = {
        "omni_outer": [],
        "beam_outer": [],
        "omni_inner": [],
        # "beam_inner": [],
        "omni_left": [],
        "beam_left": [],
        # "omni_right": [],
        # "beam_right": []
    }

    # Ustawienia anten
    theta_bins, w_fft_dB = calculate_beam_pattern(8, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)

    for d2 in d2_range:
        # Pozycje AP i STA
        # Oś X: STA1--AP1--STA2----STA3--AP2--STA4
        AP1 = np.array([0, 0])
        AP2 = np.array([d2, 0])
        STA1 = np.array([-d1, 0])
        STA2 = np.array([d1, 0])
        STA3 = np.array([d2 - d1, 0])
        STA4 = np.array([d2 + d1, 0])
        aps = np.array([AP1, AP2])
        stas = np.array([STA1, STA2, STA3, STA4])
        scen = calculations(aps)

        # --- Strategie ---
        # 1. OMNI do zewnętrznych (STA1, STA4)
        # 2. BEAM do zewnętrznych (STA1, STA4)
        # 3. OMNI do wewnętrznych (STA2, STA3)
        # 4. BEAM do wewnętrznych (STA2, STA3)
        # 5. OMNI do lewych (STA1, STA2)
        # 6. BEAM do lewych (STA1, STA2)
        # 7. OMNI do prawych (STA3, STA4)
        # 8. BEAM do prawych (STA3, STA4)

        # Helper: SINR i throughput dla pary (AP, STA), interferencja od drugiego AP
        def get_sinr_thr(AP, STA, AP_int, STA_int, use_beam, theta_bins, w_fft_dB):
            if use_beam:
                # Beamforming: ustaw wiązkę na STA
                beam_angle = angle_between(AP, STA)
                int_beam_angle = angle_between(AP_int, STA_int)
                sinr = scen.SINR_beamforming(
                    AP, beam_angle, STA,
                    [AP_int], [int_beam_angle],
                    theta_bins, w_fft_dB
                )
            else:
                # Omni: gain = 0
                d = np.linalg.norm(STA - AP)
                interf = pow(10, (Tx_PWR - scen.path_loss(np.linalg.norm(STA - AP_int), f)) / 10)
                sinr = Tx_PWR - (scen.path_loss(d, f) + 10 * np.log10(interf + noise))
            thr = sinr_to_mcs(sinr)[1]
            return sinr, thr

        # Zewnętrzne: STA1 (AP1), STA4 (AP2)
        sinr1, thr1 = get_sinr_thr(AP1, STA1, AP2, STA4, False, theta_bins, w_fft_dB)
        sinr4, thr4 = get_sinr_thr(AP2, STA4, AP1, STA1, False, theta_bins, w_fft_dB)
        results["omni_outer"].append(thr1 + thr4)
        sinr1b, thr1b = get_sinr_thr(AP1, STA1, AP2, STA4, True, theta_bins, w_fft_dB)
        sinr4b, thr4b = get_sinr_thr(AP2, STA4, AP1, STA1, True, theta_bins, w_fft_dB)
        results["beam_outer"].append(thr1b + thr4b)

        # Wewnętrzne: STA2 (AP1), STA3 (AP2)
        sinr2, thr2 = get_sinr_thr(AP1, STA2, AP2, STA3, False, theta_bins, w_fft_dB)
        sinr3, thr3 = get_sinr_thr(AP2, STA3, AP1, STA2, False, theta_bins, w_fft_dB)
        results["omni_inner"].append(thr2 + thr3)
        # sinr2b, thr2b = get_sinr_thr(AP1, STA2, AP2, STA3, True, theta_bins, w_fft_dB)
        # sinr3b, thr3b = get_sinr_thr(AP2, STA3, AP1, STA2, True, theta_bins, w_fft_dB)
        # results["beam_inner"].append(thr2b + thr3b)

        # Lewe: STA1 (AP1), STA2 (AP1)
        sinr1l, thr1l = get_sinr_thr(AP1, STA1, AP2, STA4, False, theta_bins, w_fft_dB)
        sinr2l, thr2l = get_sinr_thr(AP1, STA2, AP2, STA3, False, theta_bins, w_fft_dB)
        results["omni_left"].append(thr1l + thr2l)
        sinr1lb, thr1lb = get_sinr_thr(AP1, STA1, AP2, STA4, True, theta_bins, w_fft_dB)
        sinr2lb, thr2lb = get_sinr_thr(AP1, STA2, AP2, STA3, True, theta_bins, w_fft_dB)
        results["beam_left"].append(thr1lb + thr2lb)

        # Prawe: STA3 (AP2), STA4 (AP2)
        # sinr3r, thr3r = get_sinr_thr(AP2, STA3, AP1, STA2, False, theta_bins, w_fft_dB)
        # sinr4r, thr4r = get_sinr_thr(AP2, STA4, AP1, STA1, False, theta_bins, w_fft_dB)
        # results["omni_right"].append(thr3r + thr4r)
        # sinr3rb, thr3rb = get_sinr_thr(AP2, STA3, AP1, STA2, True, theta_bins, w_fft_dB)
        # sinr4rb, thr4rb = get_sinr_thr(AP2, STA4, AP1, STA1, True, theta_bins, w_fft_dB)
        # results["beam_right"].append(thr3rb + thr4rb)

    # Wykres sumarycznego throughput vs d2
    plt.figure(figsize=(12, 7))
    plt.plot(d2_range, results["omni_outer"], 'o-', label="Omni do zewnętrznych")
    plt.plot(d2_range, results["beam_outer"], 'o-', label="Beam do zewnętrznych")
    plt.plot(d2_range, results["omni_inner"], 's--', label="Omni do wewnętrznych")
    # plt.plot(d2_range, results["beam_inner"], 's--', label="Beam do wewnętrznych")
    plt.plot(d2_range, results["omni_left"], 'd-.', label="Omni do lewych")
    plt.plot(d2_range, results["beam_left"], 'd-.', label="Beam do lewych")
    # plt.plot(d2_range, results["omni_right"], 'x:', label="Omni do prawych")
    # plt.plot(d2_range, results["beam_right"], 'x:', label="Beam do prawych")
    plt.xlabel("Odległość AP-AP (metry)")
    plt.ylabel("Sumaryczny throughput (Mbps)")
    plt.title(f"Sumaryczny throughput vs odległość AP-AP (d1={d1}m)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Przykład użycia:
# fifth_scenario_4STA_2AP_line(d1=10, d2_range=None)