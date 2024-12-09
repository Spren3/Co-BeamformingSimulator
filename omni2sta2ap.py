import numpy as np
import matplotlib.pyplot as plt
from math import log10, pow 
room = np.zeros((100,100))
ap1=np.array([5,20])
ap2=np.array([5,50])
sta1=np.array([40,20])
sta2=np.array([15,40])
aps = np.array([ap1, ap2])
stas = np.array([sta1, sta2]) #zmienne zeby zgrupowac stacje i AP do wyswietlania na wykresie
objects=np.array([ap1,ap2,sta1,sta2])

def pickOtherAP(STA,AP):
    min=np.linalg.norm(STA-AP)
    closestAP=None
    for i in range(len(aps)):
        if np.array_equal(AP, aps[i]):
            continue
        else:
            d=np.linalg.norm(STA-aps[i])
            if d<min:
                min=d
                closestAP=aps[i]
    return d,closestAP
dist,closest=pickOtherAP(sta1,ap1)
print("najblizsza stacja do sta1 to: ",closest, "w odleglosci :",dist)
print(np.linalg.norm(np.array([0,30])-ap2))
##parametry (mozna powiedziec ze globalne)
f= 2.4
Tx_PWR = 20 # w dBm
noise=-93.97 #dBm
Bp=10 # breaking point w metrach
def path_loss(d,f):
    P=35 * log10(d/Bp)
    path_loss=40.05 + 20*log10((min(d,Bp)*f)/2.4)
    if (d>Bp):
        path_loss+=P
    return path_loss
def interf(d):
    print("txpwr to :",Tx_PWR,"a path loss: ", path_loss(d,f))
    interference=Tx_PWR-path_loss(d,f)
    interference_mW=pow(10,(interference/10))
    return interference_mW
room_size = 100  # w metrach
def SINR(STA,AP):
    epsilon=np.random.normal(0,2)
    d=np.linalg.norm(STA-AP)
    # print("odleglosc to : ",d)
    d2, clAP=pickOtherAP(STA, AP)
    print("Kontrolnie wypisanie wszystkiego: ",clAP,d2,path_loss(d,f),interf(d2))
    # sinr=Tx_PWR-(path_loss(d,f)+10*log10(noise+interf(d2)))+epsilon
    sinr=Tx_PWR-(path_loss(d,f)+interf(d2)+noise)
    return sinr

print("Dla sta1 i ap1 mamy: ",SINR(sta1,ap1))
sinr_values={}
for i in range(15,100,1):
    sta3=np.array([i,20])
    temp=SINR(sta3,ap1)
    sinr_values[i]=temp

# for x, sinr in sinr_values.items():
#     print(f"x={x}, SINR={sinr:.2f} dB")

mcs_table_ac_ax = [
    (0, 0, "BPSK", "1/2", 6.5),
    (3, 1, "QPSK", "1/2", 13),
    (6, 2, "QPSK", "3/4", 19.5),
    (10, 3, "16-QAM", "1/2", 26),
    (15, 4, "16-QAM", "3/4", 39),
    (20, 5, "64-QAM", "2/3", 52),
    (25, 6, "64-QAM", "3/4", 58.5),
    (30, 7, "64-QAM", "5/6", 65),
    (36, 8, "256-QAM", "3/4", 78),
    (41, 9, "256-QAM", "5/6", 86.7),
    (46, 10, "1024-QAM", "3/4", 96),
    (51, 11, "1024-QAM", "5/6", 106.7),
]

def sinr_to_mcs(sinr):
    for sinr_threshold, mcs_index, mod, coding, rate in mcs_table_ac_ax:
        if sinr < sinr_threshold:
            return mcs_index - 1, rate
    return mcs_table_ac_ax[-1][1], mcs_table_ac_ax[-1][4]

results_throughput = [sinr_to_mcs(sinr) for sinr in sinr_values.values()]

for sinr, (mcs, rate) in zip(sinr_values.values(), results_throughput):
    print(f"SINR: {sinr:.2f} dB -> MCS: {mcs}, Rate: {rate} Mbps")

# Wizualizacja rozmieszczenia
plt.figure(figsize=(8, 8))
plt.scatter(aps[:, 0], aps[:, 1], c='red', label='AP', marker='x', s=100)
plt.scatter(stas[:, 0], stas[:, 1], c='blue', label='STA', marker='o', s=100)
plt.xlim(0, room_size)
plt.ylim(0, room_size)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Rozmieszczenie AP i STA')
plt.xlabel('X (metry)')
plt.ylabel('Y (metry)')
plt.legend()
plt.show(block=False)

## wartosci SINR
fig, ax1=plt.subplots(figsize=(10, 6))
plt.plot(sinr_values.keys(), sinr_values.values(), marker='o', linestyle='-', color='b', label='SINR')  # Wykres liniowy z markerami
plt.title("Wykres SINR w zależności od pozycji STA na osi X")
plt.xlabel("Pozycja STA (metry)")
plt.ylabel("SINR (dB)")
plt.grid(True)  
plt.legend()
plt.show()