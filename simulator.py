import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from beam_pattern import calculate_beam_pattern, calculate_power_at_angle, rotate_beam_pattern, plot_beam_pattern_cartesian
from omni2sta2ap_angle_problem import calculations, angle_between, angle_between_points_from_perspective, compute_mean_ci, plot_means_with_ci, plot_boxplots, plot_histograms, plot_cdf
import random
@dataclass
class NetworkNode:
    """Reprezentuje węzeł sieci (AP lub STA)"""
    id: int
    x: float
    y: float
    node_type: str  # 'AP' or 'STA'
    associated_ap: int = None  # dla STA - ID przypisanego AP

class TopologyGenerator:
    """Generator topologii sieci według modelu z dokumentu IEEE 802.11bn"""
    
    # def __init__(self):
    #     # Parametry z dokumentu
    #     self.noise_floor = -94  # dBm
    #     self.wall_attenuation = 7  # dB
    #     self.max_tx_power = 16  # dBm
        
    def generate_multiroom_topology(self,
                                  topo_seed: int, 
                                  grid_size: Tuple[int, int], 
                                  room_size: float,
                                  stations_per_room: int = 4) -> Dict:
        """
        Generuje topologię wielopokojową
        
        Args:
            grid_size: (rows, cols) - wymiary siatki pokoi
            room_size: rozmiar pokoju w metrach (ρ)
            stations_per_room: liczba stacji na pokój
            
        Returns:
            Dict zawierający węzły, łącza i parametry topologii
        """
        np.random.seed(topo_seed)
        rows, cols = grid_size
        nodes = []
        # walls = []
        node_id = 0
        
        # Generowanie pokoi z AP i stacjami
        for row in range(rows):
            for col in range(cols):
                # Granice pokoju
                room_x_min = col * room_size
                room_x_max = (col + 1) * room_size
                room_y_min = row * room_size
                room_y_max = (row + 1) * room_size
                
                # Losowe umieszczenie AP w pokoju
                ap_x = np.random.uniform(room_x_min, room_x_max)
                ap_y = np.random.uniform(room_y_min, room_y_max)
                
                ap_node = NetworkNode(node_id, ap_x, ap_y, 'AP')
                nodes.append(ap_node)
                ap_id = node_id
                node_id += 1
                
                # Losowe umieszczenie stacji w tym samym pokoju
                for _ in range(stations_per_room):
                    sta_x = np.random.uniform(room_x_min, room_x_max)
                    sta_y = np.random.uniform(room_y_min, room_y_max)
                    
                    sta_node = NetworkNode(node_id, sta_x, sta_y, 'STA', ap_id)
                    nodes.append(sta_node)
                    node_id += 1
                
                # Generowanie ścian
                # Ściany pionowe
                # if col < cols - 1:
                #     walls.append(((room_x_max, room_y_min), (room_x_max, room_y_max)))
                # # Ściany poziome
                # if row < rows - 1:
                #     walls.append(((room_x_min, room_y_max), (room_x_max, room_y_max)))
        
        # Tworzenie grafu dwudzielnego
        bipartite_graph = self._create_bipartite_graph(nodes)
        
        return {
            'nodes': nodes,
            # 'walls': walls,
            'bipartite_graph': bipartite_graph,
            'topology_type': 'multiroom',
            'parameters': {
                'grid_size': grid_size,
                'room_size': room_size,
                'total_area': (cols * room_size, rows * room_size)
            }
        }
    
    def generate_open_space_topology(self, 
                                   topo_seed: int,
                                   area_size: float = 75.0,
                                   num_aps: int = 4,
                                   stations_per_ap: Tuple[int, int] = (3, 4),
                                   station_std: float = 10.0) -> Dict:
        """
        Generuje topologię otwartej przestrzeni
        
        Args:
            area_size: rozmiar kwadratu w metrach
            num_aps: liczba punktów dostępowych
            stations_per_ap: (min, max) stacji na AP
            station_std: odchylenie standardowe rozmieszczenia stacji wokół AP
            
        Returns:
            Dict zawierający węzły, łącza i parametry topologii
        """
        np.random.seed(topo_seed)
        nodes = []
        node_id = 0
        
        # Losowe rozmieszczenie AP w przestrzeni
        ap_positions = np.random.uniform(0, area_size, (num_aps, 2))
        
        for i, (ap_x, ap_y) in enumerate(ap_positions):
            # Tworzenie AP
            ap_node = NetworkNode(node_id, ap_x, ap_y, 'AP')
            nodes.append(ap_node)
            ap_id = node_id
            node_id += 1
            
            # Liczba stacji dla tego AP
            num_stations = np.random.randint(stations_per_ap[0], stations_per_ap[1] + 1)
            
            # Rozmieszczenie stacji wokół AP (rozkład normalny)
            for _ in range(num_stations):
                # Pozycja stacji z rozkładem normalnym wokół AP
                sta_x = np.random.normal(ap_x, station_std)
                sta_y = np.random.normal(ap_y, station_std)
                
                # Ograniczenie do granic obszaru
                sta_x = np.clip(sta_x, 0, area_size)
                sta_y = np.clip(sta_y, 0, area_size)
                
                sta_node = NetworkNode(node_id, sta_x, sta_y, 'STA', ap_id)
                nodes.append(sta_node)
                node_id += 1
        
        # Reassign stations to nearest APs (jak w dokumencie)
        self._reassign_to_nearest_ap(nodes)
        
        # Tworzenie grafu dwudzielnego
        bipartite_graph = self._create_bipartite_graph(nodes)
        
        return {
            'nodes': nodes,
            'bipartite_graph': bipartite_graph,
            'topology_type': 'open_space',
            'parameters': {
                'area_size': area_size,
                'num_aps': num_aps,
                'station_std': station_std
            }
        }
    
    def _reassign_to_nearest_ap(self, nodes: List[NetworkNode]):
        """Przypisuje stacje do najbliższych AP (jak w Fig. 11a)"""
        aps = [n for n in nodes if n.node_type == 'AP']
        stations = [n for n in nodes if n.node_type == 'STA']
        
        if not aps or not stations:
            return
        
        # Pozycje AP i stacji
        ap_positions = np.array([[ap.x, ap.y] for ap in aps])
        sta_positions = np.array([[sta.x, sta.y] for sta in stations])
        
        # Obliczenie macierzy odległości
        distances = cdist(sta_positions, ap_positions)
        
        # Przypisanie każdej stacji do najbliższego AP
        nearest_ap_indices = np.argmin(distances, axis=1)
        
        for i, sta in enumerate(stations):
            sta.associated_ap = aps[nearest_ap_indices[i]].id
    
    def _create_bipartite_graph(self, nodes: List[NetworkNode]) -> Dict:
        """
        Tworzy graf dwudzielny G = (V, E) gdzie V = A ∪ S
        
        Returns:
            Dict z informacjami o grafie dwudzielnym
        """
        aps = [n for n in nodes if n.node_type == 'AP']
        stations = [n for n in nodes if n.node_type == 'STA']
        
        # Zbiory wierzchołków
        A = [ap.id for ap in aps]
        S = [sta.id for sta in stations]
        V = A + S
        
        # Łącza E ⊆ A × S (potencjalne połączenia)
        E = []
        delta_plus = {}  # δ+(v) - łącza wychodzące
        delta_minus = {}  # δ-(v) - łącza przychodzące
        
        # Inicjalizacja
        for node_id in V:
            delta_plus[node_id] = []
            delta_minus[node_id] = []
        
        # Tworzenie łączy między wszystkimi AP a wszystkimi stacjami
        for ap_id in A:
            for sta_id in S:
                edge = (ap_id, sta_id)
                E.append(edge)
                delta_plus[ap_id].append(edge)
                delta_minus[sta_id].append(edge)
        
        return {
            'V': V,  # Wszystkie wierzchołki
            'A': A,  # Punkty dostępowe
            'S': S,  # Stacje
            'E': E,  # Łącza
            'delta_plus': delta_plus,   # δ+(v)
            'delta_minus': delta_minus  # δ-(v)
        }
    
    def calculate_angle(self, from_node: NetworkNode, to_node: NetworkNode) -> float:
        """
        Oblicza kąt kierunku od węzła from_node do węzła to_node
        
        Returns:
            Kąt w radianach [0, 2π], gdzie 0 to kierunek na wschód
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        
        # atan2 zwraca kąt w zakresie [-π, π]
        angle = np.arctan2(dy, dx)
        angle = np.degrees(angle)

        # Konwersja do zakresu [0, 2π]
        return angle % 360

    def calculate_interference_omni(self, 
                                        target_link: Tuple[int, int],
                                        all_transmissions: List[Tuple[int, int]], 
                                        nodes: List[NetworkNode]) -> float:
        """
        Oblicza poziom interferencji ζₑ dla anten omni
        
        Args:
            target_link: (ap_id, sta_id) - łącze docelowe
            all_transmissions: lista wszystkich jednoczesnych transmisji [(ap_id, sta_id), ...]
            nodes: lista wszystkich węzłów
            
        Returns:
            Poziom interferencji + szum w mW
        """
        target_ap_id, target_sta_id = target_link
        
        # Znajdź węzły
        node_dict = {n.id: n for n in nodes}
        target_sta_node = node_dict[target_sta_id]        
        interference=0
        # Suma interferencji od innych transmitujących AP
        for interfering_ap_id, interfering_sta_id in all_transmissions:
            if interfering_ap_id != target_ap_id:  # Pomijamy AP docelowy
                
                # Węzły interferującego łącza
                interfering_ap_node = node_dict[interfering_ap_id]
                interfering_sta_node = node_dict[interfering_sta_id]
                print(f"Interferujący AP {interfering_ap_id} na pozycji ({interfering_ap_node.x:.2f}, {interfering_ap_node.y:.2f})")
                print(f"Interferująca STA {interfering_sta_id} na pozycji ({interfering_sta_node.x:.2f}, {interfering_sta_node.y:.2f})")
                ap_int=np.array([interfering_ap_node.x,interfering_ap_node.y])
                sta=np.array([target_sta_node.x,target_sta_node.y])
                # Moc transmisji interferującego AP
                tx_power_dbm = 20
                d=np.linalg.norm(sta-ap_int)
                print("odleglosc: ",d)
                path_loss_db = calculations(ap_int).path_loss(d,f=2.4)
                # Moc sygnału interferującego w miejscu docelowej stacji
                received_power_dbm = tx_power_dbm - path_loss_db
                received_power_mw = 10**(received_power_dbm / 10)
                interference += received_power_mw
        return interference
    
    def calculate_interference_with_antennas(self, 
                                        target_link: Tuple[int, int],
                                        all_transmissions: List[Tuple[int, int]], 
                                        nodes: List[NetworkNode]) -> float:
        """
        Oblicza poziom interferencji ζₑ z uwzględnieniem charakterystyk kierunkowych anten
        
        Args:
            target_link: (ap_id, sta_id) - łącze docelowe
            all_transmissions: lista wszystkich jednoczesnych transmisji [(ap_id, sta_id), ...]
            nodes: lista wszystkich węzłów
            
        Returns:
            Poziom interferencji + szum w mW
        """
        target_ap_id, target_sta_id = target_link
        
        # Znajdź węzły
        node_dict = {n.id: n for n in nodes}
        target_sta_node = node_dict[target_sta_id]        
        interference=0
        # Suma interferencji od innych transmitujących AP
        for interfering_ap_id, interfering_sta_id in all_transmissions:
            if interfering_ap_id != target_ap_id:  # Pomijamy AP docelowy
                
                # Węzły interferującego łącza
                interfering_ap_node = node_dict[interfering_ap_id]
                interfering_sta_node = node_dict[interfering_sta_id]
                # print(f"Interferujący AP {interfering_ap_id} na pozycji ({interfering_ap_node.x:.2f}, {interfering_ap_node.y:.2f})")
                # print(f"Interferująca STA {interfering_sta_id} na pozycji ({interfering_sta_node.x:.2f}, {interfering_sta_node.y:.2f})")
                # print("int ap node: ",interfering_ap_node,"int sta : ", interfering_sta_node)
                # === KLUCZOWA RÓŻNICA: Dwa różne kąty! ===
                ap_int=np.array([interfering_ap_node.x,interfering_ap_node.y])
                sta=np.array([target_sta_node.x,target_sta_node.y])
                # 1. Kąt transmisji interferującego AP (do swojej stacji)
                #    - Ten kąt określa wzmocnienie anteny nadawczej
                # tx_angle = self.calculate_angle(interfering_ap_node, interfering_sta_node)
                theta_bins, w_fft_dB = calculate_beam_pattern(8, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)
                tx_angle=self.calculate_angle(interfering_ap_node, interfering_sta_node)
                theta_bins_rotated, w_fft_dB_rotated = rotate_beam_pattern(theta_bins, w_fft_dB, tx_angle)
                tx_antenna_gain_db=calculate_power_at_angle(theta_bins_rotated, w_fft_dB_rotated, tx_angle)
                # print("kąt tra int AP do celu to : ",tx_angle, "gain to :",tx_antenna_gain_db)
                # tx_antenna_gain_db = self.get_antenna_gain(interfering_ap_id, tx_angle)
                # 2. Kąt od interferującego AP do stacji odbierającej interferencję 
                #    - Ten kąt określa wzmocnienie w kierunku "ofiary" interferencji
                interference_angle = self.calculate_angle(interfering_ap_node, target_sta_node)
                interference_antenna_gain_db = calculate_power_at_angle(theta_bins_rotated, w_fft_dB_rotated,interference_angle)
                
                # print("kąt tra int AP do naszej STA to : ",interference_angle, "gain to :",interference_antenna_gain_db)
                # Moc transmisji interferującego AP
                tx_power_dbm = 20
                
                # EIRP (Equivalent Isotropically Radiated Power) w kierunku interferencji
                # Używamy wzmocnienia anteny w kierunku stacji odbierającej interferencję
                eirp_dbm = tx_power_dbm + interference_antenna_gain_db
                
                # Straty ścieżki od interferującego AP do docelowej stacji
                d=np.linalg.norm(sta-ap_int)
                # d=np.linalg.norm([interfering_ap_node.x - target_sta_node.x, interfering_ap_node.y - target_sta_node.y])
                # print("odleglosc: ",d)
                path_loss_db = calculations(ap_int).path_loss(d,f=2.4)
                
                # Moc sygnału interferującego w miejscu docelowej stacji
                received_power_dbm = eirp_dbm - path_loss_db
                received_power_mw = 10**(received_power_dbm / 10)
                
                interference += received_power_mw
        
        return interference


    def plot_topology(self, topology_data: Dict, figsize: Tuple[int, int] = (10, 8)):
        """Wizualizuje wygenerowaną topologię"""
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize)
        
        nodes = topology_data['nodes']
        topo_type = topology_data.get('topology_type', 'open_space')
        params = topology_data.get('parameters', {})

        # Jeśli multiroom - rysujemy pokoje jako prostokąty i granice przerywaną linią
        if topo_type == 'multiroom':
            grid_size = params.get('grid_size', (1, 1))
            room_size = params.get('room_size', 10.0)
            rows, cols = grid_size

            for r in range(rows):
                for c in range(cols):
                    x0 = c * room_size
                    y0 = r * room_size
                    # naprzemienne odcienie szarości
                    shade = 0.92 if (r + c) % 2 == 0 else 0.98
                    rect = mpatches.Rectangle((x0, y0), room_size, room_size,
                                              facecolor=str(shade), edgecolor='none', zorder=0)
                    ax.add_patch(rect)

            # rysujemy linie graniczne między pokojami (przerywane)
            for c in range(1, cols):
                x = c * room_size
                ax.plot([x, x], [0, rows * room_size], color='k', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
            for r in range(1, rows):
                y = r * room_size
                ax.plot([0, cols * room_size], [y, y], color='k', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)

        # Rysowanie węzłów (zostawiamy nad wyświetlanymi pokojami)
        for node in nodes:
            if node.node_type == 'AP':
                ax.scatter(node.x, node.y, c='red', s=100, marker='x', 
                          label='AP' if node.id == min(n.id for n in nodes if n.node_type == 'AP') else "", zorder=5)
            else:
                ax.scatter(node.x, node.y, c='blue', s=50, marker='o',
                          label='STA' if node.id == min(n.id for n in nodes if n.node_type == 'STA') else "", zorder=5)
        
        # Rysowanie przypisań
        node_dict = {n.id: n for n in nodes}
        for node in nodes:
            if node.node_type == 'STA' and node.associated_ap is not None:
                ap_node = node_dict[node.associated_ap]
                ax.plot([node.x, ap_node.x], [node.y, ap_node.y], 
                       'gray', alpha=0.3, linestyle='--', zorder=3)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        # ax.set_title(f'Topologia {topo_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Jeśli multiroom ustaw limity osi tak, aby objąć wszystkie pokoje
        if topo_type == 'multiroom':
            cols = params.get('grid_size', (1,1))[1]
            rows = params.get('grid_size', (1,1))[0]
            room_size = params.get('room_size', 60.0)
            ax.set_xlim(0, cols * room_size)
            ax.set_ylim(0, rows * room_size)

        plt.tight_layout()
        # plt.show()
        plt.savefig("scenariusz.pdf",dpi=300, bbox_inches='tight')
        plt.close("all")
        # return fig, ax
def round_sim(num_simulations: int, pattern_type: str, ap_selection: str, seed: int, topology_seed: int):
    np.random.seed(seed)
    f= 2.4 #GHz
    Tx_PWR = 20 # w dBm
    noise=0.0000000004
    Bp=10 # breaking point w metrach
    total_thr=0
    generator = TopologyGenerator()
    topology = generator.generate_multiroom_topology(
        topo_seed=topology_seed,
        grid_size=(2, 2), 
        room_size=20.0
    )
    # print("Topologia wielopokojowa:")
    # print(f"Liczba węzłów: {len(multiroom_topo['nodes'])}")
    # print(f"Liczba AP: {len(multiroom_topo['bipartite_graph']['A'])}")
    # print(f"Liczba STA: {len(multiroom_topo['bipartite_graph']['S'])}")
    # print(f"Liczba potencjalnych łączy: {len(multiroom_topo['bipartite_graph']['E'])}")
    # topology = generator.generate_open_space_topology()
    generator.plot_topology(topology)
    sim_totals =[]
    per_station=[]
    transmission_pairs = []  # Lista wszystkich par (nadawca, odbiorca)
    receiver_sets = []  # Lista zbiorów odbiorców w każdej rundzie
    for sim in range(num_simulations):
        print(f"Symulacja {sim + 1}/{num_simulations}")
        nodes = topology['nodes']
        node_dict={n.id: n for n in nodes}
        aps = [n for n in nodes if n.node_type == 'AP']
        stations = [n for n in nodes if n.node_type == 'STA']
        round_thr=0.0
        # Wybór AP do analizy
        if ap_selection == "pojedyncze":
            selected_aps = np.random.choice(aps, size=1, replace=False)
        elif ap_selection == "wszystkie":
            selected_aps = aps
        elif ap_selection == "losowo":
            selected_aps = np.random.choice(aps, size=np.random.randint(1, len(aps)+1), replace=False)
        elif ap_selection == "inteligentnie":
            # Wybierz AP-y, które mają najwięcej stacji najdalej od siebie (zewnętrzne stacje)
            # Znajdź stacje najdalej od każdego AP
            ap_sta_distances = []
            for ap in aps:
                sta_distances = [(sta, np.sqrt((ap.x - sta.x) ** 2 + (ap.y - sta.y) ** 2)) for sta in stations]
                max_sta, max_dist = max(sta_distances, key=lambda x: x[1])
                ap_sta_distances.append((ap, max_sta, max_dist))
            # Posortuj AP-y po największej odległości do stacji
            ap_sta_distances.sort(key=lambda x: x[2], reverse=True)
            # Wybierz np. dwa AP-y z największymi odległościami do swoich stacji
            selected_aps = [ap_sta_distances[0][0], ap_sta_distances[1][0], ap_sta_distances[2][0]] if len(ap_sta_distances) > 1 else [ap_sta_distances[0][0]]
        else:
            raise ValueError("Nieznana opcja wyboru AP")
        calculations(selected_aps)
        transmissions = [(ap.id, random.choice([sta.id for sta in stations if sta.associated_ap == ap.id])) for ap in selected_aps]
        print(transmissions)
        transmission_pairs.extend(transmissions)
        receiver_sets.append(tuple(sorted([rx for _, rx in transmissions])))
        # print(f"Wybrane AP do analizy: {[ap for ap in selected_aps]}")
        for link in transmissions:
            target_link = (link[0], link[1])
            ap_node = node_dict[link[0]]
            sta_node = node_dict[link[1]]
            ap=np.array([ap_node.x,ap_node.y])
            sta=np.array([sta_node.x,sta_node.y])
            print(f"ap : {ap}, stacja: {sta}")
            d=np.linalg.norm(sta-ap)
            angle = generator.calculate_angle(ap_node,sta_node)
            if pattern_type == "beam":
                theta_bins, w_fft_dB = calculate_beam_pattern(8, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)
                theta_bins_rot, w_fft_dB_rot=rotate_beam_pattern(theta_bins, w_fft_dB, angle)
                gain = calculate_power_at_angle(theta_bins_rot, w_fft_dB_rot, angle)
            else:
                gain=0
            pl = calculations(ap).path_loss(d, f)
            if ap_selection != "pojedyncze":
                if pattern_type=="beam":
                    interference = generator.calculate_interference_with_antennas(target_link, transmissions, nodes)
                    print(f"Interferencja dla łącza {link}: {10 * np.log10(interference):.2f} dB")
                else:
                    interference = generator.calculate_interference_omni(target_link,transmissions,nodes)
                    print(f"Interferencja omni dla łącza {link}: {10 * np.log10(interference):.2f} dB")
            else:
                interference=0.0
            sinr = Tx_PWR + gain - (pl + 10 * np.log10(interference+noise))
            thr = calculations.sinr_to_mcs(sinr)[1]
            per_station.append(thr)
            round_thr+=thr
            total_thr+=round_thr
            print(f"Kąt do STA: {angle:.2f}°, odległość: {d}, zysk anteny: {gain:.2f} dB, path loss: {pl:.2f} dB, interf: {10*np.log10(interference):.2f} dBm, SINR: {sinr:.2f} dB, przepustowość: {thr} Mbps")
        print("-----Całkowita przepustowość po ",sim,"rundzie to :",round_thr,"-----")
        sim_totals.append(round_thr)
    plot_histograms(transmission_pairs,receiver_sets,pattern_type,ap_selection,seed,num_simulations)
    # print(f"\nNajczęściej wybierane pary (top 10):")
    # for i, ((tx, rx), count) in enumerate(sorted_pairs[:10], 1):
    #     freq = count / total_transmissions * 100
    #     print(f"{i}. Para ({tx}, {rx}): {freq:.2f}%")

    # print(f"\nNajczęściej wybierane zbiory odbiorców (top 10):")
    # for i, (rx_set, count) in enumerate(sorted_sets[:10], 1):
    #     freq = count / total_sets * 100
    #     print(f"{i}. Zbiór {rx_set}: {freq:.2f}%")
    final=total_thr/num_simulations
    return float(np.mean(sim_totals)),sim_totals,per_station
        # for ap in selected_aps:
        #     print(f"Analiza AP {idx} na pozycji ({ap.x:.2f}, {ap.y:.2f})")
        #     # else:
        #     #     raise ValueError("Nieznany typ wzoru promieniowania")
        #     stations_for_ap = [sta for sta in stations if sta.associated_ap == ap.id]
        #     print(f"Liczba stacji przypisanych do AP {idx}: {len(stations_for_ap)}")
        #     sta_picked=np.random.choice(stations_for_ap,size=1,replace=False)
        #     sta_picked=sta_picked[0]
        #     print(f"Wybrana stacja: {sta_picked.id} na pozycji ({sta_picked.x:.2f}, {sta_picked.y:.2f})")
        #     d=np.linalg.norm([ap.x - sta_picked.x, ap.y - sta_picked.y])
        #     print(f"  STA {sta_picked} na pozycji ({sta_picked.x:.2f}, {sta_picked.y:.2f}), odległość do AP: {d:.2f} m")
# omni_rand=round_sim(100, "omni", "losowo", 34,34)[1]
# beam_rand=round_sim(100, "beam", "losowo", 34,34)[1]
# omni_sing=round_sim(100, "omni", "pojedyncze", 34,34)[1]
# beam_sing=round_sim(100, "beam", "pojedyncze", 34,34)[1]
# omni_todo=round_sim(100, "omni", "wszystkie", 37)[1]
# beam_todo=round_sim(100, "beam", "wszystkie", 37)[1]
# omni_inte=round_sim(100, "omni", "inteligentnie", 37)[1]
# beam_inte=round_sim(100, "beam", "inteligentnie", 37)[1]
# print(beam_sing)
# results=[beam_rand,omni_rand,beam_sing,omni_sing,beam_todo,omni_todo,beam_inte,omni_inte]
# labels = ["Omni random", "Beam random",
#         "Omni single",  "Beam single",
#         "Omni all", "Beam all",
#         "Omni intelligent", "Beam intelligent"]

# plt.figure(figsize=(10, 6))
# plt.bar(labels, results, color='skyblue')
# plt.xlabel('Parametry wejściowe')
# plt.ylabel('Wynik funkcji')
# plt.title('Wyniki funkcji dla różnych parametrów')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
def multiple_sims(num_sim):
    cdf_results=[]
    for i in range(1,num_sim):
        result=round_sim(100,"beam","losowo",i,34)[2]
        cdf_results+=result
    plot_cdf(cdf_results)
multiple_sims(16)
# plot_means_with_ci(results,["beam_rand","omni_rand","beam_sing","omni_sing","beam_all","omni_all","beam_inte","omni_inte"])
# plot_means_with_ci([beam_sing,omni_sing],["beamforming single","omni single"])
# plot_means_with_ci([beam_todo,omni_todo,beam_inte,omni_inte],["beamforming all","omni all","beamforming intelligently","omni intelligently"])
# plot_boxplots([beam_sing,omni_sing],["beam_sing","omni_sing"])