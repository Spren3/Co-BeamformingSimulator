import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from beam_pattern import calculate_beam_pattern, calculate_power_at_angle, rotate_beam_pattern, plot_beam_pattern_cartesian


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
                                   area_size: float = 75.0,
                                   num_aps: int = 4,
                                   stations_per_ap: Tuple[int, int] = (3, 5),
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
    
    def plot_topology(self, topology_data: Dict, figsize: Tuple[int, int] = (10, 8)):
        """Wizualizuje wygenerowaną topologię"""
        fig, ax = plt.subplots(figsize=figsize)
        
        nodes = topology_data['nodes']
        walls = topology_data['walls']
        
        # Rysowanie ścian
        for wall in walls:
            (x1, y1), (x2, y2) = wall
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.7)
        
        # Rysowanie węzłów
        for node in nodes:
            if node.node_type == 'AP':
                ax.scatter(node.x, node.y, c='red', s=100, marker='x', 
                          label='AP' if node.id == min(n.id for n in nodes if n.node_type == 'AP') else "")
            else:
                ax.scatter(node.x, node.y, c='blue', s=50, marker='o',
                          label='STA' if node.id == min(n.id for n in nodes if n.node_type == 'STA') else "")
        
        # Rysowanie przypisań
        node_dict = {n.id: n for n in nodes}
        for node in nodes:
            if node.node_type == 'STA' and node.associated_ap is not None:
                ap_node = node_dict[node.associated_ap]
                ax.plot([node.x, ap_node.x], [node.y, ap_node.y], 
                       'gray', alpha=0.3, linestyle='--')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Topologia {topology_data["topology_type"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
def round_sim(num_simulations: int, pattern_type: str, ap_selection: str, seed: int):
    np.random.seed(seed)
    f= 2.4 #GHz
    Tx_PWR = 20 # w dBm
    noise=0.0000000004
    Bp=10 # breaking point w metrach

    generator = TopologyGenerator()
    multiroom_topo = generator.generate_multiroom_topology(
        grid_size=(2, 2), 
        room_size=20.0
    )
    print("Topologia wielopokojowa:")
    print(f"Liczba węzłów: {len(multiroom_topo['nodes'])}")
    print(f"Liczba AP: {len(multiroom_topo['bipartite_graph']['A'])}")
    print(f"Liczba STA: {len(multiroom_topo['bipartite_graph']['S'])}")
    print(f"Liczba potencjalnych łączy: {len(multiroom_topo['bipartite_graph']['E'])}")
    
    for sim in range(num_simulations):
        print(f"Symulacja {sim + 1}/{num_simulations}")
        topology = generator.generate_open_space_topology()
        nodes = topology['nodes']
        aps = [n for n in nodes if n.node_type == 'AP']
        stations = [n for n in nodes if n.node_type == 'STA']
        
        # Parametry symulacji
        
        # Wybór AP do analizy
        if ap_selection == "pojedyncze":
            selected_aps = np.random.choice(aps, size=1, replace=False)
        elif ap_selection == "wszystkie":
            selected_aps = aps
        elif ap_selection == "losowo":
            selected_aps = [np.random.choice(aps)]
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
            selected_aps = [ap_sta_distances[0][0], ap_sta_distances[1][0]] if len(ap_sta_distances) > 1 else [ap_sta_distances[0][0]]
        else:
            raise ValueError("Nieznana opcja wyboru AP")
        
        for ap in selected_aps:
            idx = ap.id
            print(f"Analiza AP {idx} na pozycji ({ap.x:.2f}, {ap.y:.2f})")
            
            # Obliczenie wzoru promieniowania
            if pattern_type == "beam":
                theta_bins, w_fft_dB = calculate_beam_pattern(8, 0.5, 0, np.asarray(np.linspace(-60, 60, 11)) / 360 * np.pi)
                theta_bins_rot, w_fft_dB_rot = rotate_beam_pattern(theta_bins, w_fft_dB, angle_to_rotate)
            else:
                raise ValueError("Nieznany typ wzoru promieniowania")
            
            for sta in stations:
                d = np.sqrt((ap.x - sta.x) ** 2 + (ap.y - sta.y) ** 2)
                
                # Obliczenie kąta między AP a STA
                angle = np.degrees(np.arctan2(sta.y - ap.y, sta.x - ap.x)) % 360
                gain = calculate_power_at_angle(theta_bins, w_fft_dB, angle)
                pl = calculations(aps).path_loss(d, frequency)