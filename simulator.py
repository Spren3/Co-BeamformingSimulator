import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import numpy as np
import gymnasium
from gymnasium import spaces
from basic_scenarios import (
    angle_between,
    angle_between_points_from_perspective,
    calculations,
    compute_mean_ci,
    plot_boxplots,
    plot_cdf,
    plot_histograms,
    plot_means_with_ci,
)
from beam_pattern import (
    calculate_beam_pattern,
    calculate_power_at_angle,
    plot_beam_pattern_cartesian,
    rotate_beam_pattern,
)


def path_loss_db(distance_m: float, f: float) -> float:
    """Free-space path loss in dB for a distance in meters and frequency in GHz."""
    P=35*np.log10(distance_m/10)
    path_loss=40.05 + 20*np.log10((min(distance_m,10)*f)/2.4)
    if distance_m > 10:
        path_loss+=P
    return path_loss


def steering_vector(num_antennas: int, element_spacing: float, angle_deg: float) -> np.ndarray:
    """Steering vector for a ULA at the specified angle in degrees."""
    n = np.arange(num_antennas, dtype=np.float64)
    angle_rad = np.deg2rad(angle_deg)
    return np.exp(1j * 2.0 * np.pi * element_spacing * n * np.sin(angle_rad))


def generate_csi_vector(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    num_antennas: int,
    num_subcarriers: int,
    freq_ghz: float,
    element_spacing: float,
    k_factor: float,
    seed: int,
) -> np.ndarray:
    """Generate a simple channel state information matrix for one BS-STA pair."""
    rng = np.random.default_rng(seed)
    distance_m = float(np.linalg.norm(tx_pos - rx_pos))
    angle_deg = float(np.mod(np.degrees(np.arctan2(rx_pos[1] - tx_pos[1], rx_pos[0] - tx_pos[0])), 360.0))
    steering = steering_vector(num_antennas, element_spacing, angle_deg)

    pl_db = path_loss_db(distance_m, freq_ghz)
    los_gain = 10 ** (-pl_db / 20.0)
    subcarrier_freqs = np.linspace(freq_ghz * 1e9 - 1e6, freq_ghz * 1e9 + 1e6, num_subcarriers)
    csi = np.zeros((num_subcarriers, num_antennas), dtype=np.complex128)

    c = 3e8
    for s, f in enumerate(subcarrier_freqs):
        delay_phase = 2.0 * np.pi * f * (distance_m / c)
        los_component = np.exp(1j * delay_phase) * steering
        nlos_component = (rng.normal(scale=0.05, size=num_antennas) +
                          1j * rng.normal(scale=0.05, size=num_antennas))
        weight_los = np.sqrt(k_factor / (k_factor + 1.0))
        weight_nlos = np.sqrt(1.0 / (k_factor + 1.0))
        csi[s, :] = los_gain * (weight_los * los_component + weight_nlos * nlos_component)

    return csi


def apply_beamforming(csi: np.ndarray, beam_weights: np.ndarray) -> np.ndarray:
    """Apply beamforming weights to the CSI and return the effective channel per subcarrier."""
    return np.einsum('sa,a->s', np.asarray(csi), np.asarray(beam_weights))


def build_feature_vector(
    ap_pos: np.ndarray,
    sta_pos: np.ndarray,
    dist: float,
    beam_angle: float,
    pl_db: float,
    tx_gain_db: float,
    interference_db: float,
    sinr_db: float,
    rate: float,
    effective_csi: np.ndarray,
) -> np.ndarray:
    """Create a feature vector from link statistics and CSI magnitudes/phases."""
    scalar_features = np.array([
        float(dist),
        float(beam_angle),
        float(pl_db),
        float(tx_gain_db),
        float(interference_db),
        float(sinr_db),
        float(rate),
    ], dtype=np.float64)
    csi_mag = np.abs(np.asarray(effective_csi)).reshape(-1)
    csi_phase = np.angle(np.asarray(effective_csi)).reshape(-1)
    return np.concatenate((scalar_features, csi_mag, csi_phase)).astype(np.float32)


def csi_dataset_to_dataframe(features: np.ndarray, labels: np.ndarray, num_subcarriers: int, num_antennas: int, mode: str = 'mag_phase'):
    """Convert collected CSI features to a pandas DataFrame using per-subcarrier beamformed CSI."""
    import pandas as pd

    scalar_cols = ['dist_m', 'beam_angle_deg', 'pl_db', 'tx_gain_db', 'interference_db', 'sinr_db', 'rate']
    csi_cols = []
    if mode == 'mag_phase':
        for s in range(num_subcarriers):
            csi_cols.append(f'mag_s{s}')
            csi_cols.append(f'phase_s{s}')
    elif mode == 'mag':
        for s in range(num_subcarriers):
            csi_cols.append(f'mag_s{s}')
    else:
        for s in range(num_subcarriers):
            csi_cols.append(f'phase_s{s}')

    column_names = scalar_cols + csi_cols
    df = pd.DataFrame(features, columns=column_names)
    df['label'] = labels
    return df

@dataclass
class Config:
    num_bss: int
    num_antennas: int
    seed: int
    min_num_stas: int = 1
    max_num_stas: int = 1
    max_steps_episode: int = 200
    channel_freq: float = 2.4
    bw_mhz: float = 20.0
    tick: float = 0.05
    tx_power_dbm: float = 20.0
    noise_mw: float = 4e-10
    area_size: float = 75.0
    topology_seed: int = 0
    channel_update_interval_in_ticks: int = 1
    move_prob: float = 1.0
    client_radius: float = 8.0
    tick: float = 0.05

class Sim:
    def __init__(self, config: Config):
        self.config = config
        self.num_bs = config.num_bss
        self.num_antennas = config.num_antennas
        self.channel_freq = config.channel_freq
        self.bw_mhz = config.bw_mhz
        self.tx_power_dbm = getattr(config, 'tx_power_dbm', 20.0)
        self.noise_mw = getattr(config, 'noise_mw', 4e-10)
        self.max_steps_episode = config.max_steps_episode
        self.area_size = getattr(config, 'area_size', 75.0)
        self.tick = config.tick
        self.topology_seed = getattr(config, 'topology_seed', config.seed)
        self.channel_update_interval_in_ticks = getattr(config, 'channel_update_interval_in_ticks', 1)

        action_dim = min(max(1, self.num_antennas - 1), max(1, self.num_bs - 1))
        self.observation_space = spaces.Box(
            low=np.zeros((self.num_bs, self.num_bs * 2), dtype=np.float32),
            high=np.ones((self.num_bs, self.num_bs * 2), dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros((self.num_bs, action_dim), dtype=np.float32),
            high=np.ones((self.num_bs, action_dim), dtype=np.float32),
            dtype=np.float32,
        )

        self.num_subcarriers = 32
        self.element_spacing = 0.5
        self.k_factor = 10.0

        self._generator = TopologyGenerator()
        self.num_steps = 0
        self._episode_counter = 0
        self.reset()

    def get_spaces(self):
        action_dim = self.num_bs * min(max(1, self.num_antennas - 1), max(1, self.num_bs - 1))
        state_dim = self.num_bs * self.num_bs * 2
        return action_dim, state_dim

    def _build_topology(self):
        episode_seed = self.topology_seed + self._episode_counter
        self.topology = self._generator.generate_open_space_topology(
            topo_seed=episode_seed,
            area_size=self.area_size,
            num_aps=self.num_bs,
            stations_per_ap=(self.config.min_num_stas, self.config.max_num_stas),
        )
        # self.topology = self._generator.generate_multiroom_topology(
        #     topo_seed=episode_seed,
        #     grid_size=(3,3),
        #     room_size=20.0,
        #     stations_per_room=self.config.max_num_stas,
        # )
        self.nodes = self.topology['nodes']
        self.node_dict = {node.id: node for node in self.nodes}
        self.aps = [node for node in self.nodes if node.node_type == 'AP']
        self.stas = [node for node in self.nodes if node.node_type == 'STA']
        self.stas_by_ap = {
            ap.id: [sta for sta in self.stas if sta.associated_ap == ap.id]
            for ap in self.aps
        }

    def reset(self):
        self._episode_counter += 1
        self.num_steps = 0
        self.csi_features = []
        self.csi_labels = []
        self._build_topology()
        return self.get_observation()

    def _get_active_stas(self):
        active_stas = []
        for ap in self.aps:
            stas_for_ap = self.stas_by_ap.get(ap.id, []) or self.stas
            index = self.num_steps % len(stas_for_ap)
            active_stas.append(stas_for_ap[index])
        return active_stas

    def get_observation(self):
        obs = np.zeros((self.num_bs, self.num_bs * 2), dtype=np.float32)
        active_stas = self._get_active_stas()
        for i, ap in enumerate(self.aps):
            target_sta = active_stas[i]
            wanted_angle = self._generator.calculate_angle(ap, target_sta) / 360.0
            wanted_dist = np.linalg.norm(np.array([ap.x, ap.y]) - np.array([target_sta.x, target_sta.y]))
            wanted_pl = self.dist_to_normalized_pl(path_loss_db(wanted_dist, self.channel_freq))
            interferers = []
            for j, other_ap in enumerate(self.aps):
                if j == i:
                    continue
                other_sta = active_stas[j]
                angle = self._generator.calculate_angle(ap, other_sta) / 360.0
                dist = np.linalg.norm(np.array([ap.x, ap.y]) - np.array([other_sta.x, other_sta.y]))
                pl = self.dist_to_normalized_pl(path_loss_db(dist, self.channel_freq))
                interferers.append([angle, pl])
            obs[i, :] = np.concatenate(([wanted_angle, wanted_pl], np.array(interferers).flatten()))
        return obs

    def dist_to_normalized_pl(self, pl_db: float) -> float:
        thr = 100.0
        normalized_pl = (thr - pl_db) / (thr / 2)
        return float(np.clip(normalized_pl, 0.0, 1.0))

    def step(self, action: np.ndarray):
        assert action.shape[0] == self.num_bs
        assert action.shape[1] == min(max(1, self.num_antennas - 1), max(1, self.num_bs - 1))
        active_stas = self._get_active_stas()
        rates = []
        channel_update = (self.num_steps % self.channel_update_interval_in_ticks == 0)

        for i, ap in enumerate(self.aps):
            target_sta = active_stas[i]
            beam_angle = self._generator.calculate_angle(ap, target_sta)
            null_angles = action[i]
            null_angles = null_angles[~np.isnan(null_angles)] if null_angles.size else null_angles
            nulls_rad = np.asarray(null_angles * 360.0 / 360.0) / 360.0 * np.pi if null_angles.size else np.array([])
            theta_bins, w_fft_dB = calculate_beam_pattern(
                self.num_antennas, 0.5, 0, nulls_rad
            )
            theta_rotated, w_fft_dB_rotated = rotate_beam_pattern(theta_bins, w_fft_dB, beam_angle)
            tx_gain_db = calculate_power_at_angle(theta_rotated, w_fft_dB_rotated, beam_angle)

            dist = np.linalg.norm(np.array([ap.x, ap.y]) - np.array([target_sta.x, target_sta.y]))
            pl_db = path_loss_db(dist, self.channel_freq)

            interference_mw = 0.0
            for j, interfering_ap in enumerate(self.aps):
                if j == i:
                    continue
                interfering_sta = active_stas[j]
                curr_null_angles = action[j]
                curr_null_angles = curr_null_angles[~np.isnan(curr_null_angles)] if curr_null_angles.size else curr_null_angles
                curr_nulls_rad = np.asarray(curr_null_angles * 360.0 / 360.0) / 360.0 * np.pi if curr_null_angles.size else np.array([])
                int_theta_bins, int_w_fft_dB = calculate_beam_pattern(
                    self.num_antennas, 0.5, 0, curr_nulls_rad
                )
                int_beam_angle = self._generator.calculate_angle(interfering_ap, interfering_sta)
                int_theta_rot, int_w_fft_dB_rot = rotate_beam_pattern(
                    int_theta_bins, int_w_fft_dB, int_beam_angle
                )
                interference_angle = self._generator.calculate_angle(interfering_ap, target_sta)
                interference_gain_db = calculate_power_at_angle(
                    int_theta_rot, int_w_fft_dB_rot, interference_angle
                )
                int_dist = np.linalg.norm(np.array([interfering_ap.x, interfering_ap.y]) - np.array([target_sta.x, target_sta.y]))
                received_power_dbm = self.tx_power_dbm + interference_gain_db - path_loss_db(int_dist, self.channel_freq)
                interference_mw += 10 ** (received_power_dbm / 10)

            sinr_db = self.tx_power_dbm + tx_gain_db - (
                pl_db + 10 * np.log10(interference_mw + self.noise_mw)
            )
            _, rate = calculations.sinr_to_mcs(sinr_db, channel_width=int(self.bw_mhz))
            rate = float(rate)
            rates.append(rate)

            if channel_update:
                csi = generate_csi_vector(
                    np.array([ap.x, ap.y]),
                    np.array([target_sta.x, target_sta.y]),
                    num_antennas=self.num_antennas,
                    num_subcarriers=self.num_subcarriers,
                    freq_ghz=self.channel_freq,
                    element_spacing=self.element_spacing,
                    k_factor=self.k_factor,
                    seed=int(self.config.seed + self.num_steps + i + target_sta.id if hasattr(target_sta, 'id') else self.config.seed + self.num_steps + i),
                )
                beam_power_db = calculate_power_at_angle(theta_rotated, w_fft_dB_rotated, beam_angle)
                beam_power_linear = 10 ** (beam_power_db / 20.0)
                effective_csi = np.mean(csi, axis=1) * beam_power_linear
                interference_db = 10 * np.log10(interference_mw + self.noise_mw)
                feature = build_feature_vector(
                    np.array([ap.x, ap.y]),
                    np.array([target_sta.x, target_sta.y]),
                    dist,
                    beam_angle,
                    pl_db,
                    tx_gain_db,
                    interference_db,
                    sinr_db,
                    rate,
                    effective_csi,
                )
                self.csi_features.append(feature)
                self.csi_labels.append(rate)

        # print(f"DEBUG SLOT: total_rate={sum(rates):.2f}, per_rate={[round(r,2) for r in rates]}")

        aggregate_throughput = float(np.sum(rates))
        reward = float(np.sum(np.log2(np.maximum(rates, 1e-6))))
        aggregate_throughput_mbps = float(np.sum(rates))
        aggregate_throughput_mb_per_slot = aggregate_throughput_mbps * self.tick
        self.num_steps += 1
        obs = self.get_observation()
        done = self.num_steps >= self.max_steps_episode
        return obs, reward, done, {"aggregate_throughput_mbps": aggregate_throughput_mbps, "rates": rates}

    def close(self):
        return None

    def get_csi_dataset(self, use_pandas: bool = False, pandas_mode: str = 'mag_phase'):
        features = (
            np.vstack(self.csi_features)
            if self.csi_features
            else np.zeros((0, 7 + 2 * self.num_subcarriers), dtype=np.float32)
        )
        labels = np.array(self.csi_labels, dtype=np.float32)
        if use_pandas:
            return csi_dataset_to_dataframe(
                features,
                labels,
                self.num_subcarriers,
                self.num_antennas,
                mode=pandas_mode,
            )
        return {
            'features': features,
            'labels': labels,
        }

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
        
    def generate_multiroom_topology(self,
                                  topo_seed: int, 
                                  grid_size: tuple[int, int], 
                                  room_size: float,
                                  stations_per_room: int = 4) -> dict:
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
                                   stations_per_ap: tuple[int, int] = (3, 4),
                                   station_std: float = 10.0) -> dict:
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
    
    def _reassign_to_nearest_ap(self, nodes: list[NetworkNode]):
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
    
    def _create_bipartite_graph(self, nodes: list[NetworkNode]) -> dict:
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
                                        target_link: tuple[int, int],
                                        all_transmissions: list[tuple[int, int]], 
                                        nodes: list[NetworkNode]) -> float:
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
                                        target_link: tuple[int, int],
                                        all_transmissions: list[tuple[int, int]], 
                                        nodes: list[NetworkNode]) -> float:
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


    def plot_topology(self, topology_data: dict, figsize: tuple[int, int] = (10, 8)):
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

            # rysujemy przerywane linie graniczne między pokojami
            for c in range(1, cols):
                x = c * room_size
                ax.plot([x, x], [0, rows * room_size], color='k', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
            for r in range(1, rows):
                y = r * room_size
                ax.plot([0, cols * room_size], [y, y], color='k', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)

        # Rysowanie węzłów
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
        plt.savefig("scenariusz.pdf",dpi=300, bbox_inches='tight')
        plt.close("all")
        # return fig, ax
def round_sim(
    num_simulations: int,
    pattern_type: str,
    ap_selection: str,
    seed: int,
    topology_seed: int,
    record_csi: bool = False
):
    np.random.seed(seed)
    f= 2.4 #GHz
    Tx_PWR = 20 # w dBm
    noise=0.0000000004
    Bp=10 # breaking point w metrach
    total_thr=0
    dataset_features = []
    dataset_labels = []
    dataframes = []
    generator = TopologyGenerator()
    topology = generator.generate_multiroom_topology(
        topo_seed=topology_seed,
        grid_size=(2, 2), 
        room_size=60.0
    )
    # print("Topologia wielopokojowa:")
    # print(f"Liczba węzłów: {len(multiroom_topo['nodes'])}")
    # print(f"Liczba AP: {len(multiroom_topo['bipartite_graph']['A'])}")
    # print(f"Liczba STA: {len(multiroom_topo['bipartite_graph']['S'])}")
    # print(f"Liczba potencjalnych łączy: {len(multiroom_topo['bipartite_graph']['E'])}")
    # topology = generator.generate_open_space_topology(topology_seed)
    generator.plot_topology(topology)
    sim_totals =[]
    per_station={}
    transmission_pairs = []  # Lista wszystkich par (nadawca, odbiorca)
    receiver_sets = []  # Lista zbiorów odbiorców w każdej rundzie
    nodes = topology['nodes']
    node_dict={n.id: n for n in nodes}
    aps = [n for n in nodes if n.node_type == 'AP']
    stations = [n for n in nodes if n.node_type == 'STA']
    all_station_ids = [sta.id for sta in stations]
    per_station = {sta_id: [] for sta_id in all_station_ids}
    print("TESTOWO slownik per station: ",per_station)
    for sim in range(num_simulations):
        print(f"Symulacja {sim + 1}/{num_simulations}")
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
                # print(max_sta,max_dist)
                ap_sta_distances.append((ap, max_sta, max_dist))
            # Posortuj AP-y po największej odległości do stacji
            ap_sta_distances.sort(key=lambda x: x[2], reverse=True)
            # print(ap_sta_distances)
            # Wybierz np. dwa AP-y z największymi odległościami do swoich stacji
            selected_aps = [ap_sta_distances[0][0], ap_sta_distances[1][0],ap_sta_distances[2][0]] if len(ap_sta_distances) > 1 else [ap_sta_distances[0][0]]
            # print("selected APS:",selected_aps)
        else:
            raise ValueError("Nieznana opcja wyboru AP")
        calculations(selected_aps)
        transmissions = [(ap.id, random.choice([sta.id for sta in stations if sta.associated_ap == ap.id])) for ap in selected_aps]
        print(transmissions)
        transmission_pairs.extend(transmissions)
        receiver_sets.append(tuple(sorted([rx for _, rx in transmissions])))
        round_throughputs = {}
        # print(f"Wybrane AP do analizy: {[ap for ap in selected_aps]}")
        for link in transmissions:
            target_link = (link[0], link[1])
            ap_node = node_dict[link[0]]
            sta_node = node_dict[link[1]]
            sta_id=link[1]
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
            # if sta_id not in per_station:
            #     per_station[sta_id]=[]
            # per_station[sta_id].append(thr)
            round_thr+=thr
            total_thr+=round_thr
            print(f"Kąt do STA: {angle:.2f}°, odległość: {d}, zysk anteny: {gain:.2f} dB, path loss: {pl:.2f} dB, interf: {10*np.log10(interference):.2f} dBm, SINR: {sinr:.2f} dB, przepustowość: {thr} Mbps")
        print("-----Całkowita przepustowość po ",sim,"rundzie to :",round_thr,"-----")
        sim_totals.append(round_thr)
        for sta_id in all_station_ids:
            if sta_id in round_throughputs:
                per_station[sta_id].append(round_throughputs[sta_id])
            else:
                per_station[sta_id].append(0.0)
    plot_histograms(transmission_pairs,receiver_sets,pattern_type,ap_selection,seed,num_simulations)
    # print(f"\nNajczęściej wybierane pary (top 10):")
    # for i, ((tx, rx), count) in enumerate(sorted_pairs[:10], 1):
    #     freq = count / total_transmissions * 100
    #     print(f"{i}. Para ({tx}, {rx}): {freq:.2f}%")
    # print(f"\nNajczęściej wybierane zbiory odbiorców (top 10):")
    # for i, (rx_set, count) in enumerate(sorted_sets[:10], 1):
    #     freq = count / total_sets * 100
    #     print(f"{i}. Zbiór {rx_set}: {freq:.2f}%")
    # station_avg_thr = []
    # for sta_id, throughputs in per_station.items():
    #     avg_thr = np.mean(throughputs)
    #     station_avg_thr.append(avg_thr)
    station_avg_thr = []
    for sta_id in all_station_ids:
        throughputs = per_station[sta_id]
        avg_thr = np.mean(throughputs)
        station_avg_thr.append(avg_thr)
        # print(f"Stacja {sta_id}: średni throughput = {avg_thr:.2f} Mbps (z {len(throughputs)} symulacji, "f"{sum(1 for t in throughputs if t > 0)} transmisji)")
    final=total_thr/num_simulations
    return float(np.mean(sim_totals)), sim_totals, station_avg_thr
    return float(np.mean(sim_totals)), sim_totals, station_avg_thr, dataframes
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
# omni_todo=round_sim(100, "omni", "wszystkie", 34,34)[1]
if __name__ == '__main__':
    beam_todo = round_sim(100, "beam", "wszystkie", 70, 34, True, 32, True)[3]
    print(beam_todo)
    # beam_todo = round_sim(100, "beam", "wszystkie", 34, 34)[1]
    # omni_inte = round_sim(100, "omni", "inteligentnie", 37)[1]
    # beam_inte = round_sim(100, "beam", "inteligentnie", 37)[1]
    # results = [beam_rand, omni_rand, beam_sing, omni_sing, beam_todo, omni_todo, beam_inte, omni_inte]
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
    cdf_results2=[]
    cdf_results3=[]
    cdf_results4=[]
    omni_todo=[]
    beam_todo=[]
    omni_inte=[]
    beam_inte=[]
    beam_sing=[]
    omni_sing=[]
    beam_rand=[]
    omni_rand=[]
    for i in range(700,num_sim+900):
        # result=round_sim(100,"beam","pojedyncze",i,34)[2]
        # result=round_sim(100,"beam","inteligentnie",i,34)[2]
        # result=round_sim(100,"beam","wszystkie",i,34)[2]
        # result=round_sim(100,"beam","losowo",i,34)[2]
        # cdf_results+=result
        # result2=round_sim(100,"omni","pojedyncze",i,34)[2]
        # result2=round_sim(100,"omni","inteligentnie",i,34)[2]
        # result2=round_sim(100,"omni","losowo",i,34)[2]
        # cdf_results2+=result2
        # result3=round_sim(100,"beam","wszystkie",i,34)[2]
        # cdf_results3+=result3
        # result4=round_sim(100,"omni","wszystkie",i,34)[2]
        # cdf_results4+=result4
        ### seed 34 do topologii 4.3 -------
        ### ------- DO BOXPLOTOW CHAPTER 5
        j=200+i
        # omni_rand.append(round_sim(100, "omni", "losowo", i,j)[0])
        # beam_rand.append(round_sim(100, "beam", "losowo", i,j)[0])
        # omni_sing.append(round_sim(100, "omni", "pojedyncze", i,j)[0])
        # beam_sing.append(round_sim(100, "beam", "pojedyncze", i,j)[0])
        # omni_todo.append(round_sim(100, "omni", "wszystkie", i,j)[0])
        # beam_todo.append(round_sim(100, "beam", "wszystkie", i,j)[0])
        # omni_inte.append(round_sim(100, "omni", "inteligentnie", i,j)[0])
        # beam_inte.append(round_sim(100, "beam", "inteligentnie", i,j)[0])
        #### BARCHARTY CHAPTER 4
        # omni_rand+=round_sim(100, "omni", "losowo", i,34)[1]
        # beam_rand+=round_sim(100, "beam", "losowo", i,34)[1]
        # omni_sing+=round_sim(100, "omni", "pojedyncze", i,34)[1]
        # beam_sing+=round_sim(100, "beam", "pojedyncze", i,34)[1]
        # omni_todo+=round_sim(100, "omni", "wszystkie", i,34)[1]
        # beam_todo+=round_sim(100, "beam", "wszystkie", i,34)[1]
        beam_todo+=round_sim(100, "beam", "wszystkie", i,34,True,32,True)[1]
        # omni_inte+=round_sim(100, "omni", "inteligentnie", i,34)[1]
        # beam_inte+=round_sim(100, "beam", "inteligentnie", i,34)[1]
    # plot_means_with_ci([omni_sing,beam_sing],["omni single","beam single"])  
    # plot_means_with_ci([omni_sing,beam_sing,omni_rand,beam_rand],["omni single","beam single","omni random","beamforming random"])
    # plot_means_with_ci([omni_rand,beam_rand,omni_todo,beam_todo],["omni random","beamforming random","omni all","beamforming all"])
    # plot_means_with_ci([omni_todo,beam_todo,omni_inte,beam_inte],["omni all","beamforming all","omni heuristic","beamforming heuristic"])
    # results=[omni_sing,beam_sing,omni_rand,beam_rand,omni_todo,beam_todo,omni_inte,beam_inte]
    # labels = ["Beamforming single", "omni single",
        # "Beamforming random", "Omni random",
        # "Beamforming all", "Omni all",
        # "Beamforming intelligent","Omni intelligent"]
    # labels=["Single (O)", "Single (B)", "Random (O)", "Random (B)","All (O)", "All (B)","Heuristic (O)", "Heuristic (B)"]
    # plot_boxplots(results,labels)
    # plot_cdf(cdf_results,cdf_results2,cdf_results3,cdf_results4)
# if __name__ == "__main__":
#     multiple_sims(1)
# plot_means_with_ci(results,["beam_rand","omni_rand","beam_sing","omni_sing","beam_all","omni_all","beam_inte","omni_inte"])
# plot_means_with_ci([beam_sing,omni_sing],["beamforming single","omni single"])
# plot_means_with_ci([beam_todo,omni_todo,beam_inte,omni_inte],["beamforming all","omni all","beamforming intelligently","omni intelligently"])
# plot_boxplots([beam_sing,omni_sing],["beam_sing","omni_sing"])