"""One-off 3x3 multiroom placement and model experiment."""

import copy
import argparse
import csv
from pathlib import Path
import time

import DDPG as ddpg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from simulator import Config, NetworkNode, Sim, TopologyGenerator


GRID_SIZE = (2, 2)
ROOM_SIZE = 20.0
STATION_OFFSET = 5.0
TOPOLOGY_SEED = 66
NUM_ANTENNAS = 4
EXPERIMENT_EPISODES = 100
STEPS_PER_EPISODE = 200
RESULTS_PATH = Path(__file__).with_name(f"station_placement_results_{TOPOLOGY_SEED}.csv")
THROUGHPUT_PLOT_PATH = Path(__file__).with_name(
    f"station_placement_aggregate_throughput_{TOPOLOGY_SEED}.pdf"
)


def create_experiment_topology(
    topo_seed: int = TOPOLOGY_SEED,
    room_size: float = ROOM_SIZE,
    station_offset: float = STATION_OFFSET,
) -> dict:
    """Create a regular 3x3 topology with each AP and four STAs in a room."""
    if not 0 < station_offset < room_size / 2:
        raise ValueError("station_offset must be positive and less than half the room size")

    generator = TopologyGenerator()
    topology = generator.generate_multiroom_topology(
        topo_seed=topo_seed,
        grid_size=GRID_SIZE,
        room_size=room_size,
        stations_per_room=2,
    )

    nodes: list[NetworkNode] = topology["nodes"]
    aps = [node for node in nodes if node.node_type == "AP"]
    stations = [node for node in nodes if node.node_type == "STA"]
    rows, cols = GRID_SIZE
    # offsets = (
    #     (0.0, station_offset),
    #     (station_offset, 0.0),
    #     (0.0, -station_offset),
    #     (-station_offset, 0.0),
    # )
    offsets = (
            (0.0, station_offset),
            (0.0, -station_offset),
        )

    for ap_index, ap in enumerate(aps):
        row, col = divmod(ap_index, cols)
        center_x = (col + 0.5) * room_size
        center_y = (row + 0.5) * room_size
        ap.x = center_x
        ap.y = center_y

        associated_stations = sorted(
            (station for station in stations if station.associated_ap == ap.id),
            key=lambda station: station.id,
        )
        if len(associated_stations) != len(offsets):
            raise ValueError(f"AP {ap.id} must have exactly four associated stations")

        for station, (offset_x, offset_y) in zip(associated_stations, offsets):
            station.x = center_x + offset_x
            station.y = center_y + offset_y

    topology["parameters"]["room_size"] = room_size
    topology["parameters"]["total_area"] = (cols * room_size, rows * room_size)
    return topology


def plot_experiment_topology(
    topology: dict,
    output_path: Path = Path(__file__).with_name("station_placement_experiment.png"),
) -> None:
    """Draw the room layout and save a labeled placement plot."""
    rows, cols = GRID_SIZE
    room_size = topology["parameters"]["room_size"]
    nodes: list[NetworkNode] = topology["nodes"]
    aps = [node for node in nodes if node.node_type == "AP"]
    stations = [node for node in nodes if node.node_type == "STA"]

    fig, ax = plt.subplots(figsize=(8, 8))
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(
                Rectangle(
                    (col * room_size, row * room_size),
                    room_size,
                    room_size,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=2,
                    zorder=0,
                )
            )
            center_x = (col + 0.5) * room_size
            center_y = (row + 0.5) * room_size
            ax.plot(
                [center_x, center_x],
                [row * room_size, (row + 1) * room_size],
                color="0.85",
                linewidth=0.8,
                zorder=1,
            )
            ax.plot(
                [col * room_size, (col + 1) * room_size],
                [center_y, center_y],
                color="0.85",
                linewidth=0.8,
                zorder=1,
            )

    ax.scatter(
        [station.x for station in stations],
        [station.y for station in stations],
        color="#365f9d",
        s=30,
        zorder=3,
    )
    ax.scatter(
        [ap.x for ap in aps],
        [ap.y for ap in aps],
        color="#365f9d",
        marker="x",
        s=90,
        linewidths=2,
        zorder=4,
    )

    for ap_index, ap in enumerate(aps):
        row, col = divmod(ap_index, cols)
        label_number = col * rows + row + 1
        ax.text(
            ap.x,
            ap.y + room_size * 0.32,
            f"AP {label_number}",
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_xlim(0, cols * room_size)
    ax.set_ylim(0, rows * room_size)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, cols * room_size + 1, room_size))
    ax.set_yticks(np.arange(0, rows * room_size + 1, room_size))
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class ExperimentSim(Sim):
    """Simulator that reuses the exact same placement across resets."""

    def __init__(self, config: Config, topology: dict):
        self._experiment_topology = topology
        super().__init__(config)

    def _build_topology(self) -> None:
        self.topology = copy.deepcopy(self._experiment_topology)
        self.nodes = self.topology["nodes"]
        self.node_dict = {node.id: node for node in self.nodes}
        self.aps = [node for node in self.nodes if node.node_type == "AP"]
        self.stas = [node for node in self.nodes if node.node_type == "STA"]
        self.stas_by_ap = {
            ap.id: [sta for sta in self.stas if sta.associated_ap == ap.id]
            for ap in self.aps
        }


def evaluate_episode(env: ExperimentSim, policy) -> dict[str, float]:
    total_reward = 0.0
    total_throughput = 0.0
    steps = 0

    obs = env.reset()
    while True:
        action = np.asarray(policy(obs), dtype=np.float32).reshape(
            env.num_bs, min(env.num_antennas - 1, env.num_bs - 1)
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_throughput += float(info["aggregate_throughput_mbps"])
        steps += 1
        if done:
            break

    return {
        "average_reward": total_reward / steps,
        "average_throughput_mbps": total_throughput / steps,
    }


def plot_aggregate_throughput(
    episodes: list[int],
    throughput_series: dict[str, list[float]],
    output_path: Path = THROUGHPUT_PLOT_PATH,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    all_values = []

    for index, (label, throughputs) in enumerate(throughput_series.items()):
        if not throughputs:
            continue

        color = colors[index % len(colors)]
        values = np.asarray(throughputs, dtype=float)
        all_values.extend(values.tolist())
        ax.plot(
            episodes,
            values,
            color=color,
            alpha=0.3,
            linewidth=1.0,
            label=f"Raw {label}",
        )

        window = min(20, len(episodes))
        rolling_mean = pd.Series(values).rolling(window=window, min_periods=1).mean()
        rolling_std = (
            pd.Series(values)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )
        ax.plot(
            episodes,
            rolling_mean,
            color=color,
            linewidth=2.0,
            label=f"{label} MA ({window})",
        )
        ax.fill_between(
            episodes,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            color=color,
            alpha=0.1,
        )

    if not all_values:
        raise ValueError("No throughput values found to plot")
    value_min = min(all_values)
    value_max = max(all_values)
    padding = max((value_max - value_min) * 0.12, value_max * 0.01)
    ax.set_ylim(value_min - padding, value_max + padding)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Aggregate OBSS throughput [Mb/slot]")
    ax.set_title("Aggregate OBSS throughput vs. episode")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_saved_results(
    results_path: Path = RESULTS_PATH,
    output_path: Path = THROUGHPUT_PLOT_PATH,
) -> None:
    with results_path.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        raise ValueError(f"No episode results found in {results_path}")

    episodes = [int(row["episode"]) for row in rows]
    columns = {
        "DDPG": "aggregate_obss_throughput_mb_per_slot",
        "NeuralBandit2": "neural_bandit_aggregate_obss_throughput_mb_per_slot",
        "OracleHeuristic": "heuristic_aggregate_obss_throughput_mb_per_slot",
        "FixedNullBaseline": "fixed_null_aggregate_obss_throughput_mb_per_slot",
    }
    throughput_series = {
        label: [float(row[column]) for row in rows]
        for label, column in columns.items()
        if column in rows[0]
    }
    plot_aggregate_throughput(episodes, throughput_series, output_path)
    print(f"Wykres zapisano w: {output_path}", flush=True)


def train_learned_models(config: Config, topology: dict):
    env = ExperimentSim(config, topology)
    action_dim, state_dim = env.get_spaces()
    agent = ddpg.DDPG(state_dim, action_dim)
    neural_bandit = ddpg.NeuralBandit2(action_dim, state_dim)
    noise = ddpg.OUActionNoise(
        mean=np.zeros(action_dim),
        std_deviation=0.05 * np.ones(action_dim),
    )
    bandit_eval_env = ExperimentSim(config, topology)
    training_history = []

    for episode in range(EXPERIMENT_EPISODES):
        obs = env.reset()
        noise.reset()
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_steps = 0
        while True:
            state = obs.flatten()
            action = agent.select_action(state, noise)
            bandit_action = np.asarray(neural_bandit.predict(obs), dtype=np.float32)
            next_obs, reward, done, info = env.step(
                action.reshape(env.num_bs, env.num_antennas - 1)
            )
            next_state = next_obs.flatten()
            agent.replay_buffer.push(
                (state, next_state, action, reward, float(done))
            )

            bandit_action_tensor = ddpg.torch.FloatTensor(
                bandit_action.reshape(1, -1)
            ).to(ddpg.device)
            state_tensor = ddpg.torch.FloatTensor(state.reshape(1, -1)).to(ddpg.device)
            bandit_loss = -agent.critic(state_tensor, bandit_action_tensor).mean()
            neural_bandit.optimizer.zero_grad()
            bandit_loss.backward()
            neural_bandit.optimizer.step()
            neural_bandit.update(bandit_action, obs, reward)

            episode_reward += reward
            episode_throughput += float(info["aggregate_throughput_mbps"])
            episode_steps += 1
            obs = next_obs
            if done:
                break

        training_history.append(
            {
                "reward": episode_reward,
                "aggregate_obss_throughput_mb_per_slot": (
                    episode_throughput / max(1, episode_steps)
                ),
            }
        )
        if len(agent.replay_buffer.storage) >= ddpg.MIN_BUFFER_SIZE:
            agent.update()
        bandit_stats = evaluate_episode(
            bandit_eval_env, lambda state: neural_bandit.predict(state)
        )
        training_history[-1]["neural_bandit_reward"] = bandit_stats[
            "average_reward"
        ]
        training_history[-1][
            "neural_bandit_aggregate_obss_throughput_mb_per_slot"
        ] = bandit_stats["average_throughput_mbps"]
        print(
            f"Trening zakończony: epizod {episode + 1}/{EXPERIMENT_EPISODES}",
            flush=True,
        )

    env.close()
    bandit_eval_env.close()
    return training_history


def run_model_experiment(topology: dict) -> list[dict]:
    config = Config(
        num_bss=GRID_SIZE[0] * GRID_SIZE[1],
        num_antennas=NUM_ANTENNAS,
        seed=TOPOLOGY_SEED,
        min_num_stas=4,
        max_num_stas=4,
        max_steps_episode=STEPS_PER_EPISODE,
        topology_seed=TOPOLOGY_SEED,
    )
    training_history = train_learned_models(config, topology)
    num_bss = config.num_bss
    oracle_agents = [
        ddpg.OracleHeuristicBandit(i, num_bss, NUM_ANTENNAS)
        for i in range(num_bss)
    ]
    fixed_null_agents = [
        ddpg.FixedNullBeamBandit(i, num_bss, NUM_ANTENNAS)
        for i in range(num_bss)
    ]

    eval_env = ExperimentSim(config, topology)
    policies = {
        "heuristic": lambda obs: ddpg.build_heuristic_action(
            oracle_agents, obs, num_bss, NUM_ANTENNAS
        ),
        "fixed_null": lambda obs: ddpg.build_fixed_null_action(
            fixed_null_agents, obs, num_bss, NUM_ANTENNAS
        ),
    }
    fieldnames = [
        "episode",
        "reward",
        "aggregate_obss_throughput_mb_per_slot",
        "neural_bandit_reward",
        "neural_bandit_aggregate_obss_throughput_mb_per_slot",
    ]
    for model_name in policies:
        fieldnames.extend(
            (
                f"{model_name}_reward",
                f"{model_name}_aggregate_obss_throughput_mb_per_slot",
            )
        )
    results = []
    started_at = time.monotonic()
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        results_file.flush()
        for episode in range(EXPERIMENT_EPISODES):
            row = {"episode": episode, **training_history[episode]}
            for model_name, policy in policies.items():
                stats = evaluate_episode(eval_env, policy)
                row[f"{model_name}_reward"] = stats["average_reward"]
                row[f"{model_name}_aggregate_obss_throughput_mb_per_slot"] = stats[
                    "average_throughput_mbps"
                ]
            writer.writerow(row)
            results_file.flush()
            results.append(row)
            elapsed = time.monotonic() - started_at
            print(
                f"Ewaluacja zakończona: epizod {episode + 1}/{EXPERIMENT_EPISODES} "
                f"({elapsed / 60:.1f} min)",
                flush=True,
            )
    eval_env.close()

    print(f"\nWyniki eksperymentu zapisano w: {RESULTS_PATH}")
    plot_saved_results()
    for model_name, column in (
        ("NeuralBandit2", "neural_bandit_aggregate_obss_throughput_mb_per_slot"),
        ("OracleHeuristic", "heuristic_aggregate_obss_throughput_mb_per_slot"),
        ("FixedNullBaseline", "fixed_null_aggregate_obss_throughput_mb_per_slot"),
    ):
        average_throughput = np.mean([row[column] for row in results])
        print(
            f"{model_name}: {average_throughput:.3f} Mbps/slot "
            f"średnio w {EXPERIMENT_EPISODES} epizodach",
            flush=True,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or plot the station placement experiment.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Create the throughput chart from the existing episode CSV without training.",
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_saved_results()
        raise SystemExit(0)

    print(
        f"Start eksperymentu: {EXPERIMENT_EPISODES} epizodów treningu, "
        f"{STEPS_PER_EPISODE} kroków/epizod.",
        flush=True,
    )
    experiment_topology = create_experiment_topology()
    plot_experiment_topology(experiment_topology)
    run_model_experiment(experiment_topology)
