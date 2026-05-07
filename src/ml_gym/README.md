# FleetPy ML Gym

A [Gymnasium](https://gymnasium.farama.org/)-compatible interface for training RL agents against the FleetPy mobility simulation. The example in [example_fleetpy_gym_repo_ray.py](example_fleetpy_gym_repo_ray.py) trains a PPO agent (via Ray RLlib) to reposition vehicles across a zone system.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│  FleetPyGym (gymnasium.Env subclass)                        │
│                                                              │
│  reset() ──► spawns FleetPy in a background Thread          │
│                         │                                    │
│              FleetPy simulation runs                         │
│                         │                                    │
│              Event fired (e.g. OBSERVE_BEFORE_REPOSITIONING) │
│                         │                                    │
│              HookManager triggers Hook                       │
│               ├─ Observers collect state from FleetPy module │
│               └─ Actor pauses, puts observations on Queue    │
│                         │                                    │
│  get_observations() ◄───┘   (blocks until FleetPy yields)   │
│  translate_observation() → flat vector / dict               │
│  RL agent picks action                                       │
│  send_actor_response() ──────────────────────────────────►  │
│                         Actor resumes, applies action        │
│                         FleetPy simulation continues...      │
└──────────────────────────────────────────────────────────────┘
```

The central idea is that the FleetPy simulation is paused at predefined **hook points** (events). At each hook, **Observers** read state from a FleetPy module (e.g. the repositioning algorithm), and an **Actor** hands control back to the gym's `step()` loop to get an action from the RL agent before writing it back into FleetPy.

---

## Key components

### `FleetPyGym` — [FleetPyGymInterface.py](FleetPyGymInterface.py)

The abstract base class you subclass to create your custom environment. It:
- Runs FleetPy in a background `Thread`
- Uses a `Queue` pair to pass observations out and actions in
- Implements `reset()` and `step()` per the Gymnasium API
- Leaves two abstract methods for you to fill in

| Method | What to do |
|---|---|
| `translate_observation(observation)` | Convert the raw dict from observers into your `observation_space` format (e.g. a flat numpy array) |
| `reward(observation, action, actor_type)` | Return a scalar reward signal |

### `HookManager` — [hooks_manager.py](hooks_manager.py)

Manages a registry of `Hook` objects keyed by `Events`. Each `Hook` holds a list of observers and actors. When a hook is triggered from inside the simulation, observers run synchronously, then each actor either computes a local action or blocks on the queue waiting for a response from the gym loop.

### `Events` — [hooks_manager.py](hooks_manager.py)

An enum of moments in the FleetPy simulation where you can intercept:

| Event | When it fires |
|---|---|
| `OBSERVE_BEFORE_REPOSITIONING` | Just before the repositioning algorithm runs — the hook receives the `RepositioningBase` module |
| `OBSERVE_FLEET_STATE_AFTER_RECEIVING_STATUS_UPDATE` | After a vehicle status update, before optimization |
| `ML_OBSERVE` / `ML_ACTION` | Generic hook points for custom use |

### `AbstractObserver` — [Observers/\_\_init\_\_.py](Observers/__init__.py)

```python
class AbstractObserver(ABC):
    def observe(self, fleetpy_module) -> dict:
        ...
```

Observers receive the live FleetPy module at the hook point and return a dict that gets merged into the combined observation dict. The concrete observers for repositioning are in [Observers/repositioning_observers.py](Observers/repositioning_observers.py):

| Observer | Keys added to observation dict |
|---|---|
| `SimTimeObserver` | `sim_time` |
| `DemandForecastObserver` | `zone_to_fc_rq_origins`, `zone_to_fc_rq_destinations` |
| `ZoneBasedVehicleStatesObserver` | `zone_to_idle_vehilces`, `zone_to_overall_available_vehilces`, `zone_to_current_repositioning_vehicles` |

### `AbstractActor` — [Actors/\_\_init\_\_.py](Actors/__init__.py)

```python
class AbstractActor(ABC):
    def translate_action(self, observation, action):
        """Convert RL network output into the format FleetPy expects."""
        return action  # identity by default
```

An actor's `_act()` method is called by the hook. It sends the observation to the gym loop via the queue, waits for an action, then calls `translate_action()` to convert the RL network output into whatever format FleetPy needs before writing it into the simulation.

---

## What the Ray example does

**File:** [example_fleetpy_gym_repo_ray.py](example_fleetpy_gym_repo_ray.py)

### `FleetPyRepoRL(FleetPyGym)`

Wraps the simulation for vehicle repositioning RL training.

**`__init__`:**
- Loads `ConstantConfig` + `ScenarioConfig` from paths in `env_config`
- Supports Ray's `config.worker_index` to give each rollout worker its own output folder
- Defines spaces:
  - `action_space`: `MultiDiscrete(nr_zones × nr_zones × [fleet_size])` — for each zone-pair, how many vehicles to reposition
  - `observation_space`: `Box(shape=(3 * nr_zones,))` — idle counts, forecast origins, forecast destinations per zone
- Registers 3 observers + 1 actor on the `OBSERVE_BEFORE_REPOSITIONING` event

**`translate_observation(observation)`:**
Concatenates three per-zone arrays into a flat `float32` vector of length `3 * nr_zones`:
```
[idle_0, ..., idle_N, req_origins_0, ..., req_origins_N, req_destinations_0, ..., req_destinations_N]
```

**`reward(...)`:**
Currently returns a stub value of `0.001` — this is where real reward shaping (e.g., served requests, wait times) should go.

### `RLReposition(ZoneBasedRepositioningActor)`

Implements `translate_action()` to convert the RL agent's output into a list of `(origin_zone, target_zone)` tuples. Currently it ignores the raw RL action and instead does a random matching between zones with forecast demand and zones with idle vehicles — this is a placeholder for real action decoding.

### Training entry point

```python
config = (
    PPOConfig()
    .environment(FleetPyRepoRL, env_config=fleetpy_config)
    .env_runners(num_env_runners=0)
    .learners(num_learners=0)
)
algo = config.build()
algo.train()
```

---

## How to implement your own gym environment

### Step 1: Decide on your hook event

Choose from `Events` which moment in the simulation you want to intercept. For something other than repositioning (e.g. dispatching, pricing), you may need to add a new `Events` entry and insert a `hook_manager.trigger(event, module)` call into the relevant FleetPy simulation code.

### Step 2: Write your observers

Subclass `AbstractObserver` for each piece of state you need:

```python
from src.ml_gym.Observers import AbstractObserver

class MyObserver(AbstractObserver):
    def observe(self, fleetpy_module) -> dict:
        # fleetpy_module is whatever object FleetPy passes at your chosen event
        return {"my_key": fleetpy_module.some_value}
```

### Step 3: Write your actor

Subclass an existing actor (e.g. `ZoneBasedRepositioningActor`) or `AbstractActor` directly, and implement `translate_action()` to map RL output → FleetPy-compatible commands:

```python
from src.ml_gym.Actors.repositioning import ZoneBasedRepositioningActor

class MyActor(ZoneBasedRepositioningActor):
    def translate_action(self, observation, action):
        # action is the raw output of your RL network
        # return a list of (origin_zone, target_zone) tuples
        return [(origin, target), ...]
```

If you are targeting a completely different part of FleetPy (not repositioning), subclass `AbstractActor` directly and implement the full `_act()` method to write the action back into your `fleetpy_module`.

### Step 4: Subclass `FleetPyGym`

```python
from src.ml_gym.FleetPyGymInterface import FleetPyGym
from src.ml_gym.hooks_manager import Events
from gymnasium import spaces
import numpy as np

class MyFleetPyEnv(FleetPyGym):

    def __init__(self, config):
        # 1. Build the full FleetPy scenario config
        fleetpy_config = ...  # ConstantConfig + ScenarioConfig

        super().__init__(fleetpy_config)

        # 2. Define spaces
        self.observation_space = spaces.Box(low=0, high=1000, shape=(N,), dtype=np.float32)
        self.action_space = spaces.Discrete(K)

        # 3. Register observers and actor on your chosen event
        event = Events.OBSERVE_BEFORE_REPOSITIONING
        self.register_observer(event, MyObserver())
        self.register_actor(event, MyActor())

    def translate_observation(self, observation: dict) -> np.ndarray:
        # Convert the merged observation dict into your observation_space format
        return np.array([observation["my_key"]], dtype=np.float32)

    def reward(self, observation, action, actor_type) -> float:
        # Compute a meaningful reward — e.g. negative wait time, served trips, etc.
        return -observation.get("mean_wait_time", 0)
```

### Step 5: Wire it up to an RL library

```python
# Ray RLlib example
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(MyFleetPyEnv, env_config={"nr_zones": 6, "constant_cfg_path": ..., "var_cfg_path": ...})
    .env_runners(num_env_runners=2)  # each worker gets its own FleetPy thread
)
algo = config.build()
for _ in range(100):
    print(algo.train())
```

---

## Multi-worker notes

Each Ray rollout worker creates its own `FleetPyRepoRL` instance and thus its own background `Thread` running FleetPy. The example isolates worker outputs with:
```python
fleetpy_config[G_SCENARIO_NAME] = base_name + f"_worker_{config.worker_index}"
```
You can also select different scenario configs per worker (see the commented-out `scenario_inx` logic in the example).

---

## File map

```
src/ml_gym/
├── FleetPyGymInterface.py          # FleetPyGym base class (gymnasium.Env)
├── FleetPyMLInterface.py           # Alternative non-gym interface (scripted runs)
├── hooks_manager.py                # HookManager, Hook, Events
├── Observers/
│   ├── __init__.py                 # AbstractObserver
│   ├── repositioning_observers.py  # SimTime, DemandForecast, ZoneVehicleState
│   └── fleet_control_observers.py  # FleetState observer
├── Actors/
│   ├── __init__.py                 # AbstractActor
│   └── repositioning.py           # ZoneBasedRepositioningActor
├── MLClasses/
│   └── MLZoneBasedRepositioning.py # FleetPy repositioning module with ML hook
├── example_fleetpy_gym_repo_ray.py # Ray RLlib PPO training example (this file)
├── example_fleetpy_gym_repo.py     # Scripted (non-RL) example using FleetPyMLInterface
└── writers.py                      # JSONWriter actor for logging observations
```
