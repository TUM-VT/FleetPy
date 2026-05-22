import sys # Python標準ライブラリ"sys"モジュールの読み込み（Pythonの実行環境そのものを操作、参照するためのモジュール）
import os # Python標準ライブラリ"os"モジュールの読み込み(ファイル、フォルダ、パスなどのOperating System関連操作)
import numpy as np # 数値計算ライブラリの読み込み
import random # 乱数生成の標準ライブラリ

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # 現在のディレクトリから3段上層のフォルダを参照できるようにする

from src.ml_gym.FleetPyGymInterface import FleetPyGym # フォルダ"src"の中のサブフォルダ"ml.gym"の中のコードファイル"FleetPyMLInterface.py"からクラス"FleetPyGym"を読み込む
from src.ml_gym.hooks_manager import Events # フォルダ"src"の中のサブフォルダ"ml_gym"の中のコードファイル"hooks_manager.py"にあるクラス"Event"を読み込む
from src.ml_gym.Observers.repositioning_observers import SimTimeObserver, DemandForecastObserver, ZoneBasedVehicleStatesObserver, ZoneBasedCurrentDemandObserver # コードファイル"repositioning_observer.py"にある４つのクラスを読み込む（シミュレーション時刻取得、将来需要予測を取得、車両の分布状況を取得、現在の需要を集計）
from src.ml_gym.Actors.repositioning import ZoneBasedRepositioningActor # コードファイル"repositioning"のクラス"ZoneBasedRepositioningActor"を読み込み
from src.misc.globals import * # フォルダ"src"の中のサブフォルダ"misc"の中のコードファイル"globals.py"にある変数や定数を全て読み込む（*は"全て"を示す）
from src.misc.config import ConstantConfig, ScenarioConfig # ファイル"config.py"のクラス"ConstantConfig"と"ScenarioConfig"を読み込み(csvにまとめられたシナリオ設定値を読み取る関数)

from gymnasium import spaces # 強化学習の環境定義や状態と行動を管理する標準ライブラリ"gymnasium"の、状態の型や範囲を定義する機能"spaces"を読み込み
from stable_baselines3 import PPO


# 強化学習の出力をFleetPyの出発／到着ゾーンのペアに変換するアクター（エージェントの一部で、行動を決める要素）
class RLReposition(ZoneBasedRepositioningActor): # カッコ内は親クラスの名称。この子クラスは親クラスの機能を引き継ぐ
    """Actor that translates the RL agent's output into (origin, target) zone pairs for FleetPy.
    ZoneBasedRepositioningActor.translate_action() is the only method you need to override.
    It receives the raw observation dict and the raw action produced by the RL network, and
    must return a list of (origin_zone_id, target_zone_id) tuples. FleetPy then moves one
    idle vehicle per tuple from the origin zone to the target zone.
    ZBR.t_a()が唯一上書きが必要なメソッドであり、これは生の観測辞書とRLで生成された生の行動を受け取る。
    そして出発ゾーン/目的地ゾーンのタプルのリストを返す必要がある。
    すると、FleetPyは1つの空車を出発ゾーンから到着ゾーンに移動させる。
    The current implementation ignores the RL action and instead does a random demand-driven
    matching — replace this logic with your actual action decoding once you have a trained policy.
    現在のコードは、RLの行動を無視してランダムな需要駆動のマッチングをしているだけなので、方策学習が済んだらこのロジックを実際のものに置き換えること
    """

    # RLエージェントの行動をゾーン間再配置のリストに変換
    def translate_action(self, observation, action):
        """Convert the RL agent's action into a list of zone-to-zone repositioning moves.
        :param observation: merged dict from all registered observers
        :param action: raw output of the RL network (MultiDiscrete array in this example);
        :return: list of (origin_zone_id, target_zone_id) tuples; one vehicle moves per tuple.
        """
        print("incoming action: ", action) # 途中経過チェック用に行動を書き出す

        # TODO: translate that into your action here.
        # the output format should be a list of (origin_zone_id, target_zone_id) tuples, e.g.:
        # return [(0, 2), (0, 2), (1, 3)]  # move 2 vehicles from zone 0 to 2, and 1 vehicle from zone 1 to 3

        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_fc_rq_destinations = observation["zone_to_fc_rq_destinations"]
        zone_to_idle_vehicles = observation["zone_to_idle_vehilces"]
        zone_to_overall_available_vehilces = observation["zone_to_overall_available_vehilces"]
        zone_to_current_repositioning_vehicles = observation["zone_to_current_repositioning_vehicles"]

        # Zone -1 is a FleetPy placeholder for vehicles not yet assigned to any zone; exclude it.
        all_zone_ids = list(zone_to_idle_vehicles.keys()) # 全ゾーンID取得（ゾーン別空車台数の辞書のキーを全て書き出す）
        if -1 in all_zone_ids: # 便宜上ゾーンマイナス１（どのゾーンにもいない車を置く）を用いているので、それは除外する
            all_zone_ids.remove(-1)

        list_repo_actions = [] # 行動を格納する空のリスト

        for i in all_zone_ids: # 出発ゾーン毎に繰り返し
            idle = zone_to_idle_vehicles.get(i,0) # 辞書からキーiに対応する値を取得（キーにiが無ければ0を返す）
            if idle == 0: # 出発ゾーンに空車が無ければ、このループは即終了して次のiへ
                continue

            for j in all_zone_ids: # 到着ゾーン毎に繰り返し
                for _ in range(x_ij - y_ij): # ゾーンi→jの移動空車台数
                    list_repo_actions.append((i, j)) # 台数の分だけタプルを追加

        return list_repo_actions


# RLを用いたFleetPy再配置のギムナジウム環境
class TakashiRLRepo(FleetPyGym):
    """Gymnasium environment for RL-based vehicle repositioning in FleetPy.

    Subclasses FleetPyGym, which handles the gymnasium.Env interface and runs
    FleetPy in a background thread. This class is responsible for:
      - loading FleetPy scenario configs
      - defining the action and observation spaces
      - registering observers (what to read from FleetPy) and an actor (what to write back)
      - implementing translate_observation() and reward()

    Expected keys in `config` (passed as env_config to RLlib):
        constant_cfg_path (str): path to FleetPy constant_config.csv
        var_cfg_path      (str): path to FleetPy scenario config CSV
        nr_zones          (int): number of zones in the zone system
    """

    # FleetPyの設定を読み込む
    def __init__(self, config):
        # --- Load FleetPy configs -------------------------------------------
        constant_cfg = ConstantConfig(config["constant_cfg_path"])
        scenario_cfgs = ScenarioConfig(config["var_cfg_path"])

        study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config["constant_cfg_path"]))))# 使用したケーススタディの名称（マンハッタン／シカゴ／ミュンヘン）
        constant_cfg[G_STUDY_NAME] = study_name 
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 1
        constant_cfg["log_level"] = "info"

        # Select which scenario config to use.
        # When running multiple Ray workers you can use config.worker_index to assign
        # each worker a different scenario, e.g.:
        #   scenario_inx = (config.worker_index - 1) % len(scenario_cfgs)
        scenario_inx = 0 # シナリオの選択（例えばマンハッタンには２種類のシナリオがある）
        fleetpy_config = constant_cfg + scenario_cfgs[scenario_inx] # 設定値＝全シナリオ共通設定値＋シナリオ別設定値

        super().__init__(fleetpy_config) # 親クラス（スーパークラス）を参照

        # --- Define Gymnasium spaces ----------------------------------------
        self.nr_zones = config["nr_zones"]
        fleet_size = sum(fleetpy_config["op_fleet_composition"].values())

        # Action: for each of the nr_zones x nr_zones zone-pairs, how many vehicles to move.
        # MultiDiscrete means each element is independently bounded by fleet_size.
        # Adapt this to match the action representation your policy network produces.
        # TODO: replace with your actual action space. The current shape is just a placeholder and doesn't reflect any real constraints (e.g. available idle vehicles in origin zones).
        
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.nr_zones, self.nr_zones),
            dtype=np.float32
        )

        # Observation: flat vector of [idle, forecast_origins, forecast_destinations] per zone.
        # Shape = 3 * nr_zones. Adjust shape and bounds to match your translate_observation() output.
        # TODO: replace with your actual observation space. The current shape and bounds are just placeholders and may not reflect the true range of values in the observation.
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(3 * self.nr_zones,), dtype=np.float32)

        # --- Register observers and actor ------------------------------------
        # All three observers fire at the same hook point so their dicts are merged
        # into one combined observation passed to translate_observation() and reward().
        event = Events.OBSERVE_BEFORE_REPOSITIONING # hooksmanager.pyのクラス"Event"を用いて、再配置前のタイミングを指定
        # TODO: what observations do you want to read from FleetPy? Implement them as AbstractObserver subclasses and register them here. The current ones are just examples.
        sim_observer = SimTimeObserver() # シミュレーション時刻
        self.register_observer(event, sim_observer)
        self.register_observer(event, DemandForecastObserver())
        self.register_observer(event, ZoneBasedVehicleStatesObserver())
        self.register_observer(event, ZoneBasedCurrentDemandObserver())

        # The actor pauses the simulation, hands the observation to the gym loop,
        # waits for the RL action, then writes it back into FleetPy.
        self.register_actor(event, RLReposition())

    # FleetPyの観測データ辞書を固定長ベクトルに変換
    def translate_observation(self, observation):
        """Flatten the raw FleetPy observation dict into a fixed-size numpy vector.

        This is one of two methods you must implement when subclassing FleetPyGym.
        The output shape must match self.observation_space.

        Current encoding (length = 3 * nr_zones):
            [idle_z0, ..., idle_zN, origins_z0, ..., origins_zN, destinations_z0, ..., destinations_zN]

        :param observation: merged dict from all registered observers.
        :return: np.ndarray of shape (3 * nr_zones,), dtype float32.
        """
        print("translate observation", observation)
        # TODO: implement your actual observation translation logic here. The current implementation is just an example that combines some of the observed values into a flat vector, but you can customize it as needed based on what your observers return and what information you want to feed into the RL policy.
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_fc_rq_destinations = observation["zone_to_fc_rq_destinations"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]
        zone_to_overall_available_vehilces = observation["zone_to_overall_available_vehilces"]
        zone_to_current_repositioning_vehicles = observation["zone_to_current_repositioning_vehicles"]

        # Zone -1 is a FleetPy placeholder for vehicles not yet assigned to any zone; exclude it.ゾーンIDの取得ゾーン-1はどのゾーンにも居ない車両を扱うための仮のゾーンであるため、除外
        all_zone_ids = list(zone_to_idle_vehilces.keys())
        all_zone_ids.remove(-1)

        idle = np.array([zone_to_idle_vehilces.get(zone_id, 0) for zone_id in all_zone_ids], dtype=np.float32)
        req_origins = np.array([zone_to_fc_rq_origins.get(zone_id, 0)  for zone_id in all_zone_ids], dtype=np.float32)
        req_destinations = np.array([zone_to_fc_rq_destinations.get(zone_id, 0)  for zone_id in all_zone_ids], dtype=np.float32)

        processed_observation = np.concatenate([idle, req_origins, req_destinations], axis=0).astype(np.float32)
        return processed_observation

    # 報酬関数
    def reward(self, observation, action, actor_type):
        """Compute the scalar reward signal returned to the RL agent after each step.

        This is the second method you must implement when subclassing FleetPyGym.
        The observation dict contains the same keys as in translate_observation().

        The current implementation returns a constant placeholder — replace it with a
        meaningful signal, e.g.:
            - negative mean passenger wait time
            - number of served requests
            - negative total repositioning distance

        :param observation: merged dict from all registered observers.
        :param action: the action that was sent to the actor (raw RL output).
        :param actor_type: type of the actor that triggered this step.
        :return: scalar float reward.
        """
        unmet_demand = sum(observation['zone_to_current_demand'].values())
        reposition_cost = len(action)
        coef_repo = 0.1

        return - (unmet_demand + coef_repo * reposition_cost) 


# 強化学習の実行
if __name__ == "__main__": # このファイルを直接実行した時のみ動く

    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 現ファイルから3階層上のディレクトリへのパス
    
    # FleetPyの入力値を決めるcsvファイルに関する、デフォルト使用／不使用（外から渡す）　の分岐　
    if len(sys.argv) >= 3: # コマンドライン（ターミナル）に引数を3つ以上入力した場合（引数0:実行ファイル、引数1:全シナリオ共通設定のcsv、引数2:シナリオ別の設定csv
        const_config = sys.argv[1]
        sc_config = sys.argv[2]
    else:
        # デフォルト（引数を入力しなかった場合）で用いる２つのcsvファイルへのパス
        scs_path = os.path.join(MAIN_DIR, "studies", "ml_test", "scenarios") # studies→ml_test→scenariosのフォルダへのパス
        const_config = os.path.join(scs_path, "constant_config.csv") # studies→ml_test→scenarios→constant_config.csvのパス
        sc_config = os.path.join(scs_path, "sc_config_repo.csv")

    # env_config is forwarded to FleetPyRepoRL.__init__ as the `config` argument.
    # FleetPy環境（TakashiRepo）に渡す設定辞書
    fleetpy_config = {"nr_zones": 6,
                      "constant_cfg_path": const_config,
                      "var_cfg_path": sc_config
                      }

    # num_env_runners=0 runs rollouts in the main process (easier for debugging).
    # Increase num_env_runners to parallelize data collection across multiple FleetPy instances.
    # TODO: adjust the RLlib config as needed (e.g. learning algorithm, hyperparameters, number of workers, etc.). The current config is just a placeholder to get you started.
    # TODO: or use other lib like stable-baselines3 or your own training loop instead of RLlib if you prefer. The key part is that the environment (FleetPyRepoRL) can be used with any library that supports gymnasium.Env.

    env = TakashiRLRepo(fleetpy_config)
    
    model = PPO('MlpPolicy',
                env,
                verbose=1, # どれだけ詳細なログを出すか　→　0:無し、1:標準、2:詳細デバッグ
                learning_rate=3e-4, # 1回の更新でどれだけパラメータを変えるか
                n_steps=1024, # 環境から何ステップ分の情報を集めてから更新するか
                batch_size=64 # 1回の勾配更新で使うデータ数
                )
    
    model.learn(total_timesteps=100000)