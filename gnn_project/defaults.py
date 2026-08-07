LOG_LEVEL_DEFAULT = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

REQUEST_FEATURES_KEY = 'request_features'
VEHICLE_FEATURES_KEY = 'vehicle_features'
REQUEST_REQUEST_GRAPH_KEY = 'request_request_graph'
VEHICLE_REQUEST_GRAPH_KEY = 'vehicle_request_graph'
ASSIGNMENT_KEY = 'assignments'
INIT_ASSIGNMENT_KEY = 'init_assignments'
LABEL_KEY = 'optimal_assign'
INIT_LABEL_KEY = 'init_assign'

TRAIN_MASKS = 'train_masks'
VAL_MASKS = 'val_masks'
TEST_MASKS = 'test_masks'

TRAIN_GRAPHS = 'train_graphs.pt'
VAL_GRAPHS = 'val_graphs.pt'
TEST_GRAPHS = 'test_graphs.pt'

MEANS_FILE = 'means.parquet'
STDS_FILE = 'stds.parquet'
MINS_FILE = 'mins.parquet'
MAXS_FILE = 'maxs.parquet'

FEATURE_DICT = 'feature_dict.pkl'
ONEHOT_COLS = 'onehot_columns.pkl'

PROCESSED = 'processed'
NORM_STATS = 'norm_stats'
MODELS = 'models'
BEST_MODEL = 'best_model.pt'
STUDIES = 'studies'
RESULTS = 'results'
DATA = 'data'
TRAIN = 'train'

GNN_DATALOADER = 'GNNDataLoader'  # Custom GNN data loader
HETERO_GAT = 'HeteroGAT'  # Heterogeneous Graph Attention Network model

TIMESTEP = 'timestep'
ID = 'id'
STATUS = 'status'
TYPE = 'type'
SOURCE = 'source'
TARGET = 'target'

MEANS = 'means'
STDS = 'stds'
MINS = 'mins'
MAXS = 'maxs'

MODEL_STATE_DICT = 'model_state_dict'
POS_WEIGHT = 'pos_weight'
RR_EDGE_DIM = 'rr_edge_dim'
VR_EDGE_DIM = 'vr_edge_dim'

RR_EDGE_NAME = ('request', 'connects', 'request')
VR_EDGE_NAME = ('vehicle', 'connects', 'request')
RV_EDGE_NAME = ('request', 'rev_connects', 'vehicle')

REQUEST = 'request'
VEHICLE = 'vehicle'