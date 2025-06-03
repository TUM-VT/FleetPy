# Graph type definitions and constants for heterogeneous graphs

# Node types
NODE_TYPES = {
    'REQUEST': 'request',
    'VEHICLE': 'vehicle'
}

# Edge types and relations
EDGE_TYPES = {
    'REQUEST_TO_REQUEST': ('request', 'connects', 'request'),
    'VEHICLE_TO_REQUEST': ('vehicle', 'connects', 'request')
}

# Feature definitions
CATEGORICAL_FEATURES = {
    'req_features': ['status', 'o_pos', 'd_pos'],
    'veh_features': ['type', 'status', 'pos']
}

# Graph column names
EDGE_COLUMNS = ['source', 'target', 'label']

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Computed as 1 - (TRAIN_RATIO + VAL_RATIO)
