# Classifier MLP (architecture, not hyperparameter)
CLASSIFIER_HIDDEN = 256

# Training loop ceilings and standard regularization
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999

# LR range test (already adaptive)
LR_FIND_STEPS = 100
LR_FIND_END = 1e-2
LR_FIND_DIVERGE_FACTOR = 4.0

# CIS circuit losses (internal)
CIRCUIT_TEMPERATURE = 10.0
