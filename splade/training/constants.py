# Architecture (not hyperparameters)
CLASSIFIER_HIDDEN = 256

# Training loop bounds
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10

# Regularization (established defaults, Müller et al. 2019)
LABEL_SMOOTHING = 0.1

# JumpReLU STE bandwidth (Rajamanoharan et al. 2024, paper-specified)
# Controls sharpness of the sigmoid approximation to Heaviside in the backward pass.
# Smaller = sharper (closer to true Heaviside) but noisier gradients.
JUMPRELU_BANDWIDTH = 0.001

# Circuit loss internals
CIRCUIT_TEMPERATURE = 10.0
