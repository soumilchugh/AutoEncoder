# AutoEncoder implementation
# Skip connections between downsampling and upsampling blocks helps in graident flow
# Latent Vector Size is 300
# Loss - Mean Square Error
# Optimizer - Adam
# Can be used for denoising and compression
# Layer Class contains the Convolution and Transposed Convolution Implementation
# Data Class loads the data
# Model class sets the loss and the optimizer
# Train class runs the training and validation 
