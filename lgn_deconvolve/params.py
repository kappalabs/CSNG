import torch


# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 12

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
device = torch.device("cpu")
print('device', device)

# Batch size during training
# batch_size = 1024*2  # max on GPU
# batch_size = 40000
# batch_size = 35000
batch_size = 10000
# batch_size = 1024 * 4
# batch_size = 1024
# batch_size = 512 * 4

# Learning rate for optimizers
# lr = 1
# lr = 0.6
lr = 0.1
# lr = 0.01
# lr = 0.001
# lr = 0.0001
# lr = 0.00001
# lr = 0.000001
# lr = 0.0000001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta1 = 0.9

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
# image_size = 110

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 447

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5000