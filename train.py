import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import os 
import numpy as np
import random
import glob 
import models
from PIL import Image
from spectral import remove_conv_spectral_norm

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# set device 
device = models.device

# define maximum noise level
max_noise_level = 75.0/255.0

# define patch_shape
patch_shape = (66,66)
print('patch_shape=', patch_shape)

# existing lists
train_dir = [
    "../data/waterloo", "../data/STI/Additional", "../data/STI/Classic",
    "../data/STI/Fingerprint", "../data/STI/HighResolution",
    "../data/STI/Medical", "../data/STI/OldClassic",
    "../data/STI/Special", "../data/STI/SunAndPlanets",
    "../data/STI/Texture", "../data/cd/cd01A"
]
train_format = [
    ".bmp", ".ppm", ".ppm", ".pgm", ".bmp", ".pgm",
    ".pgm", ".png", ".ppm", ".bmp", ".JPG", ".JPG"
]

# Parse imagine_dataset paths
imagine_root = "../data/imagine_dataset"
for d in sorted(os.listdir(imagine_root)):
    full_path = os.path.join(imagine_root, d)
    if os.path.isdir(full_path):
        train_dir.append(full_path)
        train_format.append(".JPG")
        
test_dir = ["../data/standard_test_images"]
test_format = [".tif"]

val_dir = ['../data/bsds300/images/train','../data/bsds300/images/val','../data/bsds300/images/test', "../data/cd/cd02A"]
val_format = ['.jpg','.jpg','.jpg', ".JPG"]

def load_train_data():
    filenames_train = []

    # Get filenames for training
    for idx,dataset_path in enumerate(train_dir):
        filenames_train.extend(glob.glob(os.path.join(dataset_path,'*'+train_format[idx])))

    # if you want to sort, then you need to cd to the dir first, then cd out
    # filenames_train.sort(key=lambda file: int(file.split('/')[2].split('.')[0]))

    train_images = []

    for filename in filenames_train:
        train_images.append(torch.tensor(np.array(Image.open(filename).convert('L'))).float())

    n_train = len(train_images)
    print('+ n_train (without augmentation)=',n_train)

    # Normalize training images to [0,1] 
    for i,image in enumerate(train_images):
        image=image.float()
        mi=image.min()
        ma=image.max()
        image=(image-mi)/(ma-mi)
        train_images[i]=image

    # Extend trainining split by a factor of augmentation_factor
    #train_images = train_images*augmentation_factor
    #print('+ n_train (with augumentation) = ',len(train_images)) 

    return train_images


def load_test_data():
    filenames_test = []

    # Get filenames for test
    for idx,dataset_path in enumerate(test_dir):
        filenames_test.extend(glob.glob(os.path.join(dataset_path,'*'+test_format[idx])))

        # if you want to sort, then you need to cd to the dir first, then cd out
        # filenames_test.sort(key=lambda file: int(file.split('/')[2].split('.')[0]))

    test_images = []

    for filename in filenames_test:
        test_images.append(torch.tensor(np.array(Image.open(filename).convert('L'))).float())

    # Define split sizes
    n_test = len(test_images)
    print('+ n_test=', n_test)
                
    # Normalize validation images to [0,1] 
    for i,image in enumerate(test_images):
        image = image.float()
        mi = image.min()
        ma = image.max()
        image = (image-mi)/(ma-mi)
        test_images[i] = image

    return test_images


def load_val_data():

    filenames_val = []

    # Get filenames for validation
    for idx,dataset_path in enumerate(val_dir):
        filenames_val.extend(glob.glob(os.path.join(dataset_path,'*'+val_format[idx])))

    # Store validation images
    val_images = []
 
    for filename in filenames_val:
        val_images.append(torch.tensor(np.array(Image.open(filename).convert('L'))).float())
    
    # Define split sizes
    n_val = len(val_images)
    print('+ n_val=',n_val)
                
    # Normalize validation images to [0,1] 
    for i,image in enumerate(val_images):
        image = image.float()
        mi = image.min()
        ma = image.max()
        image = (image-mi)/(ma-mi)
        val_images[i] = image

    # Extend trainining split by a factor of augmentation_factor
    #val_images = val_images*augmentation_factor
    #print('+ n_val  = ',len(val_images))

    return val_images

class GrayscaleRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333)):
        super().__init__(size, scale=scale, ratio=ratio) 

class RandomRot:
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): Grayscale image tensor to be rotated.

        Returns:
            torch.Tensor: Randomly rotated image tensor.
        """
        # Generate a random rotation angle (0, 90, 180, or 270 degrees)
        angle = (torch.rand(1)*360).item()

        # Check if the input tensor is grayscale
        if img.dim() == 2:
            # Add a batch dimension
            img = img.unsqueeze(0)

        # Apply the rotation
        img = transforms.functional.rotate(img, angle)

        # Remove the batch dimension if it was added
        if img.size(0) == 1:
            img = img.squeeze(0)

        return img

class RandomRot90:
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): Grayscale image tensor to be rotated.

        Returns:
            torch.Tensor: Randomly rotated image tensor.
        """
        # Generate a random rotation angle (0, 90, 180, or 270 degrees)
        angle = random.choice([0, 90, 180, 270])

        # Check if the input tensor is grayscale
        if img.dim() == 2:
            # Add a batch dimension
            img = img.unsqueeze(0)

        # Apply the rotation
        img = transforms.functional.rotate(img, angle)

        # Remove the batch dimension if it was added
        if img.size(0) == 1:
            img = img.squeeze(0)

        return img

class NoisyCleanPatchesDataset(Dataset):
    def __init__(self, mode, patch_shape, max_noise_level, 
                 augmentate=True, fixed_noise_level=False,
                 augmentation_factor=12):
        # set mode
        if mode == 'train':
            clean_data=load_train_data()
        elif mode == 'val':
            clean_data=load_val_data()
        elif mode == 'test':
            clean_data=load_test_data()

        
        self.clean_data = clean_data
        self.max_noise_level = max_noise_level
        self.augmentate = augmentate
        self.augmentation_factor=augmentation_factor
        self.patch_shape = patch_shape
        self.fixed_noise_level=fixed_noise_level

        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90()
    ])

    def __len__(self):
        return len(self.clean_data)*self.augmentation_factor

    def __getitem__(self, idx):
        # Apply transforms if any
        idx = idx % len(self.clean_data)
        clean_image = self.clean_data[idx]
                
        # find random positions 
        height, width = clean_image.shape[-2:]
        top = torch.randint(0, height - self.patch_shape[0] + 1, (1,))
        left = torch.randint(0, width - self.patch_shape[1] + 1, (1,))
        
        # sample patch
        clean_patch=transforms.functional.crop(clean_image, 
                                               top, left, 
                                               *self.patch_shape)      
        
        # augmentate clean_path
        if self.augmentate is True:
            clean_patch = self.augmentation_transforms(clean_patch)

        
        # find random noise level 
        if self.fixed_noise_level is False:
            noise_level=torch.rand(1)*self.max_noise_level
        else:
            noise_level=torch.tensor(self.max_noise_level)
        
        # generate noise
        noise=noise_level*torch.randn_like(clean_patch)
        
        # apply noise
        noisy_patch=clean_patch+noise
        
        # # apply noise/2
        # half_noisy_patch=clean_patch+noise/2.0
        
        return noisy_patch.unsqueeze(0), clean_patch.unsqueeze(0), noise_level


# Training/validation routine
if __name__ == "__main__":
    # Where to save the final model
    saved_model_path = "checkpoints/"

    # Create training dataset 
    train_dataset = NoisyCleanPatchesDataset(mode='train',
                                             patch_shape=patch_shape,
                                             max_noise_level=max_noise_level,
                                             augmentate=True,
                                             fixed_noise_level=False)

    # Create validation dataset
    val_dataset = NoisyCleanPatchesDataset(mode='val',
                                           patch_shape=patch_shape,
                                           max_noise_level=max_noise_level,
                                           augmentate=True,
                                           fixed_noise_level=False)

    # Create training dataloader  
    batch_size = 128  
    val_batch_size=128
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)

    # Sampled val_dataset
    val_loader = DataLoader(val_dataset, 
                            batch_size=val_batch_size, 
                            shuffle=True)
    sampled_val = list(val_loader)


    denoiser_depth = 19
    model = models.conicFFDNet(depth=denoiser_depth, 
                               image_channels=1,
                               n_channels=64,
                               bias=False,
                               sn=True,
                               sca=2,
                               patch_shape=patch_shape).to(device)

    # Define loss
    criterion = nn.MSELoss(reduction='sum')

    # Define main hyperparameters
    n_epochs = 1000
    initial_lr = 1e-3
    milestone_3 = 1000
    milestone_2 = 750
    milestone_1 = 500
    milestone_0 = 250

    # Define optimizer
    # It may be useful to use weight_decay=1e-10 or something
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Initial epoch
    start_epoch = 0
    average_losses = []
    average_val_losses = []

    # Check if there are checkpoint files available, if yes, load them
    checkpoint_path_init = 'checkpoints/cnffd_sca2_bs128_p66_lr3_depth19.pth'
    checkpoint_path = 'checkpoints/cnffd_sca2_bs128_p66_lr3_depth19.pth' 
    saved_model_path = 'saved_models/cnffd_sca2_bs128_p66_lr3_depth19.pth'

    if os.path.exists(checkpoint_path_init):
        checkpoint = torch.load(checkpoint_path_init)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'average_losses' in checkpoint:
            average_losses = checkpoint['average_losses']
        if 'average_val_losses' in checkpoint:
            average_val_losses = checkpoint['average_val_losses']

    # Resume training from the last saved epoch
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        
        # Adjust lr
        if epoch > milestone_3: 
            current_lr = initial_lr*1e-4 #0.0001
        elif epoch > milestone_2: 
            current_lr = initial_lr*1e-3 #0.001
        elif epoch > milestone_1:
            current_lr = initial_lr*1e-2 #0.01
        elif epoch > milestone_0:
            current_lr = initial_lr*1e-1 #0.1
        else:
            current_lr = initial_lr
        
        # Set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        
        # This flag tells us wether any spectral normalization was done since the last debug prompt
        normalized=False 

        # Main training loop
        for batch_no, (noisy_patches, clean_patches, noise_level) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # Map images from batch to device
            noisy_patches = noisy_patches.to(device)
            clean_patches = clean_patches.to(device)
            noise_level = noise_level.to(device)

            # Forward pass
            outputs = model(noisy_patches,noise_level)

            # Compute loss and total_loss
            loss = criterion(outputs, clean_patches)/(noisy_patches.shape[0]*2)
            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            # Show batch progress
            if batch_no % 5 == 4:  # Print every 10 batches
                if normalized is True:
                    print(f"Epoch [{epoch+1}/{n_epochs}], Batch [{batch_no+1}/{len(train_loader)}], Loss: {loss.item():.4f}. Normalized.")
                else:
                    print(f"Epoch [{epoch+1}/{n_epochs}], Batch [{batch_no+1}/{len(train_loader)}], Loss: {loss.item():.4f}.")
                normalized = False

        # Validation phase
        val_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for noisy_patch, clean_patch, noise_level in val_loader:
                # Get patches
                noisy_patch = noisy_patch.to(device)
                clean_patch = clean_patch.to(device)
                noise_level = noise_level.to(device)

                # Forward pass
                outputs = model(noisy_patch,noise_level)
                
                # Compute loss and total_loss
                val_loss = criterion(outputs,clean_patch)/(noisy_patch.shape[0]*2)
                val_total_loss += val_loss.item()  # Store validation loss

        # Show epoch progress 
        average_loss = total_loss/len(train_loader)
        average_val_loss = val_total_loss/len(val_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}], Average Loss: {average_loss:.4f}, Average val Loss: {average_val_loss:.4f}")
        
        average_losses.append(average_loss)
        average_val_losses.append(average_val_loss)

        # Save the model and optimizer state after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'average_losses':average_losses,
            'average_val_losses':average_val_losses
        }
        print('Saved checkpoint at ', checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    model.eval()
    print('Training completed.')

    # Remove sn layer 
    for layer in model.conv_layers:
        if isinstance(layer, nn.Conv2d):
            remove_conv_spectral_norm(layer)

    # Save final model 
    torch.save(model, saved_model_path)
    print('Model saved at ', saved_model_path) 
