from data_path import DATA_PATH
import torch
from torch.utils.data import DataLoader
from data_loading import PalpaitineDataset, MemoryFriendlyPalpaitineDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import json

# This is a basic script to compare the outputs of the PalpaitineDataset and 
# MemoryFriendlyPalpaitineDataset. The script will plot the outputs, only works
# for batch_size=1.

device = torch.device("cpu")

#train_data = PalpaitineDataset(
    #data_path=f"{DATA_PATH}/training",
    #augment=False, use_all_data=True, experimental_data=True
#)

mean_musp = 0 #(np.mean(train_data.scatterings.detach().cpu().numpy()))
std_musp = 1 #(np.std(train_data.scatterings.detach().cpu().numpy()))
mean_mua = 0 #(np.mean(train_data.absorptions.detach().cpu().numpy()))
std_mua = 1 #(np.std(train_data.absorptions.detach().cpu().numpy()))
mean_signal = 0 #(np.mean(train_data.images.detach().cpu().numpy()))
std_signal = 1 #(np.std(train_data.images.detach().cpu().numpy()))
mean_fluence = 0 #(np.mean(train_data.fluences.detach().cpu().numpy()))
std_fluence = 1 #(np.std(train_data.fluences.detach().cpu().numpy()))

val_data = PalpaitineDataset(
    data_path=f"{DATA_PATH}/test",
    use_all_data=True, train=False, device=device, fold=0,
    mean_musp=mean_musp, std_musp=std_musp, 
    mean_mua=mean_mua, std_mua=std_mua,
    mean_signal=mean_signal, std_signal=std_signal,
    mean_fluence=mean_fluence, std_fluence=std_fluence,
    experimental_data=True
)
val_loader = DataLoader(dataset=val_data, batch_size=1)

with open(f"dataset_stats.json") as f:
    stats = json.load(f)
memory_friendly_val_data = MemoryFriendlyPalpaitineDataset(
    data_path=f"{DATA_PATH}/test", stats=stats, transform='standardise',
    fold=0, train=False, device=device, augment=False,
    use_all_data=True, experimental_data=True
)
memory_friendly_val_loader = DataLoader(
    dataset=memory_friendly_val_data, batch_size=1
)
# turn dataloaders into iterators
val_loader = iter(val_loader)
memory_friendly_val_loader = iter(memory_friendly_val_loader)

labels = ["inputs", "segmentation", "mua", "musp", "fluence", "instance_seg"]
for (sample, dataset_name) in ((next(val_loader), 'PalpaitineDataset'), 
    (next(memory_friendly_val_loader), 'MemoryFriendlyPalpaitineDataset')):
    
    
    n_frames = 0
    frames = []
    frame_labels = []
    print(f'Using {dataset_name}')
    for i in range(len(labels)):
        print(f'type(sample[{i}]): {type(sample[i])}')
        match type(sample[i]):
            case torch.Tensor:
                print(f'sample[{i}].shape: {sample[i].shape}')
                sample[i] = sample[i].detach().cpu().numpy()
            case np.ndarray:
                print(f'sample[{i}].shape: {sample[i].shape}')
            case list:
                print(f'len(sample[{i}]): {len(sample[i])}')
                sample[i] = np.asarray(sample[i])
            
        sample[i] = np.squeeze(sample[i])
        if len(sample[i].shape) == 3:
            for j, instance in enumerate(sample[i]):
                frame_labels.append(labels[i]+"_"+str(j))
                frames.append(instance)
            n_frames += sample[i].shape[0]
        else:
            frame_labels.append(labels[i])
            frames.append(sample[i])
            n_frames += 1
            
    nrows = 1 + (n_frames // 4)
    print(f'nrows: {nrows}')
    print(f'n_frames: {n_frames}')
    fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(20, 5*nrows))
    fig.suptitle(dataset_name)
    for i in range(n_frames):
        row = i // 4
        col = i % 4
        print(f'row: {row}, col: {col}')
        img = ax[row, col].imshow(frames[i], cmap='gray')
        ax[row, col].set_title(frame_labels[i])
        divider = make_axes_locatable(ax[row, col])
        cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(img, cax=cbar_ax, orientation='vertical')
    fig.tight_layout()
    fig.savefig(f'{dataset_name}.png')
        

        