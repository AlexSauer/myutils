import torch
import matplotlib.pyplot as plt

def plt_batch(batch: torch.Tensor, ind: int = 0) -> None:
    """
    Plots one slice of a batch of images
    batch: Tensor of shape [B, C, X, Y]
    """
    assert len(batch.shape) == 4, f'Dimension of batch is {batch.shape}' \
                                  f'Try plt_rgb or plt_batch'
    img = batch[ind].squeeze()

    # Check if we are dealing with an RGB image (i.e. 3 channels)
    channel_dim = np.where(torch.tensor(img.shape) == 3)[0][0]
    axes = (1,2,0) if channel_dim == 0 else (0,1,2)

    plt.imshow(img.cpu().numpy().transpose(axes))
    plt.show()



def plt_batch3d(batch: torch.Tensor, ind: int = 0, slice: int = 0) -> None:
    """
    Plots one slice of a 3D batch
    batch: Tensor of shape [B, C, X, Y, Z]
    """
    assert len(batch.shape) == 5, f'Dimension of batch is {batch.shape}' \
                                  f'Try plt_rgb or plt_batch'
    img = batch[ind].squeeze()
    if img.dim() == 4:
        img = img[:, slice]

    plt.imshow(img)
    plt.show()


