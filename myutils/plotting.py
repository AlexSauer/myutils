import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


def img_plot(img: np.array, 
             batch_element: int = 0,
             ignore_RGB: bool = False) -> None:
    """
    Simple interface to plt.imshow in order to plot an example 
    from a batch of images
    The input img can also be a pytorch Tensor.
    Accepted shapes for img are:
    - [B, C , H, W]
    - [B, H, W]
    - [C, H, W] (we different this from the current case by B > 3)
    """

    # Transform a possbile torch.Tensor to np.array
    if type(img).__name__ == 'Tensor':
        img = img.cpu().numpy()
    assert isinstance(img, np.ndarray), f'Unknown type for img: {type(img)}'
    
    # Cases: [B, C, H, W] 
    if len(img.shape) == 4:
        img = img[batch_element]
        assert len(img.shape) <= 3, f'Input has unknown shape!'
    # Case: [B, H, W]
    elif len(img.shape) == 3 and min(img.shape) > 3:
        img = img[batch_element]

    # Now the shape is either [C, H, W] or just [H, W]
    if not ignore_RGB and len(img.shape)==3:
        # Find RGB channel and reshape as plt.imshow needs RGB as last dim
        rgb_dim = np.where(np.array(img.shape) == 3)[0]
        if len(rgb_dim) and rgb_dim[0] == 0:
            img = rearrange(img, 'c w h -> w h c')

        # Channel could also be a 1 (as we need the channel for deep learning)
        img = img.squeeze()

    plt.imshow(img)
    plt.show()
