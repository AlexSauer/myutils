import matplotlib as plt
import numpy as np
from typing import Union, List

def img_plot(img: Union[List[np.ndarray], np.ndarray], 
         slice: int = None) -> None:

    if isinstance(img, list):
        n_images = len(list)
        fig, ax = plt.subplots(1, n_images)
        ax = ax.ravel()
        for i in range(n_images):
            ax[i].plot(img[i])

    elif isinstance(img, np.ndarray):
        plt.plot(img)

    else:
        raise ValueError(f'Unknown type for img: {type(img)}')


