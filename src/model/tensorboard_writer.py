import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    Args:
    -----
    logdir: string
        path where the tensorboard logs will be stored
    """

    def __init__(self, logdir):
        """ Initializing tensorboard writer """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        """ Adding a scalar for plot """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure, step):
        """ Adding a whole new figure to the tensorboard """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return

    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return

    def image_grid_2D(self, images, n_rows, n_cols):
        figure = plt.figure(figsize=(12, 8))
        for i in range(n_rows * n_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            img = images[i].permute(1, 2, 0).detach().cpu()
            # plt.imshow(img, cmap="gray", vmin=0, vmax=1)
            plt.imshow(img, cmap="gray")

        return figure

    def image_grid_1D(self, profiles, n_rows, n_cols):
        figure = plt.figure(figsize=(12, 8))
        for i in range(n_rows * n_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            prof = profiles[i].detach().cpu()
            plt.plot(prof)

        return figure

    def add_grid(self, images, n_rows, n_cols, step, tag):
        """
        Add grid of images, either 4d or 5d
        """
        if len(images.shape) == 5:
            num_im, b, c, w, h = images.shape
            images = torch.reshape(images, (num_im*b, c, w, h))
        elif len(images.shape) == 4:
            b, c, w, h = images.shape
        figure, ax = plt.subplots(n_rows, n_cols)
        figure.set_size_inches(3*n_cols, 3*n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                a = ax[j] if n_rows==1 else ax[i, j]
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                img = images[i*j+j].permute(1, 2, 0) # .detach().cpu()
                # a.imshow(img, cmap="gray", vmin=0, vmax=1)
                a.imshow(img, cmap="gray")
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)

    
    
    def add_grid_rdmaps(self, images, n_rows, n_cols, step, tag):
        assert isinstance(images, torch.Tensor), f"images must be a torch.Tensor, but got {type(images)}"
        if len(images.shape) == 5:
            num_im, b, c, w, h = images.shape
            images = torch.reshape(images, (num_im*b, c, w, h))
        
        assert len(images.shape) == 4, "The images should have shape (B, C, W, H)"

        # Create a 2D grid of titles like the rdmaps
        titles_grid = []
        for i in range(n_rows):
            for j in range(n_cols):
                title = "Original" if (i * n_cols + j) % 2 == 0 else "Reconstructed"
                titles_grid.append(title)

        figure, ax = plt.subplots(n_rows, n_cols)
        figure.set_size_inches(5*n_cols, 3*n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                a = ax[j] if n_rows==1 else ax[i, j]
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                idx = i * n_cols + j
                img = images[idx].permute(1, 2, 0)  # (H, W, C)
                # 1 channel
                mesh = a.pcolor(img[:, :, 0])
                a.set_title(titles_grid[idx])
                figure.colorbar(mesh, ax=a)
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return 