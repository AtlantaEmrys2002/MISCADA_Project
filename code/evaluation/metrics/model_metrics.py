import matplotlib.pyplot as plt

def plot_losses(model, losses):

    epochs = losses[0]
    loss = losses[1]

    plt.plot(epochs, losses)