import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_metrics(loss, acc):
    fig, ax1 = plt.subplots()
    x = np.arange(1, 41, 1)
    y1 = loss
    ax1.plot(x, y1, 'b-')
    ax1.set_xlabel('Epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    y2 = acc
    ax2.plot(x, y2, 'r-')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    fig.savefig('../../dsi-capstone-data/test/test6/loss_vs_acc.png')
    plt.show()
    pass

def main():
    # load the pickled history files
    history = pickle.load(open("../../dsi-capstone-data/test/test6/model_A_history.pkl", "rb" ))
    val_acc = ['%.2f' % elem for elem in history.get('val_acc')]
    val_loss = ['%.2f' % elem for elem in history.get('val_loss')]
    train_acc = ['%.2f' % elem for elem in history.get('acc')]
    train_loss = ['%.2f' % elem for elem in history.get('loss')]
    plot_metrics(val_loss, val_acc)
    pass


if __name__ == '__main__':
    main()
