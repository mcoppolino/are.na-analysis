import matplotlib.pyplot as plt
import numpy as np
import os
from preprocess import get_model_data

def long_tail_plot(M, output_dir='./plots'):
    """
    :param M: adjacency matrix

    plots a line plot of user vs num collaborators, sorted by num collaborators decreasing
    """
    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y = sorted(np.sum(M, axis=1), reverse=True)
    plt.plot(y)
    plt.title('Long Tail Plot (channel vs num collaborators)')
    plt.xlabel('Sorted Channel ID Index')
    plt.ylabel('Number of Collaborators')
    plt.savefig(output_dir + '/long_tail_plot.png')


def main():
    model_data = get_model_data()
    long_tail_plot(model_data['M'])


if __name__ == '__main__':
    main()