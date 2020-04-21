import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

from get_model_data import get_model_data
from stats import normalize_predictions, RMSE, accuracy


def plot_matrix(mat, output_dir, title):
    print("Plotting %s" % title)

    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.spy(mat, precision=0.1, marker=1, alpha=0.5)
    plt.title(title)
    plt.xlabel('Channel')
    plt.ylabel('Collaborator')
    plt.savefig(output_dir + '/%s.png' % title)
    plt.close()


def long_tail_plot(M, output_dir):
    """
    :param M: adjacency matrix
    :param output_dir: directory to save plot
    plots a line plot of user vs num collaborators, sorted by num collaborators decreasing
    """
    print('Plotting long tail')

    y = sorted(np.sum(M, axis=1), reverse=True)
    plt.plot(y, color='blue')

    total = sum(y)
    running = 0
    head_index = 0
    while running < total / 2:
        running += y[head_index]
        head_index += 1

    plt.fill_between(np.arange(head_index+1), y[:head_index+1], color='lightblue')
    plt.fill_between(np.arange(head_index, len(y)), y[head_index:], color='lightsalmon')

    plt.title('Long Tail Plot (Channel vs Collaborators Count)')
    plt.xlabel('Sorted Channel ID Index')
    plt.ylabel('Number of Collaborators')
    legend_elements = [Patch(facecolor='lightblue', label='Head'),
                       Patch(facecolor='lightsalmon', label='Tail')]

    plt.legend(handles=legend_elements)
    plt.savefig(output_dir + '/long_tail_plot.png')
    plt.close()


def plot_singular_values(D, orig_matrix, output_dir):
    """
    Plots singular values of SVD to determine the optimal truncated dimension via inspection
    """
    print('Plotting singular values of %s' % orig_matrix)

    singular_values = np.diag(D)
    plt.plot(singular_values)
    plt.title('Singular values of %s' % orig_matrix)
    plt.xlabel('Nth Largest Singular Value')
    plt.ylabel('Value')
    plt.savefig(output_dir + '/%s_svs.png' % orig_matrix)
    plt.close()


def plot_accuracy_t_sweep(M, T, T_hat, output_dir):
    print('Plotting t sweep over accuracy...')
    T_hat_norm = normalize_predictions(T_hat)
    t_values = np.linspace(0, 1, 11)
    test_set_accs, zero_set_accs = [], []
    for t in t_values:
        test_set_acc, zero_set_acc = accuracy(M, T, T_hat_norm, t)
        test_set_accs.append(test_set_acc)
        zero_set_accs.append(zero_set_acc)

    plt.plot(t_values, test_set_accs, label='Test Set Accuracy')
    plt.plot(t_values, zero_set_accs, label='Zero Set Accuracy')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.title('t Sweep of Model Accuracy')
    plt.xlabel('Value of t')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(output_dir + '/accuracy.png')


def main():
    output_dir = './plots'

    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_data = get_model_data()
    M = model_data['M']
    T = model_data['T']
    T_hat = model_data['T_hat']
    # M_D = model_data['M_D']
    # T_D = model_data['T_D']
    #
    # long_tail_plot(M, output_dir)
    # plot_singular_values(M_D, 'M', output_dir)
    # plot_singular_values(T_D, 'T', output_dir)
    # plot_matrix(M, output_dir, 'M (sorted)')
    plot_accuracy_t_sweep(M, T, T_hat, output_dir)


if __name__ == '__main__':
    main()