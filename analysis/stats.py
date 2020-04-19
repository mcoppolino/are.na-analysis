import math
from SVDModel import SVDModel


def get_score(M_hat, T, a):
    R = 0
    M_hat = list(M_hat)
    T = list(T)
    for u in range(M_hat.size[0]):
        R_u = 0
        R_star = sum(1/(math.pow(2, (k/a-1))) for k in range(T[u].count(1)))
        for i in range(M_hat.size[1]):
            if M_hat[u][i] >= 0.1 and T[u][i] == 1:
                R_u = R_u + (M_hat[u][i])/math.pow(2, (i)/(a-1))
        R = R + R_u/R_star

    return R


def main():
    model = SVDModel(load_dir='../data/model')
    r = get_score(model.M_hat, model.T, 1)
    print(r)

if __name__ == '__main__':
    main()