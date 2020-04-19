import numpy as np
from get_model_data import get_model_data

# R_u = sum over channels i (M[u][i] / 2 ** (i - 1 / a - 1)) where M[u] is sorted by M_hat[u] decreasing
def r_score(M, M_hat, a=0.5):
    # for u in range(M_hat.shape[0]):
    #     R_u = 0
    #     R_star = sum(1/(math.pow(2, (k/a-1))) for k in range(T[u].count(1)))
    #     for i in range(M_hat.size[1]):
    #         if M_hat[u][i] >= 0.1 and T[u][i] == 1:
    #             R_u = R_u + (M_hat[u][i])/math.pow(2, (i)/(a-1))
    #     R = R + R_u/R_star
    #
    # return R

    r = []
    for u in range(M.shape[0]):
        r_u = 0

        # indices that would sort M_hat[u] from high to low
        sorted_rec_indices = np.flip(np.argsort(M_hat[u]))

        for i in sorted_rec_indices:
            f_u = M[u][i]
            denom = 2 ** (i / (a-1))
            r_u += f_u / denom

        r.append(r_u)

    r_star = 10 ** 61  # TODO calculate R_star (might be this value? just a guess)

    r_score = sum(r_u / r_star for r_u in r)
    return r_score

def RMSE(M, T, T_hat):
    """
    Calculates the root mean squared error of the test set, documentation found on page 16 of
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf
    """
    test_set = np.where(M != T)

    r = M[test_set]
    r_hat = T_hat[test_set]

    sum_diff = np.sum(np.sqrt(r - r_hat))
    test_set_size = test_set[0].shape[0]
    rmse = np.sqrt(sum_diff / test_set_size)

    return rmse


def MAE(M, T, T_hat):
    """
    Calculates the mean absolute error of the test set, documentation found on page 16 of
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf
    """
    test_set = np.where(M != T)

    r = M[test_set]
    r_hat = T_hat[test_set]

    sum_diff = np.sum(np.absolute(r - r_hat))
    test_set_size = test_set[0].shape[0]
    print(test_set_size)
    mae = np.sqrt(sum_diff / test_set_size)

    return mae


def main():
    data = get_model_data()
    M = data['M']
    T = data['T']
    M_hat = data['M_hat']
    T_hat = data['T_hat']

    r = r_score(M, M_hat)
    rmse = RMSE(M, T, T_hat)
    mae = MAE(M, T, T_hat)
    print(r, mae, rmse)

if __name__ == '__main__':
    main()