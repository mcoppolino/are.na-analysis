import numpy as np
from get_model_data import get_model_data


# https://stats.stackexchange.com/questions/28287/evaluating-recommender-systems-with-implicit-binary-ratings-only?fbclid=IwAR0143c6c2pzlLxQD36P2WEYzIR5C2wfWvhKhGd3LmoQagauLdmSw1yzIq0
def RSCORE(M, M_hat, a=0.5, t=0.1):
    """
    :param M, M_hat: results of model :param a: 'patience of user'
    :param t: the minimum threshold of predictions matrix to recommend a channel to a user
    :return: R-Score of model's predictions
    ("how likely it is that a user will view a recommended item, assuming a list of items is recommended to him")


    """
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

        for idx, i in enumerate(sorted_rec_indices[:2]):
            if M_hat[u][i] < t:
                break
            f_u = M[u][i]
            denom = 2 ** (idx / (a-1))
            r_u += f_u / denom

        r_star = sum(1 / (2 ** (i / a - 1)) for i in range(np.sum(M[u])))
        r.append(r_u / r_star)

    r_score = sum(r)
    return r_score


def RMSE(M, T, T_hat, t=0.5):
    """
    Calculates the root mean squared error of the test set, documentation found on page 16 of
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf
    """
    test_set = np.where(M != T)

    r = M[test_set]
    r_hat = np.where(T_hat[test_set] > 0.5, 1, 0)  # cast T_hat to [0,1] by comparing to t

    sum_diff = np.sum(np.sqrt(r - r_hat))
    test_set_size = test_set[0].shape[0]
    rmse = np.sqrt(sum_diff / test_set_size)

    return rmse


def MAE(M, T, T_hat, t=0.5):
    """
    Calculates the mean absolute error of the test set, documentation found on page 16 of
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf
    """
    test_set = np.where(M != T)

    r = M[test_set]
    r_hat = np.where(T_hat[test_set] > 0.5, 1, 0)  # cast T_hat to [0,1] by comparing to t

    sum_diff = np.sum(np.absolute(r - r_hat))
    test_set_size = test_set[0].shape[0]
    mae = np.sqrt(sum_diff / test_set_size)

    return mae


def accuracy(M, T, T_hat, t=0.5):
    """
    :param M, T, T_hat: results of model
    :param t: the minimum threshold of predictions matrix to recommend a channel to a user
    :return: rec_acc, the proportion of correct recommendations in the test set
             non_rec_acc, the proportion of correct non-recommendations in the test set
    """
    test_set = np.where(M != T)
    rec_prediction = T_hat[test_set]
    non_rec_prediction = T_hat[~test_set]

    rec_correct = np.count_nonzero(np.where(rec_prediction >= t))
    non_rec_correct = np.count_nonzero(np.where(non_rec_prediction >= t))

    rec_acc = rec_correct / len(rec_prediction[0])
    non_rec_acc = non_rec_correct / len(non_rec_prediction[0])

    return rec_acc, non_rec_acc


def normalize_predictions(predictions):
    """
    :param predictions: predictions output by model (M_hat, T_hat)
    :return: normalized predictions (divide each row by max)
    """
    return predictions / predictions.sum(axis=1, keepdims=True)


def main():
    # extract data using analysis.get_model_data
    data = get_model_data()
    M = data['M']
    T = data['T']
    M_hat = data['M_hat']
    T_hat = data['T_hat']

    # normalize predictions by max value per user
    M_hat_norm = normalize_predictions(M_hat)
    T_hat_norm = normalize_predictions(T_hat)

    # calculate metrics
    r = RSCORE(M, M_hat)
    rmse = RMSE(M, T, T_hat_norm)
    mae = MAE(M, T, T_hat_norm)
    rec_acc, non_rec_acc = accuracy(M, T, T_hat_norm)

    # print metrics
    metrics = '''
    R-score: %d
    Root Mean Squared Error: %d
    Mean Average Error: %d
    Recommendation Accuracy: %d
    Non-Recommendation Accuracy: %d
    ''' % (r, rmse, mae, rec_acc, non_rec_acc)

    print(metrics)


if __name__ == '__main__':
    main()