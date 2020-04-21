import numpy as np
from get_model_data import get_model_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


# https://stats.stackexchange.com/questions/28287/evaluating-recommender-systems-with-implicit-binary-ratings-only?fbclid=IwAR0143c6c2pzlLxQD36P2WEYzIR5C2wfWvhKhGd3LmoQagauLdmSw1yzIq0
def RSCORE(M, M_hat, a=2, t=0.5):
    """
    :param M, M_hat: results of model :param a: 'patience of user'
    :param t: the minimum threshold of predictions matrix to recommend a channel to a user
    :return: R-Score of model's predictions
    ("how likely it is that a user will view a recommended item, assuming a list of items is recommended to him")
    """
    print('Calculating r-score...')
    r = []
    for u in range(M.shape[0]):
        r_u = 0

        # indices that would sort M_hat[u] from high to low
        sorted_rec_indices = np.flip(np.argsort(M_hat[u]))

        for idx, i in enumerate(sorted_rec_indices):
            if M_hat[u][i] < t:
                break
            f_u = M[u][i]
            denom = 2 ** (idx / (a-1))
            r_u += f_u / denom

        r_star = sum(1 / (2 ** (i / a - 1)) for i in range(int(np.sum(M[u]))))
        r.append(r_u / r_star)

    r_score = sum(r)
    return r_score


def RMSE(M, T, T_hat, t=0.5):
    """
    Calculates the root mean squared error of the test set, documentation found on page 16 of
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf
    """
    print('Calculating root mean squared error...')
    test_set = np.where(M != T)
    y_pred = np.where(T_hat[test_set] > t, 1, 0)
    y_true = M[test_set]

    return mean_squared_error(y_true, y_pred)


def accuracy(M, T, T_hat, t=0.5):
    """
    :param M, T, T_hat: results of model
    :param t: the minimum threshold of predictions matrix to recommend a channel to a user
    :return: rec_acc, the proportion of correct recommendations in the test set
    """
    print('Calculate testing accuracy (t=%f)...' % t)

    test_set = np.where(M != T)
    y_pred = np.where(T_hat[test_set] > t, 1, 0)
    y_true = M[test_set]
    test_set_acc = accuracy_score(y_true, y_pred)

    zero_set = np.where(M != 0)
    y_pred = np.where(T_hat[zero_set] > t, 1, 0)
    y_true = M[zero_set]
    zero_set_acc = accuracy_score(y_true, y_pred)

    return test_set_acc, zero_set_acc


def normalize_predictions(predictions):
    """
    :param predictions: predictions output by model (M_hat, T_hat)
    :return: normalized predictions (divide each row by max)
    """
    print('Normalizing predictions...')
    return predictions / predictions.max(axis=1, keepdims=True)


def main():
    # extract data using analysis.get_model_data
    data = get_model_data()
    M = data['M']
    T = data['T']
    M_hat = data['M_hat']
    T_hat = data['T_hat']

    # normalize predictions by max value per user
    T_hat_norm = normalize_predictions(T_hat)

    # calculate metrics
    rec_acc = accuracy(M, T, T_hat_norm)
    rmse = RMSE(M, T, T_hat_norm)
    r = RSCORE(M, M_hat)

    metrics = '''
        Recommendation Accuracy: %f
        Root Mean Squared Error: %f
        R-score: %f
    ''' % (rec_acc, rmse, r)

    print(metrics)


if __name__ == '__main__':
    main()