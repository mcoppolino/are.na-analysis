import numpy as np


def get_model_data(path):
    with np.load(path + "/svd.npz") as data:
        return_dict = {}
        types_of_matrcies = ['M', 'T', 'M_U', 'M_D', 'M_V', 'M_U_trunc', 'M_D_trunc', 'M_V_trunc', 'T_U', 'T_D', 'T_V', 'T_U_trunc', 'T_D_trunc', 'T_V_trunc', 'M_hat', 'T_hat']
        for t in types_of_matrcies:
            return_dict[t] = data[t]
            print(t + ":")
            print(data[t])

        return return_dict