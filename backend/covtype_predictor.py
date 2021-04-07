import numpy as np
import math
from model.model import COVModel
from sklearn.preprocessing import MinMaxScaler
from joblib import load


def input_preprocessing(data):
    quant_min = np.array([1859, 0, 0, 0, 0, 0, 0, 0, 0])
    quant_max = np.array([3858, 360, 66, 1418, 7117, 255, 255, 255, 7173])
    wild_min = np.zeros(4)
    wild_max = np.ones(4)
    soil_min = np.zeros(40)
    soil_max = np.ones(40)
    min_arr = np.concatenate([quant_min, wild_min, soil_min]).reshape(1, -1)
    max_arr = np.concatenate([quant_max, wild_max, soil_max]).reshape(1, -1)
    ref = np.concatenate([min_arr, max_arr,data.reshape(1,-1)])
    scaler = MinMaxScaler()
    scaler.fit_transform(ref)
    data = ref[2].reshape(1,-1)
    return data


class COV_predictor:

    def __init__(self):
        self.model = load('/Users/chenyuyan/PycharmProjects/MAIS202_project/venv/backend/model/classifier.joblib')

    def predict(self, request):
        user_inputs = request.form  # get all the inputs
        size = len(user_inputs) - 1  # ignore the first input
        quant_values = np.zeros(size)  # array to store all the quantitative values
        wilderness_type_arr = np.zeros(4)
        soil_type_arr = np.zeros(40)

        # reference: https://stackoverflow.com/questions/17752301/dynamic-form-fields-in-flask-request-form
        i = 0
        ignore = True
        for key in user_inputs.keys():
            if ignore:
                ignore = False
                continue
            quant_values[i] = float(user_inputs[key])
            i += 1

        wilderness_type = quant_values[size - 2]
        soil_type = quant_values[size - 1]
        # wilderness_type starts from 1
        wilderness_type_arr[int(wilderness_type) - 1] = 1
        # soil_type starts from 1
        soil_type_arr[int(soil_type) - 1] = 1
        # remove the wilderness_type and soil_type
        quant_values = quant_values[:size - 2]
        values = np.concatenate([quant_values, wilderness_type_arr, soil_type_arr])
        print(values)
        # standardize the values
        values = input_preprocessing(values)
        cov = {
            1: 'Spruce/Fir',
            2: 'Lodgepole Pine',
            3: 'Ponderosa Pine',
            4: 'Cottonwood/Willow',
            5: 'Aspen',
            6: 'Douglous-fir',
            7: 'Krummholz'
        }

        return cov[self.model.predict(values)[0]]
