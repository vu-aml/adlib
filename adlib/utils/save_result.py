#save experimental result to data type that is easier for future usage
#pandas Dataframe -> csv file

import pandas as pd

def save_result(result_arr, column_name):
    """
    result -> pandas Dataframe structure
    :param result_arr: np.array type
    :param column_name: list
    :return: csv file
    """
    df = pd.DataFrame(result_arr, column_name)
    