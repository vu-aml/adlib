# data_transform_test.py
# Tests the data transform implementation
# Matthew Sedam

from adlib.adversaries.datatransform.data_transform import DataTransform
from adlib.adversaries.datatransform.poisoning.poison import open_dataset


def test_data_transform():
    print()
    print('###################################################################')
    print('START data transform attack.\n')

    args = {'beta': 0.1,
            'dataset': ('./data_reader/data/raw/data-transform/'
                        'house-processed.csv'),
            'epsilon': 0.001,
            'eta': 0.5,
            'initialization': 'randflip',
            'lambd': 1,
            'logdir': './results',
            'logind': 0,
            'model': 'ridge',
            'multiproc': False,
            'numinit': 1,
            'objective': 1,
            'optimizey': False,
            'partct': 4,
            'poisct': 75,
            'rounding': False,
            'seed': 123,
            'sigma': 1.0,
            'testct': 500,
            'trainct': 300,
            'validct': 250,
            'visualize': False}

    x, y = open_dataset(args['dataset'], args['visualize'])

    attacker = DataTransform(**args)
    poisoned_x, poisoned_y = attacker.attack((x, y))

    print('\nEND data transform attack.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_data_transform()
