# data_transform_test.py
# Tests the data transform implementation
# Matthew Sedam

from adlib.adversaries.datatransform.data_transform import DataTransform


def test_data_transform():
    print()
    print('###################################################################')
    print('START data transform attack.\n')

    attacker = DataTransform()
    attacker.attack(None)

    print('\nEND data transform attack.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_data_transform()
