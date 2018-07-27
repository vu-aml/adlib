from setuptools import find_packages, setup


def get_requirements():
    requirements = []
    with open('requirements.txt') as file:
        for line in file:
            requirements.append(line[:-1])
    return requirements


setup(
    name='adlib',
    version='1.2.1',
    description='Game-theoretic adversarial machine learning library providing '
                'a set of learner and adversary modules.',
    url='https://github.com/vu-aml/adlib',
    license='MIT',
    author='Alexandra, Alexander deGroot, Ethan Raymond, Jiazhi Zhang, '
           'Bumsu Jung, Yevgeniy Vorobeychik, Matthew Sedam',
    author_email='yevgeniy.vorobeychik@vanderbilt.edu',
    platforms=['any'],
    packages=find_packages(),
    install_requires=get_requirements(),
)
