from setuptools import find_packages, setup

setup(
    name='adlib',
    version='0.1.0',
    description='Game-theoretic adversarial machine learning library providing '
                'a set of learner and adversary modules.',
    url='https://github.com/vu-aml/adlib',
    license='MIT',
    author='Alexandra, Alexander deGroot, Ethan Raymond, Jiazhi Zhang, '
           'Bumsu Jung, Yevgeniy Vorobeychik, Matthew Sedam',
    author_email='yevgeniy.vorobeychik@vanderbilt.edu',
    platforms=['any'],
    packages=find_packages(),
    install_requires=['pyyaml'],
)
