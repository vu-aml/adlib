from setuptools import find_packages, setup

setup(
    name='pyamll',
    version='0.0.1',
    description="Game-theoretic adversarial machine learning library providing a set of learner and adversary modules.",
    url='https://github.com/yvorobey/aml',
    license='',
    author='Alexandra, Alexander deGroot, Ethan Raymond, Jiazhi Zhang, Bumsu Jung, Yevgeniy Vorobeychik',
    author_email='yevgeniy.vorobeychik@vanderbilt.edu',
    platforms=['any'],
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
        'scipy>=0.15',
        'sklearn>=0.18',
        'matplotlib>=2.0.0',
    ]
)
