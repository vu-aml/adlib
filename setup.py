from setuptools import find_packages, setup

setup(
    name='adlib',
    version='0.0.1',
    description="Game-theoretic adversarial machine learning library providing a set of learner and adversary modules.",
    url='https://github.com/vu-aml/adlib',
    license='',
    author='Alexandra, Alexander deGroot, Ethan Raymond, Jiazhi Zhang, Bumsu Jung, Yevgeniy Vorobeychik',
    author_email='yevgeniy.vorobeychik@vanderbilt.edu',
    platforms=['any'],
    packages=find_packages(),
    install_requires=['pyyaml'],
)
