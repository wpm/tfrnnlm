import tfrnnlm
from setuptools import setup

setup(
    name="tfrnnlm",
    version=tfrnnlm.__version__,
    packages=["tfrnnlm"],
    url="https://github.com/wpm/tfrnnlm",
    license="MIT",
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Example TensorFlow RNN Language Model",
    entry_points={
        "console_scripts": ["tfrnnlm=tfrnnlm.main:main"],
    },
    install_requires=["tensorflow"]
)
