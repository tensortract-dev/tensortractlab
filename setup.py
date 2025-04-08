from setuptools import setup

install_requires = [
    'streamlit==1.44.1',
    'streamlit-extras==0.6.0',
    'tensortract2==0.0.1',
    'vocaltractlab==0.5.6',
    ]

setup(
    name='tensortractlab',
    version='0.0.0',
    description='A PyTorch implementation of TensorTractLab',
    author='Paul Krug',
    url='https://github.com/Altavo/tensortractlab',
    license='GPLv3',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['tensortractlab'],
    install_requires=install_requires,
)