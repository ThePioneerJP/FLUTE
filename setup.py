from setuptools import setup, find_packages

setup(
    name='FLUTE',
    version='0.1.0',
    description='A package for prompt processing and language model interaction',
    author='The Pioneer',
    url='https://github.com/thepioneerjp/FLUTE',
    packages=find_packages(),
    install_requires=[
        'openai',
        'anthropic',
        'google-generativeai',
        'python-dotenv',
        'pytest'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
)