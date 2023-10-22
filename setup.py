from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='gms',
  version='0.2.0',
  author='plugg1N (Nikita Zhamkov Dmitrievich)',
  author_email='nikitazhamkov@gmail.com',
  description='General Model Selection Module',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/plugg1N/gms-module',
  packages=find_packages(),
  install_requires=['requests>=2.25.1'],
  classifiers=[
    'Programming Language :: Python :: 3.10',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='python machine-learning ml models ai',
  project_urls={
    'Documentation': 'https://github.com/plugg1N/gms-module/blob/main/README.md',
    'Project_github': 'https://github.com/plugg1N/gms-module'
  },
  python_requires='>=3.7'
)