from setuptools import setup, find_packages


with open('README.md') as f:
	readme = f.read()

with open('LICENSE/license.txt') as f:
	license = f.read()


setup(
    name='rl for set-down',
    version='0.1.0',
    description='reinforcement learning for offshore crane operations',
    long_description=readme,
    author='Mingcheng Ding',
    author_email='dingmingcheng@gmail.com',
    url=None,
    license=license,
    packages=find_packages(exclude=('LICENSE','docs','tests'))
)