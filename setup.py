from setuptools import setup

if __name__ == '__main__':
    setup(
        name="numpytorch",
        version='0.1.0',
        description="Monkey-patched numpy with pytorch syntax",
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ashawkey/numpytorch',
        author='kiui',
        author_email='ashawkey1999@gmail.com',
        packages=['numpytorch',],
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3 ',
        ],
        keywords='deep learning, tensor manipulation, machine learning, scientific computations',
        install_requires=[
            'numpy>=1.20',
            'forbiddenfruit',
        ],
    )