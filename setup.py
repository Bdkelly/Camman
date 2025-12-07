from setuptools import setup, find_packages

setup(
    name="ball_tracking_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "PyQt5",
        "albumentations",
        "torch",
        "torchvision",
        "opencv-python",
        "pyserial",
        "bleak"
    ],
    entry_points={
        'console_scripts': [
            'camman=guiapp.camman:main',
        ],
    },
)