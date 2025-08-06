from setuptools import setup
from setuptools.command.install import install
import subprocess

class InstallLocalPackage(install):
    def run(self):
        install.run(self)

        print("Running installation of external dependency...")
        subprocess.call(
            "python3 external/setup.py install", shell=True
        )

setup(
    cmdclass={'install': InstallLocalPackage}
)