import os
import re
import codecs
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="wen",
    version=find_version("wen", "__init__.py"),
    url="https://github.com/metaescape/Wen",
    author="mE",
    author_email="metaescape@foxmail.com",
    description="中文语言服务",
    packages=find_packages(
        exclude=["user_studies", "notebooks", "client", "imgs"]
    ),
    python_requires=">=3.11",
    install_requires=["pygls>=0.12.2"],
)
