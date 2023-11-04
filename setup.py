import setuptools


setuptools.setup(
    name="Wen",
    version="0.1",
    author="mE",
    url="https://github.com/metaescape/Wen",
    author_email="metaescape@foxmail.com",
    description="中文语言服务",
    long_description="",
    license="Apache 2.0",
    packages=["wen"],
    python_requires=">=3.11",
    install_requires=["pygls>=0.12.2"],
)
