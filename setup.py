from setuptools import find_packages, setup

setup(
    name = "mcqgenerator",
    version = "0.0.1",
    author = "Manav Baldewa",
    author_email = "manavbaldewa2001@gmail.com",
    install_requires = ["ai21", "langchain", "stramlit", "python-dotenv", "pyPDF2"],
    pachages = find_packages()
)