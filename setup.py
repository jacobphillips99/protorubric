from setuptools import find_packages, setup

setup(
    name="open-rubric",
    version="0.1.0",
    description="Open-source tools for autograding LLM answers with rubrics",
    author="Jacob Phillips",
    author_email="jacob.phillips8905@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
