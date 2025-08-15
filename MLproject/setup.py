from setuptools import find_packages,setup

def get_requirements(filename):
    """Read requirements from a file and return as a list."""
    with open(filename, 'r') as file:
        requirements = []
        for line in file:
            line = line.strip()
            # Skip empty lines, comments, and editable installs
            if line and not line.startswith('#') and not line.startswith('-e'):
                requirements.append(line)
        return requirements

setup(
    name="MLproject",
    version="0.0.1",
    author="Shivaprasad",
    author_email="shivaprasad01000@example.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)