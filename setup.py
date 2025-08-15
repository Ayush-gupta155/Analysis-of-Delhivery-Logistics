from setuptools import find_packages, setup
from typing import List

var = '-e .'

def get_requirements(file_path:str) ->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','')for req in requirements]

        if var in requirements:
            requirements.remove(var)


    return requirements

setup(
    name='ETA Analysis of Delivery Logistics Company',
    version='0.0.1',
    author='Ayush Gupta',   
    author_email='agayushgupta12@gmail.com',
    packages=find_packages(),
    install_requirements = get_requirements('requirements.txt')
    

)
