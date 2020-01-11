# Contributing Guidelines

If you are unsure where to start, click on the issues tab to see open [issues](https://github.com/ucbpylegroup/QETpy/issues). 

### Development Process

* For first time contributors
    - Go to https://github.com/ucbpylegroup/QETpy and click the “fork” button to create your own copy of the project.
    - Clone the project to your local computer:
    ```
    git clone https://github.com/your-username/QETpy.git
    ```
* Develop your contribution:
    - Make sure you are up to date
    ```
    git checkout master
    git pull
    ```
    - Create development/feature branch to work on. 
    ```
    git checkout -b new-feature-name
    ```
    
    - Make sure to locally commit your progress regularly, 
    ```
    git add new-file-you-are-changing
    git commit -m 'short descriptive message about what you changed'
    ```
    
* Submit your changes
    - Push to your fork on GitHub
    
    ```
    git push origin new-feature-name
    ```
    
    - Go to your GitHub project online and click on the "Pull Request" button. Provide a detailed description of the changes that you've made. If you are addressing an issue, see [closing issues](https://help.github.com/en/github/managing-your-work-on-github/closing-issues-using-keywords). For example, if you write: "Resolves issue #2", when the pull request is merged, issue #2 will be automatically closed. 
    
* Your pull request will then be reviewed by the core development team. The pull request will either be merged to the master, or changes may be requested. 
    
### Style Guidelines

* All code should have tests.
* All code should be documented.
* The code style and naming conventions should match that of the existing code. We try to adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions (remove trailing white space, no tabs, etc.). 

__If the above criteria are not met, you will be asked to change your code.__ 

### Testing

All code that you are commiting should have an associated test. Ideally your test should cover all the functionality of the code, but this is easier said than done. If you are usure about how to write a test, look at the tests in the test folder of this repository, and search online about how to write good unit tests for python. Once your code is written, you should test your tests locally to make sure that 1) they work, and 2) the code coverage of the package either increases or stays the same. To do this you will need to install the following:

```
pip install pytest
pip install pytest-cov
```

Then to run the tests, from the base directory of the repository,

```
py.test --cov
```




    
