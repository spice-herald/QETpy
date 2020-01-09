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
    
    - Go to your GitHub project online and click on the "Pull Request" button. Provide a detailed description of the changes that you've made. If you are addressing an issue see [closing issues](https://help.github.com/en/github/managing-your-work-on-github/closing-issues-using-keywords). For example, if you write: resolves issue #2, when the pull request is merged, issue #2 will be automatically closed. 
    
* Your pull request will then be reviewd by the core development team. The pull request will either be merged to the master, or changes may be requested. 
    
### Style Guidlines

* All code should have tests (see test coverage below for more details).
* All code should be documented.
* The code style and naming conventions should match that of the existing code. We try to adhere to PEP 8 conventions (remove trailing white space, no tabs, etc.). 

If the above criteria are not met, you will be asked to change your code. 


    
