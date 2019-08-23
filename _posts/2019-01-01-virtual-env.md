---
layout: post
title: "Python Virtual Environment"
subtitle: "Creating and using a virtual environment for Python"
tags: [Virtual environment, pip]
comments: true
---

When you start a new Python project, the chances are, you freshly install all the required packages and start working. When you finish your work, the project is all good and behaves as expected. That is because it satisfies all the "dependencies" among packages. In time, you keep working on other projects, sometimes reinstalling or updating some packages along the way. Now, if you go back to your first project and run it, you may encounter some problems. That is because the dependencies may not be satisfied anymore due to conflicting versions.

One remedy is that you can work in a separate Python environment for each of your projects. Whatever changes you make in an environment does not affect the others. Moreover, you can keep track of the versions of the packages that are "required" for your project. Later, if you need to transfer your project to another computer or share it with someone else, the exact environment can be recreated using that tracking file. Let's see how it is done.

---

### Create a virtual environment called ***venv*** using `virtualenv` module
```python
python3 -m venv venv  # on macOS

py -m venv venv  # on Windows
```

### Activate the virtual environment
```python
source venv/bin/activate  # on macOS

...\venv\Scripts\activate  # on Windows
```

### See the installed packages
```python
pip list
```

### Install a new package, e.g. numpy
```python
pip install numpy
```

### Deactivate the virtual environment
```python
deactivate
```

### Delete the virtual environment
```python
rm -rf venv/
```

### How to create a requirements.txt file?
```python
pip freeze > requirements.txt
```

### How to use requirements.txt file after creating a new environment?
```python
pip install -r requirements.txt
```

### How to make system packages available when creating a new environment?
```python
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

### How to see the local packages that we installed? (Not the system packages we brought)
```python
pip list --local
```
