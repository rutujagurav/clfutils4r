Package created following this official python [documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

Delete any old versions from `dist/` and then run the following commands:

1. `python3 -m pip install --upgrade build`
2. `python3 -m build`
3. `python3 -m pip install --upgrade twine`
4. `python3 -m twine upload --repository pypi dist/*`
    
    You will be prompted for a username and password. For the username, use __token__. For the password, use the token value, including the pypi- prefix.