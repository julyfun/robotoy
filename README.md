```console
python setup.py build --ex \
itp_state \
ring_buffer \
&& pip install -e .
```

**upload:**

```console
python setup.py sdist bdist_wheel --ex all
twine upload dist/* --verbose
# twine upload --repository testpypi dist/* --verbose
```

## Dep

```
sudo apt -y install python3-pybind11
uv pip install pip
```

