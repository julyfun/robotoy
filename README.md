```console
python setup.py build --ex \
itp_state \
ring_buffer \
&& pip install -e .
```

```console
python setup.py sdist bdist_wheel --ex all
twine upload --repository testpypi dist/* --verbose
```
