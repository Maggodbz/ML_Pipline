

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) tensor_add.cpp -o tensor_add$(python3-config --extension-suffix)
```
