# Change to the ml_pipeline directory
cd ./ml_pipeline

# Remove and create the build directory
rm -r ./build
mkdir ./build
cd ./build

# Run cmake and make to compile the C++ code
cmake ..
make

# Add build directory to PYTHONPATH to ensure Python can find the compiled module
export PYTHONPATH=$(pwd):$PYTHONPATH

# Auto-generate .pyi stub file using pybind11-stubgen
pybind11-stubgen my_pytorch --output-dir ./