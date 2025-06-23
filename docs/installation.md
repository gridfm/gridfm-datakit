# Installation

1. ⭐ Star the [repository](https://github.com/Grid-FM/grid_data_synthetic) on GitHub to support the project!

2. Clone the repository and set up a Python virtual environment:

```bash
git clone git@github.com:Grid-FM/grid_data_synthetic.git
cd grid_data_synthetic
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip  # Upgrade pip to ensure compatibility with pyproject.toml
pip3 install .
```

### For Developers

Install with development and testing dependencies:

```bash
pip3 install -e '.[test,dev]'
```
