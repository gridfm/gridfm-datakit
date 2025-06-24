# Installation

1. ⭐ Star the [repository](https://github.com/gridfm/gridfm-datakit) on GitHub to support the project!

2. Run:

    ```bash
    python -m pip install --upgrade pip  # Upgrade pip
    pip install gridfm-datakit
    ```


### For Developers

To install the latest development version from github, simply follow these steps:


```bash
git clone https://github.com/gridfm/gridfm-datakit.git
cd "gridfm-datakit"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip  # Upgrade pip to ensure compatibility with pyproject.toml
pip3 install -e '.[test,dev]'
```
