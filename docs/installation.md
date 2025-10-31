# Installation

1. ⭐ Star the [repository](https://github.com/gridfm/gridfm-datakit) on GitHub to support the project!

2. Install gridfm-datakit

```bash
python -m pip install --upgrade pip  # Upgrade pip
pip install gridfm-datakit
```

3. Install Julia with Powermodels and Ipopt
```bash
gridfm_datakit setup_pm
```

### For Developers

To install the latest development version from github, simply follow these steps instead of 2.


```bash
git clone https://github.com/gridfm/gridfm-datakit.git
cd "gridfm-datakit"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip  # Upgrade pip to ensure compatibility with pyproject.toml
pip3 install -e '.[test,dev]'
```
