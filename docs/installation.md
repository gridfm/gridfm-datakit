# Installation

1. ⭐ Star the [repository](https://github.com/gridfm/gridfm-datakit) on GitHub to support the project!

2. Make sure you have Python 3.10, 3.11, or 3.12 installed. ⚠️ Windows users: Python 3.12 is not supported. Use Python 3.10.11 or 3.11.9.

3. Install gridfm-datakit

```bash
python -m pip install --upgrade pip  # Upgrade pip
pip install gridfm-datakit
```

4. Install Julia with PowerModels and Ipopt
```bash
gridfm_datakit setup_pm
```

### For Developers

To install the latest development version from GitHub, follow these steps instead of step 3.


```bash
git clone https://github.com/gridfm/gridfm-datakit.git
cd "gridfm-datakit"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip  # Upgrade pip to ensure compatibility with pyproject.toml
pip3 install -e '.[test,dev]'
```
