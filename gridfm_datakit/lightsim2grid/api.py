try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from lightsim2grid import network as lightsim2grid_network

    LIGHTSIM2GRID_AVAILABLE = hasattr(lightsim2grid_network, "init_from_matpower")
except ImportError:
    LIGHTSIM2GRID_AVAILABLE = False
    lightsim2grid_network = None


def is_lightsim2grid_available() -> bool:
    """Check if a lightsim2grid version able to read MATPOWER data is available."""
    return LIGHTSIM2GRID_AVAILABLE


def check_lightsim2grid_available() -> None:
    """Check if lightsim2grid is available, raise ImportError if not."""
    if not LIGHTSIM2GRID_AVAILABLE:
        raise ImportError(
            "A recent lightsim2grid (with lightsim2grid.network.init_from_matpower) "
            "is required for this functionality. "
            "Install it with: pip install gridfm-datakit[lightsim2grid]",
        )
