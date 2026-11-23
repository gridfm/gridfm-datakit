"""Several pypowsybl loads on one bus share the bus load."""

import numpy as np
import pytest

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.powsybl.convert import update_powsybl

pytestmark = pytest.mark.skipif(
    not powsybl.is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


class TestSeveralLoadsOnOneBus:
    """from_powsybl sums the loads of a bus, update_powsybl splits it back."""

    def test_update_keeps_every_load(self):
        import pypowsybl as pp

        pp_net = pp.network.create_ieee14()
        pp_net.create_loads(
            id="B4-L2",
            voltage_level_id="VL4",
            bus_id="B4",
            p0=10.0,
            q0=2.0,
        )
        net = powsybl.from_powsybl(pp_net)
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_net.per_unit = False
        before = pp_net.get_loads()[["p0", "q0"]].copy()
        update_powsybl(pp_net, net, mapping)
        pp_net.per_unit = False
        after = pp_net.get_loads()[["p0", "q0"]]
        assert np.allclose(after.loc[before.index], before)
