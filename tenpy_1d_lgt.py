import cProfile
from pathlib import Path

import numpy as np
import scipy.sparse
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import Site, set_common_charges

# hardcoded for MWE
model_name = "1D-SU3-2flavor-2level"
flavors = tuple("ud")
qops = ["Rlnk", "Llnk", "Nu", "Nd"]
qmods = [3, 3, 1, 1]


def LGT_unit_cell(Lu):
    operators = Path("qhlgt-models", model_name, "out").glob("*.npz")
    operators = {op.stem: scipy.sparse.load_npz(op).toarray() for op in operators}

    # charge values
    qflat = np.column_stack([operators[k].diagonal() for k in qops])

    # fermionic operators FIXME: based on op name, dangerous
    fermionic = {
        k for k in operators if not "N" in k and sum(k.count(f) for f in flavors) % 2
    }

    unit_cell = []
    for i in range(Lu):
        # indpenendent link symmetries for each link of the unit cell
        qnames = [f"lnk{(i + j) % Lu}" for j in range(2)] + qops[2:]
        leg = LegCharge.from_qflat(ChargeInfo(qmods, qnames), qflat)
        # sort later, via set_common_charges (doubles from_ndarray calls)
        site = Site(leg, **operators, sort_charge=False)
        site.need_JW_string = fermionic
        unit_cell.append(site)
    set_common_charges(unit_cell, sort_charge=True)

    return unit_cell


class LGTModel(CouplingModel, MPOModel):
    """
    Lattice Gauge Theory with Lu independent link constraints.
    A unit cell consists of Lu dressed sites, each embedding
    matter & gauge dofs.

    For 2-site DMRG, Lu>2 should be used.
    With staggered fermions Lu must be even.
    """

    def __init__(self, L, Lu):
        lat = Lattice([L], LGT_unit_cell(Lu))
        CouplingModel.__init__(self, lat)

        # link streght and mass terms
        for u1 in range(Lu):
            dx, u2 = divmod(u1 + 1, Lu)
            self.add_onsite(1.0, u1, "Nlnk", "electric")
            for f in flavors:
                self.add_onsite(1.0, u1, f"N{f}", "mass")
                self.add_coupling(
                    1.0j,
                    u1,
                    f"L{f}_hc",
                    u2,
                    f"{f}R_hc",
                    dx,
                    category="hopping",
                    plus_hc=True,
                )

        MPOModel.__init__(self, lat, self.calc_H_MPO())


cProfile.run("LGTModel(L=8, Lu=4)", "prof")
