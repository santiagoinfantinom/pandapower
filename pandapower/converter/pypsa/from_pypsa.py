# -*- coding: utf-8 -*-

import numpy as np
import logging

import pandapower as pp
import pypsa


def create_pypsa_test_net():
    network = pypsa.Network()
    network.add("Bus","MV bus", v_nom=20, v_mag_pu_set=1.02)
    network.add("Bus","LV1 bus", v_nom=.4)
    network.add("Bus","LV2 bus", v_nom=.4)
    network.add("Transformer", "trafo", type="0.4 MVA 20/0.4 kV", bus0="MV bus", bus1="LV1 bus")
    network.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
#  network.add("Line", "LV cable2", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="MV bus", length=0.1)
    network.add("Generator", "External Grid", bus="MV bus", control="Slack")
    network.add("Load", "LV load", bus="LV2 bus", p_set=0.1, q_set=0.0, sign=-1)
    network.add("Load", "static gen", bus="LV2 bus", p_set=0.2, q_set=0.07, sign=1)
    network.lpf()
    network.pf(use_seed=True)
    return network


def from_pypsa(pnet):

    def convert_buses():
        net.bus["name"] = pnet.buses.index.values
        net.bus["vn_kv"] = pnet.buses["v_nom"].values
        net.bus["in_service"] = True

    def convert_lines():
        net.line["from_bus"] = pnet.lines["bus0"].map(bus_lookup).values
        net.line["to_bus"] = pnet.lines["bus1"].map(bus_lookup).values
        net.line["length_km"] = pnet.lines["length"].values
        net.line["r_ohm_per_km"] = pnet.lines["r"].values / pnet.lines["length"].values
        net.line["x_ohm_per_km"] = pnet.lines["x"].values / pnet.lines["length"].values
        net.line["length_km"] = pnet.lines["length"].values
        net.line["parallel"] = pnet.lines["num_parallel"].values
        net.line["df"] = pnet.lines["terrain_factor"].values
        net.line["in_service"] = True
        # FIXME
        net.line.loc[:, ["c_nf_per_km", "g_us_per_km", "max_i_ka"]] = [0, 0, 1]

    def convert_ext_grids():
        for idx, v in pnet.generators.query("control == 'Slack'").iterrows():
            pp.create_ext_grid(net, bus_lookup[v.bus], vm_pu=pnet.buses.at[v.bus, "v_mag_pu_set"])

    def convert_trafos():
        for idx, v in pnet.transformers.query("type != ''").iterrows():
            pp.create_transformer(net, hv_bus=bus_lookup[v.bus0], lv_bus=bus_lookup[v.bus1],
                                  std_type=v.type)

    def convert_loads():
        def convert_element(element, d):
            net[element]["name"] = d.index.values
            net[element]["bus"] = d["bus"].map(bus_lookup).values
            net[element]["p_mw"] = d["p_set"].values
            net[element]["q_mvar"] = d["q_set"].values
            net[element].loc[:, ["scaling", "in_service"]] = [1., True]
        convert_element("load", pnet.loads[pnet.loads.sign <= 0])
        convert_element("sgen", pnet.loads[pnet.loads.sign > 0])
        net.load.loc[:, ["const_z_percent", "const_i_percent"]] = [0., 0.]

    net = pp.create_empty_network()
    convert_buses()
    bus_lookup = dict(zip(net.bus.name, net.bus.index))
    convert_lines()
    convert_ext_grids()
    convert_trafos()
    convert_loads()
    return net

def check_results_equal(pnet, net):
    pv = pnet.buses_t.v_mag_pu.loc["now"]
    assert np.allclose(pv.loc[net.bus.name].values, net.res_bus.vm_pu.values)
    pa = pnet.buses_t.v_ang.loc["now"] * 180. / np.pi
    assert np.allclose(pa.loc[net.bus.name].values, net.res_bus.va_degree.values)

# TODO transformer tap
# TODO transformer std type
# TODO transformer without std type

if __name__ == '__main__':
    pnet = pypsa.Network()
    pnet = create_pypsa_test_net()
    net = from_pypsa(pnet)
    pp.create_ext_grid(net, 0)
    pp.runpp(net, calculate_voltage_angles=True, max_iteration=100)
    check_results_equal(pnet, net)
