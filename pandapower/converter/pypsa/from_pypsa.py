# -*- coding: utf-8 -*-

import numpy as np
import os
import logging
import pandas as pd
import pandapower as pp
import pypsa
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries, control_diagnostic
from pandapower.control import ContinuousTapControl, ConstControl
import tempfile


from egoio.tools import db
from sqlalchemy.orm import sessionmaker
from ding0.core import NetworkDing0
from ding0.tools.results import save_nd_to_pickle
import os
import sys
import pandas as pd
from edisgo import EDisGo
from edisgo.tools import pypsa_io

def create_pypsa_test_net():
    network = pypsa.Network()
    network.set_snapshots(range(4))
    network.add("Bus","MV bus", v_nom=20, v_mag_pu_set=1.02)
    network.add("Bus","LV1 bus", v_nom=.4)
    network.add("Bus","LV2 bus", v_nom=.4)
    network.add("Transformer", "trafo", type="0.4 MVA 20/0.4 kV", bus0="MV bus", bus1="LV1 bus")
    network.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
#  network.add("Line", "LV cable2", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="MV bus", length=0.1)
    network.add("Generator", "External Grid", bus="MV bus", control="Slack")
    network.add("Load", "LV load", bus="LV2 bus", p_set=0.1, q_set=0.0, sign=-1)
    network.add("Load", "static gen", bus="LV2 bus", p_set=0.2, q_set=0.07, sign=1)
    #network.pf(use_seed=True)


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
        for idx, v in pnet.generators.query("control == 'Slack'").iterrows():            pp.create_ext_grid(net, bus_lookup[v.bus], vm_pu=pnet.buses.at[v.bus, "v_mag_pu_set"])

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

    def create_data_source(n_timesteps=10):
        profiles = pd.DataFrame()
        profiles['load1'] = np.random.random(n_timesteps) * 2e1
        profiles['load2'] = np.random.random(n_timesteps) * 2e1
        profiles['load2_mv_p'] = np.random.random(n_timesteps) * 4e1
        profiles['load2_mv_q'] = np.random.random(n_timesteps) * 1e1
        profiles['static gen'] = np.random.random(n_timesteps) * 4e1
        profiles['static gen2'] = np.random.random(n_timesteps) * 1e1

        profiles['load3_hv_p'] = profiles.load2_mv_p + abs(np.random.random())
        profiles['load3_hv_q'] = profiles.load2_mv_q + abs(np.random.random())

        profiles['slack_v'] = np.clip(np.random.random(n_timesteps) + 0.5, 0.8, 1.2)
        profiles['trafo_v'] = np.clip(np.random.random(n_timesteps) + 0.5, 0.9, 1.1)

        profiles["trafo_tap"] = np.random.randint(-3, 3, n_timesteps)

        ds = DFData(profiles)

        return profiles, ds

    def convert_ts_profiles(pandasnet):
        """
        Adapt pypsa format to pandas: Creates one dataframe containing
        the values for "p" and "q" for all loads and generators in every
        time step
        """
        profiles = pd.DataFrame()
        for i in range(0,len(pandasnet.load['name'])):
            name = pandasnet.load['name'][i]
            if name in pnet.loads_t['p_set']:
                profiles[name + '_p'] = pnet.loads_t['p_set'][pandasnet.load['name'][i]]
            if name in pnet.loads_t['q_set']:
                profiles[name + '_q'] = pnet.loads_t['q_set'][pandasnet.load['name'][i]]
        for j in range(0,len(pandasnet.sgen['name'])):
            name = pandasnet.sgen['name'][i]
            if name in pnet.generators_t['p_set']:
                profiles[name + '_p'] = pnet.generators_t['p_set'][pandasnet.sgen['name'][j]]
            if name in pnet.generators_t['q_set']:
                profiles[name + '_q'] = pnet.generators_t['q_set'][pandasnet.sgen['name'][j]]

        ds = pp.timeseries.DFData(profiles)

        return profiles,ds

    def convert_ts_profiles2(pandasnet):
        """
         Creates Profiles for load_p, load_q, sgen_p, and sgen_q with
         their respective timeseries' values
         """
        prof_load_p = pd.DataFrame()
        prof_load_q = pd.DataFrame()
        prof_sgen_p = pd.DataFrame()
        prof_sgen_q = pd.DataFrame()

        for i in range(0, len(pandasnet.load['name'])):
            # Append timeseries profile to dataframe
            name = pandasnet.load['name'][i]
            if name in pnet.loads_t['p_set']:
                prof_load_p[name + '_p'] = pnet.loads_t['p_set'][pandasnet.load['name'][i]]
            if name in pnet.loads_t['q_set']:
                prof_load_q[name + '_q'] = pnet.loads_t['q_set'][pandasnet.load['name'][i]]
        for j in range(0, len(pandasnet.sgen['name'])):
            name = pandasnet.sgen['name'][i]
            if name in pnet.generators_t['p_set']:
                prof_sgen_p[name + '_p'] = pnet.generators_t['p_set'][pandasnet.sgen['name'][j]]
            if name in pnet.generators_t['q_set']:
                prof_sgen_q[name + '_q'] = pnet.generators_t['q_set'][pandasnet.sgen['name'][j]]

        #Convert profiles to the proper pandapower DFData format. Resets index of rows to convert timeseries indexing into discrete steps
        ds_load_p = pp.timeseries.DFData(prof_load_p.reset_index(drop=True))
        ds_load_q = pp.timeseries.DFData(prof_load_q.reset_index(drop=True))
        ds_sgen_p =pp.timeseries.DFData(prof_sgen_p.reset_index(drop=True))
        ds_sgen_q =pp.timeseries.DFData(prof_sgen_q.reset_index(drop=True))

        ds = [ds_load_p,ds_load_q,ds_sgen_p, ds_sgen_q]
        return ds

    def create_controllers(data_source):
        """
        Converts the load/gen profiles from pypsa into Panda profiles and DFData
        """
        ds = data_source

        for i in range(0,len(ds.df.columns)):
            if 'load' or 'Load' in ds.df.columns[i]:
                if '_p' in ds.df.columns[i]:
                    ConstControl(net, 'load', 'p_mw', element_index=i,
                                      data_source=ds, profile_name=ds.df.columns[i])
                if '_q' in ds.df.columns[i]:
                    ConstControl(net, 'load', 'q_mvar', element_index=i,
                                      data_source=ds, profile_name=ds.df.columns[i])
            elif 'gen' or 'Gen' in ds.df.columns[i]:
                if '_p' in ds.df.columns[i]:
                    ConstControl(net, 'sgen', 'p_mw', element_index=i,
                                      data_source=ds, profile_name=ds.df.columns[i])
                if '_q' in ds.df.columns[i]:
                    ConstControl(net, 'sgen', 'q_mvar', element_index=i,
                                      data_source=ds, profile_name=ds.df.columns[i])

        """
        load_names = []
        load_elems = []
        sgen_names = []
        sgen_elems = []
        for i in range(0, net['load'].shape[0]):
            load_names.append(net.load['name'][i])
            load_elems.append(i)
        for j in range(0, net['sgen'].shape[0]):
            sgen_names.append(net.sgen['name'][j])
            sgen_elems.append(i)
        """

    def create_controllers_2(ds):
        """
        Uses the created datasources to create 1 controller per profile for all loads/sgens
        """
        ds_load_p, ds_load_q, ds_sgen_p, ds_sgen_q = ds

        if not ds_load_p.df.empty:
            cont_loads_p = ConstControl(net, element='load', element_index=net.load.index,
                                    variable='p_mw',
                                    data_source=ds_load_p, profile_name=list(ds_load_p.df.columns),
                                    level=0)
        if not ds_load_q.df.empty:
            cont_loads_q = ConstControl(net, element='load', element_index=net.load.index,
                                    variable='q_mvar', data_source=ds_load_q, profile_name=list(ds_load_q.df.columns),
                                    level = 1)
        if not ds_sgen_p.df.empty:
            cont_sgen_p = ConstControl(net, element='sgen', element_index=net.sgen.index,
                                    variable='p_mw', data_source=ds_sgen_p, profile_name=list(ds_sgen_p.df.columns),
                                   level = 2)
        if not ds_sgen_q.df.empty:
            cont_sgen_q = ConstControl(net, element='sgen', element_index=net.sgen.index,
                                    variable='q_mvar', data_source=ds_sgen_q, profile_name=list(ds_sgen_q.df.columns),
                                   level = 3)

    def create_output_writer(net, time_steps, output_dir):
        ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".json")
        # these variables are saved to the harddisk after / during the time series loop
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')
        ow.log_variable('res_line', 'p_from_mw')
        ow.log_variable('res_line', 'p_to_mw')
        ow.log_variable('res_line', 'i_ka')
        return ow

    def convert_time_series_data():
        #profiles, ds = create_data_source()
        ds = convert_ts_profiles2(net)
        create_controllers_2(ds)
        output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        ow = create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps, numba=False, output_writer=ow)

    net = pp.create_empty_network()
    convert_buses()
    bus_lookup = dict(zip(net.bus.name, net.bus.index))
    convert_lines()
    convert_ext_grids()
    convert_trafos()
    convert_loads()

    if len(pnet.snapshots) != 0:
        n_timesteps = len(pnet.snapshots)
        time_steps = range(0, n_timesteps)
        convert_time_series_data()

    return net

def check_results_equal(pnet, net, t_series = False):
    pv = pnet.buses_t.v_mag_pu.loc["now"]
    assert np.allclose(pv.loc[net.bus.name].values, net.res_bus.vm_pu.values)
    pa = pnet.buses_t.v_ang.loc["now"] * 180. / np.pi
    assert np.allclose(pa.loc[net.bus.name].values, net.res_bus.va_degree.values)
    if t_series == True:
        res_v_ts = []
        for i in range(0,len(pnet.snapshots)):
            res_v_ts.append(pnet.buses_t.v_mag_pu.iloc[i])

def edisgo_test():

    ding0_grid = 'ding0_grid_example.pkl'
    """
    mv_grid_districts = [460]

    engine = db.connection(section='oedb')
    session = sessionmaker(bind=engine)()

    nd = NetworkDing0(name='network')

    nd.run_ding0(session=session,
    mv_grid_districts_no=mv_grid_districts)

    save_nd_to_pickle(nd, filename=ding0_grid)
    """

    worst_case_analysis = 'worst-case'

    timeindex = pd.date_range('1/1/2011', periods=4, freq='H')
    # load time series (scaled by annual demand)
    timeseries_load = pd.DataFrame({'residential': [0.0001] * len(timeindex),
                                    'retail': [0.0002] * len(timeindex),
                                    'industrial': [0.0015] * len(timeindex),
                                    'agricultural': [0.00005] * len(timeindex)},
                                   index=timeindex)

    # feed-in time series of fluctuating generators (scaled by nominal power)
    timeseries_generation_fluctuating = \
        pd.DataFrame({'solar': [0.2] * len(timeindex),
                      'wind': [0.3] * len(timeindex)},
                     index=timeindex)
    # feed-in time series of dispatchable generators (scaled by nominal power)
    timeseries_generation_dispatchable = \
        pd.DataFrame({'biomass': [1] * len(timeindex),
                      'coal': [1] * len(timeindex),
                      'other': [1] * len(timeindex)},
                     index=timeindex)

    edisgo = EDisGo(
        ding0_grid="ding0_grid_example.pkl",
        timeseries_load="demandlib",
        timeseries_generation_fluctuating="oedb",
        timeseries_generation_dispatchable=timeseries_generation_dispatchable,
        timeindex=timeindex)

    edisgo_pypsa = pypsa_io.to_pypsa(edisgo.network, None, timesteps=edisgo.network.timeseries.timeindex)
    edisgo_pypsa.t_series = True

    edisgo_net = from_pypsa(edisgo_pypsa)

    """
    edisgo_pypsa.lpf()
    now = edisgo_pypsa.snapshots[0]
    angle_diff = pd.Series(edisgo_pypsa.buses_t.v_ang.loc[now, edisgo_pypsa.lines.bus0].values -
                           edisgo_pypsa.buses_t.v_ang.loc[now, edisgo_pypsa.lines.bus1].values,
                           index=edisgo_pypsa.lines.index)
    
    (angle_diff * 180 / np.pi).describe()
    """
    edisgo.analyze()
    Res_edisgo = edisgo.network.results
    #check_results_equal()


if __name__ == '__main__':

    #TODO: Check Results equal for time series
    #TODO: Check ERROR:pandapower.timeseries.output_writer:Error at index [0, 1, 2] for res_bus[v_mag]: 'the label [v_mag] is not in the [columns]'
    #TODO Check ValueError: Power flow analysis did not converge.

    edisgo_test()
    """
    pnet = pypsa.Network()
    pnet = create_pypsa_test_net()
    net = from_pypsa(pnet)
    pp.create_ext_grid(net, 0)
    pp.runpp(net, calculate_voltage_angles=True, max_iteration=100)
    """

    #check_results_equal(pnet, net)

