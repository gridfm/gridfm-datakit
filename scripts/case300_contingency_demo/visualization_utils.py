import numpy as np
import pandas as pd
import os
import glob
import shutil
import random
#import mplcursors
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from ipywidgets import interactive, Layout
from tqdm import tqdm
import ipywidgets as widgets

from IPython.display import display, clear_output

from GridDataGen.utils.io import *
from GridDataGen.utils.process_network import *
from GridDataGen.utils.config import *
from GridDataGen.utils.stats import *
from GridDataGen.utils.param_handler import *
from GridDataGen.utils.load import *
from GridDataGen.utils.solvers import *
from pandapower.auxiliary import pandapowerNet

from typing import Optional, Union

import pandapower as pp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Grid(object):
    
    node_types = {1:"PQ", 2:"PV", 3:"REF"}
    node_shapes = {"REF": "s", "PV": "^", "PQ": "o"}
    node_sizes = {"REF": 50, "PV": 25, "PQ": 30}

        
    def __init__(
            self,
            data_root:str, 
            model_output_path:str,
            ll_ac_threshold:float=1.0,
            ll_dc_threshold:float=0.95,
            ll_model_threshold:float=0.95
            ):
        
        self.data_root = data_root
        self.model_output_path = model_output_path
        self.ll_ac_threshold = ll_ac_threshold
        self.ll_dc_threshold = ll_dc_threshold
        self.ll_model_threshold = ll_model_threshold

        self.node_data = pd.read_csv(os.path.join(self.data_root, "pf_node.csv"))
        self.bus_params = pd.read_csv(os.path.join(self.data_root, "bus_params.csv"), dtype={"type":int})
        self.edge_params = pd.read_csv(os.path.join(self.data_root, "edge_params.csv"), dtype={'from_bus': int, 'to_bus': int})
        self.edge_params.insert(0, 'edge_idx', self.edge_params.index)
        self.model_data = pd.read_csv(self.model_output_path)
        self.model_data.drop([self.model_data.columns[0], "PQ", "PV", "REF"], axis=1, inplace=True)
        self.model_data.rename(columns={"PD":"Pd_pred", "QD":"Qd_pred", "PG":"Pg_pred", "QG":"Qg_pred", "VM":"Vm_pred", "VA":"Va_pred"}, inplace=True)

        # merge node_data and model_data
        self.node_data = self.node_data.merge(self.model_data, left_on=['scenario','bus'], right_on = ['scenario','bus'], how='inner')
        
         # Correct model outputs of Voltage magnitude and angle values for PV buses
        self.node_data["Vm_pred_corrected"] = self.node_data["Vm_pred"]
        self.node_data["Va_pred_corrected"] = self.node_data["Va_pred"]
        self.node_data.loc[self.node_data["PV"]==1, "Vm_pred_corrected"] = self.node_data.loc[self.node_data["PV"]==1, "Vm"]
        self.node_data.loc[self.node_data["REF"]==1, "Va_pred_corrected"] = self.node_data.loc[self.node_data["REF"]==1, "Va"]

        # Correct dc values
        self.node_data["Vm_dc_corrected"] = self.node_data["Vm_dc"]
        self.node_data["Va_dc_corrected"] = self.node_data["Va_dc"]
        self.node_data.loc[self.node_data["PV"]==1, "Vm_dc_corrected"] = self.node_data.loc[self.node_data["PV"]==1, "Vm"]
        self.node_data.loc[self.node_data["REF"]==1, "Va_dc_corrected"] = self.node_data.loc[self.node_data["REF"]==1, "Va"]

        # Create map of scenario -> total load
        self.scenario_load_df = self.node_data.groupby(["scenario"])["Pd"].sum().to_frame()
        self.scenario_load_df.reset_index(inplace=True)
        self.scenario_load_df.sort_values(by=["Pd"], inplace=True)
        self.scenario_load_df["Pd"] /= 1000.0 #self.scenario_load_df["Pd"]
        self.scenario_load_sorted = self.scenario_load_df["scenario"].tolist()
        self.scenario_load_map = { #in Kw
            k:np.round(v) for (k,v) in zip(
                self.scenario_load_df["scenario"].tolist(),
                self.scenario_load_df["Pd"].tolist()
            )
        }
        
        # create list of selectable scenarios by user, based on unique scenarios from node_data
        self.valid_scenarios = list(self.node_data["scenario"].unique().astype(np.int32))

        # convert branch_idx_removed data to scenario -> line index list lookup table
        branch_idx_removed_df = pd.read_csv(os.path.join(self.data_root, "branch_idx_removed.csv"))
        self.edges_removed = {
            k:[np.int32(x) for x in sorted([v1, v2]) if not np.isnan(x)] 
            for (k, v1, v2) in 
            zip(branch_idx_removed_df.scenario.tolist(), 
                branch_idx_removed_df["0"].tolist(), 
                branch_idx_removed_df["1"].tolist())
        }

        # filter scenarios according to model predictions and data gen overlap
        #self.edges_removed = {k:v for k,v in self.edges_removed.items() if k in node_data_scenarios}
        
        # create an edge_index -> node-tuple 
        self.edge_nodes = {
            k:(v1,v2) for k,v1,v2 in zip(
                self.edge_params['edge_idx'].tolist(),
                self.edge_params["from_bus"].tolist(),
                self.edge_params["to_bus"].tolist(),
            )
        }

        # create dictionary to map line-contingencies to all scenarios that have this contingency (i.e. all load examples)
        self.contingency_scenarios = {}
        for key, val in self.edges_removed.items():
            tuple_val = tuple(val)#frozenset(val)
            if tuple_val in self.contingency_scenarios:
                self.contingency_scenarios[tuple_val].append(key)
            else:
                self.contingency_scenarios[tuple_val] = [key]

        # start calculating branch current and loading       
        self.sn_mva = 100
        base_kV = self.bus_params["baseKV"].values

        # Extract from- and to-bus indices
        from_idx = self.edge_params["from_bus"].values.astype(np.int32)
        to_idx = self.edge_params["to_bus"].values.astype(np.int32)

        # Extract branch admittance coefficients
        Yff = self.edge_params["Yff_r"].values + 1j * self.edge_params["Yff_i"].values
        Yft = self.edge_params["Yft_r"].values + 1j * self.edge_params["Yft_i"].values
        Ytf = self.edge_params["Ytf_r"].values + 1j * self.edge_params["Ytf_i"].values
        Ytt = self.edge_params["Ytt_r"].values + 1j * self.edge_params["Ytt_i"].values

        # Extract base voltages for from- and to-buses
        self.Vf_base_kV = base_kV[from_idx]
        self.Vt_base_kV = base_kV[to_idx]

        # Number of lines/edges and buses
        self.nl = self.edge_params.shape[0]
        self.nb = self.bus_params.shape[0]

        # i = [0, 1, ..., nl-1, 0, 1, ..., nl-1], used for constructing Yf and Yt
        i = np.hstack([np.arange(self.nl), np.arange(self.nl)])

        # Construct from-end admittance matrix Yf using the linear combination:
        # Yf[b, :] = y_ff_b * e_f + y_ft_b * e_t
        self.Yf = csr_matrix((np.hstack([Yff, Yft]), (i, np.hstack([from_idx, to_idx]))), shape=(self.nl, self.nb))
        self.Yt = csr_matrix((np.hstack([Ytf, Ytt]), (i, np.hstack([from_idx, to_idx]))), shape=(self.nl, self.nb))

        self.rate_a = self.edge_params["rate_a"].values

        del Yff, Yft, Ytf, Ytt

        # Construct a base graph from all the lines and nodes
        self.Gb, self.Gb_pos = self._create_graph()

        # Set some titles for plotting
        self.edge_var_labels = {
            "rate_a":"Line Ampre [A]", 
            "loading":"Line Loading [x100%]"}
        self.node_var_labels = {
            "Pd":"Pd []", "Qd":"Qd []", "Pg":"Pg []", "Qg":"Qg []",
            "Vm":"Voltage Magnitude [pu]", "Va":"Voltage Angle []"}
        

    def _calc_V(self, vm:np.ndarray, va:np.ndarray) -> np.ndarray:
        """
        """
        V = vm * np.exp(1j * va * np.pi/180.0)
        return V
    

    def _calc_I(self, V:np.ndarray) -> np.ndarray:
        """
        """
        If_pu = self.Yf @ V  # From-end currents in per-unit (I_f = Y_f V)
        If_kA = np.abs(If_pu) * self.sn_mva / (np.sqrt(3) * self.Vf_base_kV)  # Conversion to kA

        # Construct to-end admittance matrix Yt:
        # Yt[b, :] = y_tf_b * e_f + y_tt_b * e_t
        It_pu = self.Yt @ V  # To-end currents in per-unit (I_t = Y_t V)
        It_kA = np.abs(It_pu) * self.sn_mva / (np.sqrt(3) * self.Vt_base_kV)  # Conversion to kA

        return If_kA, It_kA
    

    def _calc_loading(self, If_kA:np.ndarray, It_kA:np.ndarray) -> np.ndarray:
        """
        """
        limitf = self.rate_a / (self.Vf_base_kV * np.sqrt(3))
        limitt = self.rate_a / (self.Vt_base_kV * np.sqrt(3))

        loadingf = If_kA / limitf
        loadingt = It_kA / limitt

        return np.maximum(loadingf, loadingt)
            

    def compute_loadings(self):#, scenario:Optional[Union[int,list]]=None):
        """
        """
        #nl = self.edge_params.shape[0]
        scenario_l, from_bus_l, to_bus_l, edge_idx_l, rate_a = [], [], [], [], []
        If_true_l, It_true_l = [], []
        If_pred_l, It_pred_l = [], []
        If_dc_pred_l, It_dc_pred_l = [], []
        loading_true_l, loading_pred_l, loading_dc_pred_l = [], [], []

        for cidx, sidx in enumerate(tqdm(self.node_data["scenario"].unique(), desc="Computing loadings")):

            node_df_sidx = self.node_data[self.node_data["scenario"] == sidx]

            # construct complex voltages
            V_true = self._calc_V(node_df_sidx["Vm"].values, 
                                  node_df_sidx["Va"].values)
            V_pred = self._calc_V(node_df_sidx["Vm_pred_corrected"].values, 
                                  node_df_sidx["Va_pred_corrected"].values)
            V_dc_pred = self._calc_V(node_df_sidx["Vm_dc_corrected"].values, 
                                     node_df_sidx["Va_dc_corrected"].values)

            # calculate currents
            If_true, It_true = self._calc_I(V_true)
            If_pred, It_pred = self._calc_I(V_pred)
            If_dc_pred, It_dc_pred = self._calc_I(V_dc_pred)

            # calculate loadings
            loading_true = self._calc_loading(If_true, It_true)
            loading_pred = self._calc_loading(If_pred, It_pred)
            loading_dc_pred = self._calc_loading(If_dc_pred, It_dc_pred)

            # assign nan to dropped lines' loadings
            loading_true[self.edges_removed[sidx]] = np.nan
            loading_pred[self.edges_removed[sidx]] = np.nan
            loading_dc_pred[self.edges_removed[sidx]] = np.nan

            # TODO Need to make this pandas code more efficient
            # add loadings to edge_params
            scenario_l += [sidx]*self.nl
            edge_idx_l += self.edge_params["edge_idx"].tolist()
            from_bus_l += self.edge_params["from_bus"].tolist()
            to_bus_l += self.edge_params["to_bus"].tolist()
            rate_a += self.edge_params["rate_a"].tolist()
            If_true_l += list(If_true)
            It_true_l += list(It_true)
            If_pred_l += list(If_pred)
            It_pred_l += list(It_pred)
            If_dc_pred_l += list(If_dc_pred)
            It_dc_pred_l += list(It_dc_pred)
            loading_true_l += list(loading_true)
            loading_pred_l += list(loading_pred)
            loading_dc_pred_l += list(loading_dc_pred)

        # create edge_data data-frame to store information for plotting
        self.edge_data = pd.DataFrame(
            {
                "scenario": scenario_l,
                "edge_idx": edge_idx_l,
                "from_bus": from_bus_l,
                "to_bus": to_bus_l,
                "rate_a": rate_a,
                "If": If_true_l,
                "It": It_true_l,
                "If_pred": If_pred_l,
                "It_pred": It_pred_l,
                "If_dc_pred": If_dc_pred_l,
                "It_dc_pred": It_dc_pred_l,
                "loading": loading_true_l,
                "loading_pred": loading_pred_l,
                "loading_dc_pred": loading_dc_pred_l
                })

        # calculate line and node violations from thresholds
        node_var = "Vm"
        edge_var = "loading"
        self.bus_vmin = self.bus_params["vmin"].tolist()[0]
        self.bus_vmax = self.bus_params["vmax"].tolist()[0]

        # calculate node violations from bus vmin and vmax
        self.node_data[node_var+"_violation"] = self.node_data[node_var].apply(lambda x: int((x>self.bus_vmax) or (x<self.bus_vmin)) if ~np.isnan(x) else 0)
        self.node_data[node_var+"_pred"+"_violation"] = self.node_data[node_var+"_pred"].apply(lambda x: int((x>self.bus_vmax-0.01) or (x<self.bus_vmin+0.01)) if ~np.isnan(x) else 0)
        #self.node_data[node_var+"_dc_pred"+"_violation"] = 0 

        # calculate edge violations from bus vmin and vmax
        self.edge_data[edge_var+"_violation"] = self.edge_data[edge_var].apply(lambda x: int(x>self.ll_ac_threshold) if ~np.isnan(x) else 0)
        self.edge_data[edge_var+"_pred"+"_violation"] = self.edge_data[edge_var+"_pred"].apply(lambda x: int(x>self.ll_model_threshold) if ~np.isnan(x) else 0)
        #self.edge_data[edge_var+"_dc_pred"+"_violation"] = self.edge_data[edge_var+"_dc_pred"].apply(lambda x: int(x>self.ll_dc_threshold) if ~np.isnan(x) else 0)


    def _create_graph(self):
        """
        """
        # create list of all nodes with parameters as data
        graph_nodes = [ 
            (np.int32(u), {"type":Grid.node_types[t], "vmin":vmn, "vmax":vmx, "baseKV":bkv})
            for u, t, vmn, vmx, bkv in zip(
                self.bus_params.bus.tolist(),
                self.bus_params.type.tolist(),
                self.bus_params.vmin.tolist(),
                self.bus_params.vmax.tolist(),
                self.bus_params.baseKV.tolist(),
            )
        ]

        # create list of all edges with parameters as data
        graph_edges = [
            (np.int32(u), np.int32(v), {"edge_idx":idx, "rate_a":rate_a})
            for u, v, idx, rate_a in zip(
                self.edge_params.from_bus.tolist(),
                self.edge_params.to_bus.tolist(),
                self.edge_params.edge_idx.tolist(),
                self.edge_params.rate_a.tolist()
            )
            if u != v
            ]
        
        # create graph
        #G = nx.MultiGraph()
        G = nx.Graph()
        G.add_nodes_from(graph_nodes)
        G.add_edges_from(graph_edges)
        
        # set node positions
        pos = nx.spring_layout(G, seed=42)

        return G, pos
    

    def _draw_nodes(self, ax, nodes, node_colors, nt):
            """
            Draw nodes from Gb on a given axis
            """
            nx.draw_networkx_nodes(
                self.Gb,
                self.Gb_pos,
                nodelist=nodes,
                node_color=node_colors,
                ax=ax,
                node_size=Grid.node_sizes[nt],
                node_shape=Grid.node_shapes[nt],
                edgecolors='black', 
                linewidths=0.5
                )

    
    def plot_single_contingency_case(
            self, 
            scenario:int,
            edge_var:str, 
            node_var:str,
            pf_ac:bool=True,
            #node_threshold:float
            ):
        """
        Plot PandaPower and GridFM graphs of a single (contingency) example, showing line and node variables.
        Line violations for the single example are indicated by red.
        """
        edges_removed = self.edges_removed[scenario]
        #print(f"Line(s) dropped: ", edges_removed)

        # get all scenarios for the current contingency (load perturbations)
        contingency_scenarios = sorted(list(set(self.contingency_scenarios[tuple(edges_removed)]).intersection(set(self.valid_scenarios))))
        #print(f"Number of scenarios with lines {','.join([str(x) for x in edges_removed])} dropped: ", len(contingency_scenarios))

        # select from node_data and edge_data dataframes the current contingency
        ndf = self.node_data[self.node_data.scenario == scenario]
        edf = self.edge_data[self.edge_data.scenario == scenario]
        
        # set colormaps for edges and nodes
        #clrs_n = ["lightblue", "red"]
        #cmap_n_name = 'nodes_cmap'
        #cmap_n = mcolors.LinearSegmentedColormap.from_list(cmap_n_name, clrs_n, N=256)
        #cmap_n = plt.get_cmap('cool')
        cmap_n = plt.get_cmap('jet')
        cmap_n = mcolors.LinearSegmentedColormap.from_list('trunc_cool', cmap_n(np.linspace(0.2, 0.8, 256)))
        vmin_n = np.nanmin([np.nanmin(ndf[node_var].tolist()),np.nanmin(ndf[node_var+"_pred"])])
        vmax_n = np.nanmax([np.nanmax(ndf[node_var].tolist()),np.nanmax(ndf[node_var+"_pred"])])
        norm_n = mcolors.Normalize(vmin=vmin_n, vmax=vmax_n)
        sm_n = plt.cm.ScalarMappable(cmap=cmap_n, norm=norm_n)

        #cmap_e = plt.get_cmap('cool')
        cmap_e = plt.get_cmap('jet')
        cmap_e = mcolors.LinearSegmentedColormap.from_list('trunc_cool', cmap_e(np.linspace(0.2, 0.8, 256)))
        vmin_e = np.nanmin([np.nanmin(edf[edge_var].tolist()),np.nanmin(edf[edge_var+"_pred"])])
        vmax_e = np.nanmax([np.nanmax(edf[edge_var].tolist()),np.nanmax(edf[edge_var+"_pred"])])
        norm_e = mcolors.Normalize(vmin=vmin_e, vmax=vmax_e)
        sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
        
        
        # ======== Create figure
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, figure=fig)
        plt.clf()

        # ----- left axis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"AC Solver")
        ax1.text(0.05, 0.98, 
                 f'Total Load: {self.scenario_load_map[scenario]}MW', 
                 horizontalalignment='left', 
                 verticalalignment='top', 
                 transform=ax1.transAxes)

        # draw nodes, using node_var value color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            dfnc_filtered = ndf[ndf[nt]==1.0]
            nodes = dfnc_filtered["bus"].tolist()
            node_vals = dfnc_filtered[node_var].tolist()
            if nt=="PQ":
                node_colors = cmap_n(norm_n(node_vals)) 
            else:
                node_colors = "lightgray"
            self._draw_nodes(ax=ax1, nodes=nodes, node_colors=node_colors, nt=nt)

        # highlight node violations with red
        for nt, ns in Grid.node_shapes.items():
            if nt == "PQ":
                dfnc_filtered = ndf[(ndf[nt]==1.0) & (ndf[node_var+"_violation"]==1)]
                nodes = dfnc_filtered["bus"].tolist()
                node_colors = "red"
                self._draw_nodes(ax=ax1, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list = [(u,v) for u,v in zip(edf.from_bus.tolist(), edf.to_bus.tolist())]
        edge_vals = edf[edge_var].tolist()
        edge_color = cmap_e(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_color,
            ax=ax1,
            width=1.5)

        # show contingency lines as dashed/thicker
        con_edge_list = [self.edge_nodes[li] for li in edges_removed]
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=con_edge_list,
            edge_color='k',
            style=':',
            ax=ax1,
            width=3)
        
        # highlight line violations
        exceed_edge_list = [(u,v) for u,v,e in 
                            zip(edf.from_bus.tolist(), 
                                edf.to_bus.tolist(),
                                edf[edge_var+"_violation"])
                                if e==1]
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=exceed_edge_list,
            edge_color='r',
            style='-',
            ax=ax1,
            width=3)

        # ----- right axis
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title(f"GridFM")# (contingency {scenario})")
        ax2.text(0.05, 0.98, 
                 f'Total Load: {self.scenario_load_map[scenario]}MW', 
                 horizontalalignment='left', 
                 verticalalignment='top', 
                 transform=ax2.transAxes)
        
        # draw nodes, using only single color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            dfnc_filtered = ndf[ndf[nt]==1.0]
            nodes = dfnc_filtered["bus"].tolist()
            node_vals = dfnc_filtered[node_var+"_pred"].tolist()
            if nt=="PQ":
                node_colors = cmap_n(norm_n(node_vals)) 
            else:
                node_colors = "lightgray"
            self._draw_nodes(ax=ax2, nodes=nodes, node_colors=node_colors, nt=nt)
         
        # highlight node violations with red
        for nt, ns in Grid.node_shapes.items():
            if nt == "PQ":
                dfnc_filtered = ndf[(ndf[nt]==1.0) & (ndf[node_var+"_pred"+"_violation"]==1)]
                nodes = dfnc_filtered["bus"].tolist()
                node_colors = "red"
                self._draw_nodes(ax=ax2, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list = [(u,v) for u,v in zip(edf.from_bus.tolist(), edf.to_bus.tolist())]
        edge_vals = edf[edge_var+"_pred"].tolist()
        edge_color = cmap_e(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_color,
            ax=ax2,
            width=1.5)
        
        # show contingency lines as dashed/thicker
        con_edge_list = [self.edge_nodes[li] for li in edges_removed]
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=con_edge_list,
            edge_color='k',
            style=':',
            ax=ax2,
            width=3)
        
        # highlight line violations
        exceed_edge_list = [(u,v) for u,v,e in 
                            zip(edf.from_bus.tolist(), 
                                edf.to_bus.tolist(),
                                edf[edge_var+"_pred"+"_violation"])
                                if e==1]
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=exceed_edge_list,
            edge_color='r',
            style='-',
            ax=ax2,
            width=3)
        
        ax1.sharex(ax2)
        ax1.sharey(ax2)
        
        l_violation, = plt.plot([None], [None], ls="-", lw=3, color='r')
        n_violation = plt.scatter([None], [None], marker="o", s=40, color='r')
        m_ref = plt.scatter([None], [None], marker="s", s=40, color='0.5')
        m_pv = plt.scatter([None], [None], marker="^", s=40, color='0.5')
        m_pq = plt.scatter([None], [None], marker="o", s=40, color='0.5')
        ax1.legend([l_violation, n_violation, m_ref, m_pv, m_pq], 
                    ['Line Violation', 'Bus Violation', 'REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=5)
        ax2.legend([l_violation, n_violation, m_ref, m_pv, m_pq], 
                    ['Line Violation', 'Bus Violation', 'REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=5)

         # show edge colorbar
        sm_e.set_array([])
        plt.colorbar(sm_e, 
                     label=self.edge_var_labels[edge_var], 
                     ax=ax1,
                     orientation='horizontal', 
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)

        # show node colorbar
        sm_n.set_array([])
        plt.colorbar(sm_n, 
                     label=self.node_var_labels[node_var],
                     ax=ax2,
                     orientation='horizontal',
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.tight_layout()
        fig.canvas.header_visible = False
        plt.show()
    

    def plot_single_causal_analysis_case(
            self, 
            line_idx:int,
            edge_var:str, 
            node_var:str,
            pf_ac:bool=True,
            #node_threshold:float
            ):
        """
        Plot PandaPower and GridFM graphs showing all other line contingencies that cause the given line to fail, as well as 
        the statistics (line loading) of the specified line.
        """
        # collect all scenarios for which the given line was overloaded
        line_violation_scenarios = self.edge_data[(self.edge_data["edge_idx"]==line_idx) & (self.edge_data[edge_var+"_violation"]==1)]["scenario"].tolist()
        line_violation_scenarios_pred = self.edge_data[(self.edge_data["edge_idx"]==line_idx) & (self.edge_data[edge_var+"_pred"+"_violation"]==1)]["scenario"].tolist()
        causal_edges = {k:0 for k in self.edge_data["edge_idx"].tolist()}
        causal_edges_pred = {k:0 for k in self.edge_data["edge_idx"].tolist()}
        # ASK: Include only N-1 or N-1 & N-2
        for s in line_violation_scenarios:
            edges_removed = self.edges_removed[s]
            if len(edges_removed) == 1:
                causal_edges[edges_removed[0]] += 1

        for s in line_violation_scenarios_pred:
            edges_removed = self.edges_removed[s]
            if len(edges_removed) == 1:
                causal_edges_pred[edges_removed[0]] += 1


        #cmap_e = plt.get_cmap('jet')
        clrs_e = ["lightgrey", "green", "orange", "purple"]
        cmap_e_name = 'edges_cmap'
        cmap_e = mcolors.LinearSegmentedColormap.from_list(cmap_e_name, clrs_e, N=256)
        vmin_e = np.nanmin([np.nanmin(list(causal_edges.values())),np.nanmin(list(causal_edges_pred.values()))])
        vmax_e = np.nanmax([np.nanmax(list(causal_edges.values())),np.nanmax(list(causal_edges_pred.values()))])
        norm_e = mcolors.Normalize(vmin=vmin_e, vmax=vmax_e)
        sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
        
        
        # ======== Create figure
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, figure=fig)
        plt.clf()

        # ----- left axis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"AC Solver")

        # draw nodes, using node_var value color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            nodes = self.node_data[(self.node_data["scenario"]==self.valid_scenarios[0]) & (self.node_data[nt]==1)]["bus"].tolist()
            node_colors = "lightgrey"
            self._draw_nodes(ax=ax1, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list, edge_vals = zip(*[(self.edge_nodes[k], v) for k,v in causal_edges.items()])
        edge_colors = cmap_e(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            ax=ax1,
            width=1.5)

        # show selected line in thick black
        cb = nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=[self.edge_nodes[line_idx]],
            edge_color='k',
            style='-',
            ax=ax1,
            width=3)
        cb.remove()

        # ----- right axis
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title(f"GridFM")# (contingency {scenario})")
        
        # draw nodes, using node_var value color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            nodes = self.node_data[(self.node_data["scenario"]==self.valid_scenarios[0]) & (self.node_data[nt]==1)]["bus"].tolist()
            node_colors = "lightgrey"
            self._draw_nodes(ax=ax2, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list_pred, edge_vals_pred = zip(*[(self.edge_nodes[k], v) for k,v in causal_edges_pred.items()])
        edge_colors_pred = cmap_e(norm_e(edge_vals_pred))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list_pred,
            edge_color=edge_colors_pred,
            ax=ax2,
            width=1.5)

        # show selected line in thick black
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=[self.edge_nodes[line_idx]],
            edge_color='k',
            style='-',
            ax=ax2,
            width=3)
        
        ax1.sharex(ax2)
        ax1.sharey(ax2)
        
        m_ref = plt.scatter([None], [None], marker="s", s=40, color='0.5')
        m_pv = plt.scatter([None], [None], marker="^", s=40, color='0.5')
        m_pq = plt.scatter([None], [None], marker="o", s=40, color='0.5')
        ax1.legend([m_ref, m_pv, m_pq], 
                    ['REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=3)
        ax2.legend([m_ref, m_pv, m_pq], 
                    ['REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=3)

         # show edge colorbar
        sm_e.set_array([])
        plt.colorbar(sm_e, 
                     label=f"Number of causal contingencies affecting line {line_idx}", 
                     ax=ax1,
                     orientation='horizontal', 
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)

        # show node colorbar
        plt.colorbar(sm_e, 
                     label=f"Number of causal contingencies affecting line {line_idx}", 
                     ax=ax2,
                     orientation='horizontal', 
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.tight_layout()
        fig.canvas.header_visible = False
        plt.show()

    
    def plot_all_lines_causal_cases(
            self, 
            edge_var:str, 
            node_var:str,
            pf_ac:bool=True,
            ):
        """
        Plot PandaPower and GridFM graphs showing for each line the total number of 
        violations caused because of that line's contingency across the data.
        """

        # collect all N-1 edges and corresponding line violations for AC solver
        edge_list = []
        num_edge_violations = []
        num_edge_violations_pred = []
        #num_node_violations = []
        #num_node_violations_pred = []
        for cl,cs in self.contingency_scenarios.items():
            if len(cl) == 1:
                cl_df = self.edge_data[self.edge_data.scenario.isin(cs)]
                nev, nev_pred = cl_df[[edge_var+"_violation", edge_var+"_pred"+"_violation"]].sum()
                num_edge_violations.append(nev)
                num_edge_violations_pred.append(nev_pred)
                edge_list.append(self.edge_nodes[cl[0]])

        #cmap_e = plt.get_cmap('jet')
        clrs_e = ["lightgrey", "green", "orange", "purple"]
        cmap_e_name = 'edges_cmap'
        cmap_e = mcolors.LinearSegmentedColormap.from_list(cmap_e_name, clrs_e, N=256)
        vmin_e = np.nanmin([np.nanmin(num_edge_violations),np.nanmin(num_edge_violations_pred)])
        vmax_e = np.nanmin([np.nanmax(num_edge_violations),np.nanmax(num_edge_violations_pred)])
        norm_e = mcolors.Normalize(vmin=vmin_e, vmax=vmax_e)
        sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
        
        
        # ======== Create figure
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, figure=fig)
        plt.clf()

        # ----- left axis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"AC Solver")

        # draw all nodes
        for nt, ns in Grid.node_shapes.items():
            nodes = self.node_data[(self.node_data["scenario"]==self.valid_scenarios[0]) & (self.node_data[nt]==1)]["bus"].tolist()
            node_colors = "lightgrey"
            self._draw_nodes(ax=ax1, nodes=nodes, node_colors=node_colors, nt=nt)

        # draw all edges in grey
        nx.draw_networkx_edges(self.Gb, self.Gb_pos, edge_color="lightgrey", ax=ax1, width=1.5)
        
        # draw edges with statistics
        edge_colors = cmap_e(norm_e(num_edge_violations))
        nx.draw_networkx_edges(self.Gb, self.Gb_pos, edgelist=edge_list, edge_color=edge_colors, ax=ax1, width=1.5)
        

        # ----- right axis
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title(f"GridFM")
        
       # draw all nodes
        for nt, ns in Grid.node_shapes.items():
            nodes = self.node_data[(self.node_data["scenario"]==self.valid_scenarios[0]) & (self.node_data[nt]==1)]["bus"].tolist()
            node_colors = "lightgrey"
            self._draw_nodes(ax=ax2, nodes=nodes, node_colors=node_colors, nt=nt)

        # draw all edges in grey
        nx.draw_networkx_edges(self.Gb, self.Gb_pos, edge_color="lightgrey", ax=ax2, width=1.5)
        
        # draw edges with statistics
        edge_colors_pred = cmap_e(norm_e(num_edge_violations_pred))
        nx.draw_networkx_edges(self.Gb, self.Gb_pos, edgelist=edge_list, edge_color=edge_colors_pred, ax=ax2, width=1.5)
        
        ax1.sharex(ax2)
        ax1.sharey(ax2)
        
        m_ref = plt.scatter([None], [None], marker="s", s=40, color='0.5')
        m_pv = plt.scatter([None], [None], marker="^", s=40, color='0.5')
        m_pq = plt.scatter([None], [None], marker="o", s=40, color='0.5')
        ax1.legend([m_ref, m_pv, m_pq], 
                    ['REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=3)
        ax2.legend([m_ref, m_pv, m_pq], 
                    ['REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=3)

         # show edge colorbar
        sm_e.set_array([])
        plt.colorbar(sm_e, 
                     label=f"Number of violations caused by all line contingencies", 
                     ax=ax1,
                     orientation='horizontal', 
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)

        # show node colorbar
        cb = plt.colorbar(sm_e, 
                     label=f"Number of violations caused by all line contingencies", 
                     ax=ax2,
                     orientation='horizontal', 
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)
        cb.remove()

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.tight_layout()
        fig.canvas.header_visible = False
        plt.show()


    def plot_full_grid_violations( # still to be implemented
            self, 
            edge_var:str, 
            node_var:str,
            pf_ac:bool=True,
            ):
        """
        Plot PandaPower and GridFM graphs of a single (contingency) example, showing line and node variables.
        Line violations for the single example are indicated by red.
        """
        
        # copy node and edge dataframes
        ndf = self.node_data.copy(deep=True)#[self.node_data["PQ"]==1]
        edf = self.edge_data.copy(deep=True)#[]#.copy(deep=True)

        # aggregate line violations across all examples
        sum_cols = [edge_var+'_violation', edge_var+"_pred"+"_violation"]
        keep_cols = ["from_bus", "to_bus"]
        edf = edf.groupby(["edge_idx"]).agg(
            {**{col: 'sum' for col in sum_cols}, **{col: 'first' for col in keep_cols}}).reset_index()
        #edf = edf.groupby(["edge_idx"])[[edge_var+"_violation",edge_var+"_pred"+"_violation"]].sum()
        #edf.reset_index(inplace=True)
        #self.edf = edf

        # aggregate node violations across all examples
        sum_cols = [node_var+"_violation", node_var+"_pred"+"_violation"]
        keep_cols = ["PQ", "PV", "REF"]
        ndf = ndf.groupby("bus").agg(
            {**{col: 'sum' for col in sum_cols}, **{col: 'first' for col in keep_cols}}).reset_index()
        #ndf = ndf.groupby(["bus"])[[node_var+"_violation",node_var+"_pred"+"_violation"]].sum()
        #ndf.reset_index(inplace=True)
        #self.ndf = ndf

        # rank lines and nodes according to total # of violations
        top_n = 20
        # nodes and nodes pred
        nodes_ranked, node_violations_ranked = list(zip(*sorted(
            [(a,b) for a,b in zip(ndf["bus"].tolist(), ndf[node_var+"_violation"].tolist()) if b>0.1], 
            reverse=True, 
            key=lambda x: int(x[1]))[0:top_n]))
        nodes_pred_ranked, node_pred_violations_ranked = list(zip(*sorted(
            [(a,b) for a,b in zip(ndf["bus"].tolist(), ndf[node_var+"_pred"+"_violation"].tolist()) if b>0.1], 
            reverse=True, 
            key=lambda x: int(x[1]))[0:top_n]))
        # lines
        lines_ranked, line_violations_ranked = list(zip(*sorted(
            [(a,b) for a,b in zip(edf["edge_idx"].tolist(), edf[edge_var+"_violation"].tolist()) if b>0.1], 
            reverse=True, 
            key=lambda x: int(x[1]))[0:top_n]))
        lines_pred_ranked, line_pred_violations_ranked = list(zip(*sorted(
            [(a,b) for a,b in zip(edf["edge_idx"].tolist(), edf[edge_var+"_pred"+"_violation"].tolist()) if b>0.1], 
            reverse=True, 
            key=lambda x: int(x[1]))[0:top_n]))
        
        # add small delta to all aggregations for logarithmic colorbar plotting
        ndf[[node_var+"_violation", node_var+"_pred"+"_violation"]] += 0.1
        edf[[edge_var+"_violation", edge_var+"_pred"+"_violation"]] += 0.1

        # set colormaps for edges and nodes
        clrs_ne = ["deepskyblue", "springgreen", "orange", "red"]
        cmap_ne_name = 'ne_cmap'
        cmap_ne = mcolors.LinearSegmentedColormap.from_list(cmap_ne_name, clrs_ne, N=256)
        vmin_n = np.nanmin([np.nanmin(ndf[node_var+"_violation"].tolist()),np.nanmin(ndf[node_var+"_pred"+"_violation"])])
        vmax_n = np.nanmax([np.nanmax(ndf[node_var+"_violation"].tolist()),np.nanmax(ndf[node_var+"_pred"+"_violation"])])
        norm_n = mcolors.LogNorm(vmin=vmin_n+0.001, vmax=vmax_n)
        sm_n = plt.cm.ScalarMappable(cmap=cmap_ne, norm=norm_n)

        vmin_e = np.nanmin([np.nanmin(edf[edge_var+"_violation"].tolist()),np.nanmin(edf[edge_var+"_pred"+"_violation"])])
        vmax_e = np.nanmax([np.nanmax(edf[edge_var+"_violation"].tolist()),np.nanmax(edf[edge_var+"_pred"+"_violation"])])
        norm_e = mcolors.LogNorm(vmin=vmin_e+0.001, vmax=vmax_e)
        sm_e = plt.cm.ScalarMappable(cmap=cmap_ne, norm=norm_e)
        
        
        # ======== Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], figure=fig)
        plt.clf()

        # ----- left axis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"AC Solver")
        
        # draw nodes, using node_var value color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            dfnc_filtered = ndf[ndf[nt]==1.0]
            nodes = dfnc_filtered["bus"].tolist()
            node_vals = dfnc_filtered[node_var+"_violation"].tolist()
            if nt=="PQ":
                node_colors = cmap_ne(norm_n(node_vals)) 
            else:
                node_colors = "lightgray"
            self._draw_nodes(ax=ax1, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list = [(u,v) for u,v in zip(edf.from_bus.tolist(), edf.to_bus.tolist())]
        edge_vals = edf[edge_var+"_violation"].tolist()
        edge_color = cmap_ne(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_color,
            ax=ax1,
            width=1.5)

        # show edge colorbar
        sm_e.set_array([])
        plt.colorbar(sm_e, 
                     label=f"Total line {edge_var} violations", 
                     ax=ax1,
                     orientation='horizontal', 
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)

        # # plot true node ranking
        # ax2 = fig.add_subplot(gs[1, 0])
        # ax2.barh(range(top_n), node_violations_ranked, align="center", zorder=2)
        # ax2.set_yticks(range(top_n), labels=nodes_ranked)
        # ax2.invert_yaxis()
        # ax2.set_title(f"PQ buses ranked by total number of {node_var} violations")
        # ax2.set_xlabel("Number of Violations")
        # ax2.set_ylabel("Ranked Bus IDs")
        # ax2.grid(visible=True, which='major', axis='x', color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)

        # plot true edge ranking
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.barh(range(top_n), line_violations_ranked, align="center", zorder=2)
        ax3.set_yticks(range(top_n), labels=lines_ranked)
        ax3.invert_yaxis()
        ax3.set_title(f"Lines ranked by total number of {edge_var} violations")
        ax3.set_xlabel("Number of Violations")
        ax3.set_ylabel("Ranked Line IDs")
        ax3.grid(visible=True, which='major', axis='x', color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)


        # ----- right axis
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.set_title(f"GridFM")# (contingency {scenario})")
        # ax4.text(0.03, 0.98, 
        #          f'Tot. Load: {self.scenario_load_map[scenario]}kW', 
        #          horizontalalignment='left', 
        #          verticalalignment='top', 
        #          transform=ax4.transAxes)
        
        # draw nodes, using only single color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            dfnc_filtered = ndf[ndf[nt]==1.0]
            nodes = dfnc_filtered["bus"].tolist()
            node_vals = dfnc_filtered[node_var+"_pred"+"_violation"].tolist()
            if nt=="PQ":
                node_colors = cmap_ne(norm_n(node_vals)) 
            else:
                node_colors = "lightgray"
            self._draw_nodes(ax=ax4, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list = [(u,v) for u,v in zip(edf.from_bus.tolist(), edf.to_bus.tolist())]
        edge_vals = edf[edge_var+"_pred"+"_violation"].tolist()
        edge_color = cmap_ne(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_color,
            ax=ax4,
            width=1.5)
        
        # show node colorbar
        sm_n.set_array([])
        plt.colorbar(sm_n, 
                     label=f"Total bus {node_var} violations",
                     ax=ax4,
                     orientation='horizontal',
                     location='bottom',
                     use_gridspec=True,
                     aspect=30,
                     pad=0.05)
        

        # # plot predicted node ranking
        # ax5 = fig.add_subplot(gs[1, 1])
        # ax5.barh(range(top_n), node_pred_violations_ranked, align="center", zorder=2)
        # ax5.set_yticks(range(top_n), labels=nodes_pred_ranked)
        # ax5.invert_yaxis()
        # ax5.set_title(f"PQ buses ranked by total number of {node_var} violations")
        # ax5.set_xlabel("Number of Violations")
        # ax5.set_ylabel("Ranked Bus IDs")
        # ax5.grid(visible=True, which='major', axis='x', color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)

        # plot true edge ranking
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.barh(range(top_n), line_pred_violations_ranked, align="center", zorder=2)
        ax6.set_yticks(range(top_n), labels=lines_pred_ranked)
        ax6.invert_yaxis()
        ax6.set_title(f"Lines ranked by total number of {edge_var} violations")
        ax6.set_xlabel("Number of Violations")
        ax6.set_ylabel("Ranked Line IDs")
        ax6.grid(visible=True, which='major', axis='x', color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)

        ax1.sharex(ax4)
        ax1.sharey(ax4)
        
        m_ref = plt.scatter([None], [None], marker="s", s=40, color='0.5')
        m_pv = plt.scatter([None], [None], marker="^", s=40, color='0.5')
        m_pq = plt.scatter([None], [None], marker="o", s=40, color='0.5')
        ax1.legend([m_ref, m_pv, m_pq], 
                    ['REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=5)
        ax4.legend([m_ref, m_pv, m_pq], 
                    ['REF', 'PV', 'PQ'], 
                    bbox_to_anchor=[0.02, -0.01], loc='lower left', 
                    frameon=False, ncol=5)

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.tight_layout()
        fig.canvas.header_visible = False
        plt.show()
        


    def plot_graph_aggregate(
            self, 
            scenario:int,
            edge_var:str, 
            node_var:str,
            edge_threshold:float,
            node_threshold:float
            ):
        """
        """
        edges_removed = self.edges_removed[scenario]
        #(f"edges_removed: ", edges_removed)

        # get all scenarios for the current contingency (load perturbations)
        contingency_scenarios = sorted(list(set(self.contingency_scenarios[tuple(edges_removed)]).intersection(set(self.valid_scenarios))))
        #print(f"Number of scenarios with lines {','.join([str(x) for x in edges_removed])} dropped: ", len(contingency_scenarios))

        # select from node_data and edge_data dataframes the current contingency
        ndf = self.node_data[self.node_data.scenario.isin(contingency_scenarios)]
        edf = self.edge_data[self.edge_data.scenario.isin(contingency_scenarios)]
        
        # calculate line violations from threshold
        edf[edge_var+"_violation"] = edf[edge_var].apply(lambda x: int(x>edge_threshold) if ~np.isnan(x) else 0)
        edf[edge_var+"_pred"+"_violation"] = edf[edge_var+"_pred"].apply(lambda x: int(x>edge_threshold) if ~np.isnan(x) else 0)

         # calculate violations from threshold
        ndf[node_var+"_violation"] = ndf[node_var].apply(lambda x: int(x>node_threshold) if ~np.isnan(x) else 0)
        ndf[node_var+"_pred"+"_violation"] = ndf[node_var+"_pred"].apply(lambda x: int(x>node_threshold) if ~np.isnan(x) else 0)

        # aggregate violations according to from_bus and to_bus
        edf_agg = edf.groupby(["from_bus", "to_bus"])[[edge_var+"_violation", edge_var+"_pred"+"_violation"]].sum()
        edf_agg.reset_index(inplace=True)
        #self.edf = edf

        # set colormaps for edges and nodes
        #clrs_n = ["lightblue", "red"]
        #cmap_n_name = 'nodes_cmap'
        #cmap_n = mcolors.LinearSegmentedColormap.from_list(cmap_n_name, clrs_n, N=256)
        cmap_n = plt.get_cmap('jet')
        vmin_n = np.nanmin([np.nanmin(ndf[node_var].tolist()),np.nanmin(ndf[node_var+"_pred"])])
        vmax_n = np.nanmax([np.nanmax(ndf[node_var].tolist()),np.nanmax(ndf[node_var+"_pred"])])
        norm_n = mcolors.Normalize(vmin=vmin_n, vmax=vmax_n)
        sm_n = plt.cm.ScalarMappable(cmap=cmap_n, norm=norm_n)
        
        #cmap_e = plt.get_cmap('autumn_r')
        #cmap_e = plt.get_cmap('Wistia')
        clrs = ["green", "orange", "red"]
        cmap_e_name = 'edges_cmap'
        cmap_e = mcolors.LinearSegmentedColormap.from_list(cmap_e_name, clrs, N=256)
        vmin_e = np.nanmin([np.nanmin(edf_agg[edge_var+"_violation"].tolist()),np.nanmin(edf_agg[edge_var+"_pred"+"_violation"])])
        vmax_e = np.nanmax([np.nanmax(edf_agg[edge_var+"_violation"].tolist()),np.nanmax(edf_agg[edge_var+"_pred"+"_violation"])])
        norm_e = mcolors.Normalize(vmin=vmin_e, vmax=vmax_e)
        sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
        #print("Vmin_e, Vmax_e = ", vmin_e, vmax_e)
        
        
        # ======== Create figure
        fig = plt.figure(figsize=(14, 14))
        gs = gridspec.GridSpec(3, 2, height_ratios=[3,1,1],figure=fig)
        plt.clf()

        # -------------------------- left axis
        ax1 = fig.add_subplot(gs[0, 0]) 
        ax1.set_title(f"AC Solver ({scenario})")

        # draw nodes, using only single color and different shapes, sizes
        #self._draw_nodes(ax=ax1)
        # draw nodes, using only single color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            dfnc_filtered = ndf[ndf[nt]==1.0]
            nodes = dfnc_filtered["bus"].tolist()
            node_vals = dfnc_filtered[node_var].tolist()
            node_colors = cmap_n(norm_n(node_vals)) 
            self._draw_nodes(ax=ax1, nodes=nodes, node_colors=node_colors, nt=nt)
        
        # gather edge_list and values, and draw edges
        edge_list = [(u,v) for u,v in zip(edf_agg.from_bus.tolist(), edf_agg.to_bus.tolist())]
        edge_vals = edf_agg[edge_var+"_violation"].tolist()
        edge_color = cmap_e(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_color,
            ax=ax1,
            width=1.5)

        # show contingency lines as dashed/thicker
        con_edge_list = [self.edge_nodes[li] for li in edges_removed]
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=con_edge_list,
            edge_color='k',
            style=':',
            ax=ax1,
            width=2.5)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title(f"All line loadings for line-[{','.join([str(x) for x in edges_removed])}] contingency")
        ax2.hist(edf[edge_var].tolist(), bins=100)
        ax2.axvline(x=edge_threshold, color='r', ls="--", lw=1)
        ax2.set_xlabel("Line loading [100%]")
        ax2.set_ylabel("Frequency")
        ax2.set_yscale("log")

        ax3 = fig.add_subplot(gs[2, 0])
        scenarios_lines_present = sorted(list(set(self.valid_scenarios).difference(set(self.contingency_scenarios[tuple(edges_removed)]))))
        scenarios_lines_present_data = self.edge_data[self.edge_data.scenario.isin(scenarios_lines_present)][edge_var].tolist()
        ax3.set_title(f"Line loadings for line(s)-[{','.join([str(x) for x in edges_removed])}]")
        ax3.hist(scenarios_lines_present_data, bins=100)
        ax3.axvline(x=edge_threshold, color='r', ls="--", lw=1)
        ax3.set_xlabel("Line loading [100%]")
        ax3.set_ylabel("Frequency")
        ax3.set_yscale("log")

        # ----- right axis
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.set_title("GridFM")

        # draw nodes, using only single color and different shapes, sizes
        #self._draw_nodes(ax=ax4)

        # draw nodes, using only single color and different shapes, sizes
        for nt, ns in Grid.node_shapes.items():
            dfnc_filtered = ndf[ndf[nt]==1.0]
            nodes = dfnc_filtered["bus"].tolist()
            node_vals = dfnc_filtered[node_var].tolist()
            node_colors = cmap_n(norm_n(node_vals)) 
            self._draw_nodes(ax=ax4, nodes=nodes, node_colors=node_colors, nt=nt)

        # gather edge_list and values, and draw edges
        edge_list = [(u,v) for u,v in zip(edf_agg.from_bus.tolist(), edf_agg.to_bus.tolist())]
        edge_vals = edf_agg[edge_var+"_pred"+"_violation"].tolist()
        edge_color = cmap_e(norm_e(edge_vals))
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=edge_list,
            edge_color=edge_color,
            ax=ax4,
            width=1.5)
        
        # show contingency lines as dashed/thicker
        con_edge_list = [self.edge_nodes[li] for li in edges_removed]
        nx.draw_networkx_edges(
            self.Gb,
            self.Gb_pos,
            edgelist=con_edge_list,
            edge_color='k',
            style=':',
            ax=ax4,
            width=2.5)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title(f"All line loadings for line-[{','.join([str(x) for x in edges_removed])}] contingency")
        ax5.hist(edf[edge_var+"_pred"].tolist(), bins=100)
        ax5.axvline(x=edge_threshold, color='r', ls="--", lw=1)
        ax5.set_xlabel("Line loading [100%]")
        ax5.set_ylabel("Frequency")
        ax5.set_yscale("log")

        ax6 = fig.add_subplot(gs[2, 1])
        scenarios_lines_present = sorted(list(set(self.valid_scenarios).difference(set(self.contingency_scenarios[tuple(edges_removed)]))))
        scenarios_lines_present_data = self.edge_data[self.edge_data.scenario.isin(scenarios_lines_present)][edge_var+"_pred"].tolist()
        ax6.set_title(f"Line loadings for line(s)-[{','.join([str(x) for x in edges_removed])}]")
        ax6.hist(scenarios_lines_present_data, bins=100)
        ax6.axvline(x=edge_threshold, color='r', ls="--", lw=1)
        ax6.set_xlabel("Line loading [100%]")
        ax6.set_ylabel("Frequency")
        ax6.set_yscale("log")


        # link graph axes
        ax1.sharex(ax4)
        ax1.sharey(ax4)
        
        
        # show edge colorbars
        sm_e.set_array([])
        plt.colorbar(sm_e, 
                     label="Line Violations (freq.)", 
                     ax=ax1,
                     orientation='horizontal', 
                     location='bottom',
                     aspect=40,
                     pad=0.05)

        # show node colorbar
        sm_n.set_array([])
        plt.colorbar(sm_n, 
                     label=self.node_var_labels[node_var],
                     ax=ax4,
                     orientation='horizontal',
                     location='bottom',
                     aspect=45,
                     pad=0.05)

        plt.tight_layout()
        fig.canvas.header_visible = False
        plt.show()
        #return fig
