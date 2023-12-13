import sys
# sys.path.insert(0,'/eos/home-a/antoniov/SWAN_projects/env/uproot-py39/lib/python3.9/site-packages')
# print ( sys.path )
import uproot
import h5py
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(
    {'font.size': 18,
     'font.family': 'sans-serif',
     'legend.fontsize': 14,
     'axes.labelsize': 22,
     'axes.labelpad': 8.0,
     'xtick.labelsize': 14,
     'ytick.labelsize': 14
    }
    )
print ( mpl.rcParams )
import mplhep
plt.style.use(mplhep.style.CMS)

def create_dataframes( fileName, tree_name="T" ):
    
    root = uproot.open( fileName )
    tree = root[ tree_name ]
    print ( tree.keys() )
    
    events = tree.arrays( tree.keys() , library="ak", how="zip" )
    events[ "event_number" ] = ( np.arange( len(events) ) + 1 )
    
    particles = events[ "part" ]
    particles["event_number"] = events.event_number
    jets = events[ "jet" ]
    jets["event_number"] = events.event_number
    
    arr_particles = ak.flatten( particles, axis=1 )
    df_part = pd.DataFrame( arr_particles.to_numpy() )
    
    arr_jets = ak.flatten( jets, axis=1 )
    df_j = pd.DataFrame( arr_jets.to_numpy() ).rename( columns={ 'idx': 'jet_idx' } )
    
    df_part_in_j = df_part[ df_part.jet_idx >= 0 ].set_index( ["event_number", "jet_idx"] )
    df_j = df_j.set_index( ["event_number", "jet_idx"] )
    df_part_in_j = df_part_in_j.join( df_j, rsuffix="_jet" )
    
    return df_part, df_j, df_part_in_j

def df_tag_sorted_groupby( df_part_in_j, df_part, df_j, tagging, number_of_const ):
    
    # df_part_in_j=df_particles_in_jet
    # df_j=df_jets
   
    # df_particles_in_jets = df_part[ df_part.jet_idx >= 0 ].set_index( ["event_number", "jet_idx"] )
    # df_particles_in_jets = df_particles_in_jets.join( df_j, rsuffix="_jet" )
    
    if tagging == 'W':
        df_particles_in_jets_tag = df_part_in_j.loc[ df_part_in_j.Wtag == 1 ]
        df_particles_in_jets_notag = df_part_in_j.loc[ df_part_in_j.Wtag == 0 ]
    if tagging == 'QCD':
        df_particles_in_jets_tag = df_part_in_j.loc[ df_part_in_j.Qtag == 1 ]
        df_particles_in_jets_notag = df_part_in_j.loc[ df_part_in_j.Qtag == 0 ]
    if tagging == 't':
        df_particles_in_jets_tag = df_part_in_j.loc[ df_part_in_j.Ttag == 1 ]
        df_particles_in_jets_notag = df_part_in_j.loc[ df_part_in_j.Ttag == 0 ]
        
    df_particles_in_jets_tag_groupby = df_particles_in_jets_tag.reset_index().groupby( ["event_number", "jet_idx"] )
    df_particles_in_jets_notag_groupby = df_particles_in_jets_notag.reset_index().groupby( ["event_number", "jet_idx"] )

    if tagging == 'QCD': 
        df_particles_in_jets_tag_sorted = df_particles_in_jets_notag_groupby.apply( lambda df: df.sort_values('pt', ascending=False).head( number_of_const ) ).drop( columns=["event_number", "jet_idx"] ).reset_index().set_index( ["event_number", "jet_idx"] )
    else:
        df_particles_in_jets_tag_sorted = df_particles_in_jets_tag_groupby.apply( lambda df: df.sort_values('pt', ascending=False).head( number_of_const ) ).drop( columns=["event_number", "jet_idx"] ).reset_index().set_index( ["event_number", "jet_idx"] )
        df_particles_in_jets_notag_sorted = df_particles_in_jets_notag_groupby.apply( lambda df: df.sort_values('pt', ascending=False).head( number_of_const ) ).drop( columns=["event_number", "jet_idx"] ).reset_index().set_index( ["event_number", "jet_idx"] )
        
    df_particles_in_jets_tag_sorted_groupby = df_particles_in_jets_tag_sorted.groupby( ["event_number", "jet_idx"] )
    
    return df_particles_in_jets_tag_sorted, df_particles_in_jets_tag_sorted_groupby

def df_filling(df, number_of_const):
    
    # df=df_particles_in_jets_tag_sorted_groupby
    
    df_count = df.count()
    df_count_notfull = df_count.loc[ df_count['id'] != number_of_const ]
    
    df_count_tofill = df_count_notfull['id'].rename({'id':'count'})
    df_count_tofill = number_of_const - df_count_tofill
    
    df_particles_tofill = df_count_notfull.loc[ df_count_notfull.index.repeat( df_count_tofill ) ]
    df_particles_tofill.loc[:,:] = 0 
    
    return df_particles_tofill

def df_padding(df_tag_sorted, df_fill):
    
    # df_tag_sorted=df_particles_in_jets_tag_sorted
    # df_fill=df_particles_tofill
    
    df_particles_in_jets_tag_sorted_padded = pd.concat( [ df_tag_sorted, df_fill ] ).sort_values( ["event_number", "jet_idx"] )
    
    df_count_padded = df_particles_in_jets_tag_sorted_padded.groupby( ["event_number", "jet_idx"] ).count()
    
    return df_particles_in_jets_tag_sorted_padded, df_count_padded

def df_to_numpy(df_padded, df_count, number_of_const):
    
    # df_padded=df_particles_in_jets_tag_sorted_padded
    # df_count=df_count_padded
    
    columns_part = [ 'pt','eta','phi','energy']
    columns = [ 'pt','eta','phi','energy', 'pt_jet', "eta_jet", 'phi_jet', 'energy_jet' ]
    
    arr_part_tag_sorted_padded = df_padded[ columns_part ].to_numpy()
    arr_data_tag_sorted_padded = df_padded[ columns ].to_numpy()
    
    arr_data_tag_sorted_padded_reshaped = arr_data_tag_sorted_padded.reshape( len(df_count), number_of_const, len(columns) )
    arr_part_tag_sorted_padded_reshaped = arr_part_tag_sorted_padded.reshape( len(df_count), number_of_const, len(columns_part) )
    arr_jet_tag_sorted_padded_reshaped = arr_data_tag_sorted_padded_reshaped.reshape(len(df_count), -1)[:, 4:8]
    
    return arr_part_tag_sorted_padded_reshaped, arr_jet_tag_sorted_padded_reshaped

def to_h5py(arr_part, arr_jet, tagging, outputName):
    
    # arr_part=arr_part_tag_sorted_padded_reshaped
    # arr_jet=arr_jet_tag_sorted_padded_reshaped
    
    if tagging == 'W':
        with h5py.File( outputName, 'w' ) as hf:
            hf.create_dataset("particle_features",  data=arr_part)
            hf.create_dataset("jet_features",  data=arr_jet)
    if tagging == 'QCD':
        with h5py.File( outputName, 'w' ) as hf:
            hf.create_dataset("particle_features",  data=arr_part)
            hf.create_dataset("jet_features",  data=arr_jet)
    if tagging == 't':
        with h5py.File( outputName, 'w' ) as hf:
            hf.create_dataset("particle_features",  data=arr_part)
            hf.create_dataset("jet_features",  data=arr_jet)
            
    return hf

    