import os
import pandas as pd
import urllib.request
import numpy as np
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import DBSCAN

#######################################################################

"""

This notebook uses an enzyme label list to generates the Xs for CNN training.

input: enzyme_classification, dtype = .csv. The whole list of enzymes for classification, containing their labels and types

output: batch_summary, dtype = .csv. The list of enzymes which have documented site information, containing their labels, types, and atom space statistics.

"""

#######################################################################

"""

Directory Tree:

-- raw_structure
    1A9Q.cif
    ...
    1A9Q.pdb
    ...

-- atom_list
    1A9Q_atom_list.csv
    ...

-- Xs
    1A9Q.npy
    ...

batch_classification.csv

batch_summary.csv
"""

#######################################################################

####CHANGE TO CURRENT WORKING DIRECTORY
TOP_DIR = '/Users/kenny/Desktop/chem195/enzyme_classifier/Xs_generator'
DATA_DIR = 'raw_structure'
ATOM_DIR = 'atom_list'
Xs_DIR = 'Xs'
ERROR_FILE = '404notfound.txt'
CLASSIFICATION = 'test_classification.csv'

def read_batch():
    #load batch_classfication.csv
    os.chdir(TOP_DIR)
    batch = pd.read_csv(CLASSIFICATION) # Columns are 'label' and 'type'
    return batch

def make_change(name):
    """
    Change into the directory NAME, making it if it does not exist.
    """
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

def write_error(label):
    if not os.path.exists(ERROR_FILE):
        os.system('touch ' + ERROR_FILE)
    with open(ERROR_FILE,'a') as a:
        a.write(label + '\n')

def get_pdbfiles(labels):
    """
    Arg(s): List of PDB labels
    Return(s): None
    Effect(s):
      (1) Import .pdb files for all entries in LABELS, stored in DATA_DIR
      (2) Write PDB labels to ERROR_FILE for entries that could not be found
    """
    os.chdir(TOP_DIR)
    make_change(DATA_DIR) # Change into DATA_DIREC

    count, total = 0, len(labels)
    for label in labels:
        # Print progress
        if count % 10 == 0:
            print('{0}/{1} files have been downloaded'.format(count, total))
        # Download file for LABEL
        label += '.pdb'
        if not os.path.exists(label):   # Don't redownload files that exist
            try:
                urllib.request.urlretrieve('http://files.rcsb.org/download/' + label, label)
            except urllib.error.HTTPError:
                write_error(label)
        count += 1
    print('{0}/{1} files have been downloaded'.format(total, total))
    os.chdir(TOP_DIR)

def get_ciffiles(labels):
    """
    Arg(s): List of PDB labels
    Return(s): None
    Effect(s):
      (1) Import .cif files for all entries in LABELS, stored in DATA_DIR
      (2) Write PDB labels to ERROR_FILE for entries that could not be found
    """
    os.chdir(TOP_DIR)
    make_change(DATA_DIR) # Change into DATA_DIREC

    count, total = 0, len(labels)
    for label in labels:
        # Print progress
        if count % 10 == 0:
            print('{0}/{1} files have been downloaded'.format(count, total))
        # Download file for LABEL
        label += '.cif'
        if not os.path.exists(label):   # Don't redownload files that exist
            try:
                urllib.request.urlretrieve('http://files.rcsb.org/download/' + label, label)
            except urllib.error.HTTPError:
                write_error(label)
        count += 1
    print('{0}/{1} files have been downloaded'.format(total, total))
    os.chdir(TOP_DIR)

def download_raw_structure(labels):
    get_pdbfiles(labels)
    get_ciffiles(labels)

def parse_pdb(label):
    #parse the pdb file and build the enzyme structure
    os.chdir(TOP_DIR)
    os.chdir(DATA_DIR)
    parser=PDB.PDBParser(QUIET=True)
    struct=parser.get_structure(label, label+'.pdb')
    HasHet = False
    for residue in struct[0].get_residues():
        if residue.get_id()[0][0] == "H":
            HasHet = True
    return struct, HasHet

def parse_cif(label):
    #parse the cif file and return the site information if there is any
    mmcif_dict = MMCIF2Dict(label+'.cif')
    if "_struct_site.id" in mmcif_dict: #if there is site info documented in cif file, get the info of cat and bind residues.
        siteseq = mmcif_dict["_struct_site_gen.auth_comp_id"],mmcif_dict["_struct_site_gen.auth_asym_id"],mmcif_dict["_struct_site_gen.auth_seq_id"],mmcif_dict["_struct_site_gen.site_id"]
        siteseq = np.asarray(siteseq)
        siteseq.shape=(4,-1)

        #extract the id of important sites as "siteinfo"
        siteinfo = mmcif_dict["_struct_site.id"],mmcif_dict["_struct_site.details"]
        siteinfo = np.asarray(siteinfo)
        siteinfo.shape=(2,-1)

        HasSite = True

    else:
        siteseq = np.asarray([])
        siteinfo = np.asarray([])
        HasSite = False

    return siteseq, siteinfo, HasSite

def get_cat_bind(siteinfo, siteseq):
    catseq=[[],[],[]]
    bindseq=[[],[],[]]
    #identify those sites that are catalytic sites and store their id as "catid"
    cat_id=[]
    for istring in range(len(siteinfo[1])):
        site_string = siteinfo[1][istring]
        if site_string.find("CATALY") != -1 or site_string.find("cataly") != -1 or site_string.find("Cataly") != -1:
            cat = siteinfo[0][istring]
            cat_id.append(cat)

    #catagorize "siteseq" into catalysis and binding residues, as "catseq" and "bindseq", respectively
    bindseq=[[],[],[]]
    for i in range(len(siteseq[0])):
        if siteseq[3][i] in cat_id:
            for j in range(3):
                catseq[j].append(siteseq[j][i])
        else:
            for j in range(3):
                bindseq[j].append(siteseq[j][i])
    catseq=np.asarray(catseq)
    bindseq=np.asarray(bindseq)
    return catseq, bindseq

def unique_bindseq(catseq, bindseq):
    ubindseq=[[],[],[]]
    #remove catseq and duplicate residues from "bindseq", and store these unique bindseq as "ubindseq"
    seen = catseq.tolist()
    for i in range(len(bindseq[0])):
        dup = False
        for j in range(len(seen[0])):
            if bindseq[0][i] == seen[0][j] and bindseq[1][i] == seen[1][j] and bindseq[2][i] == seen[2][j]:
                dup = True
        if not dup:
            for k in range(3):
                ubindseq[k].append(bindseq[k][i])
                seen[k].append(bindseq[k][i])
    ubindseq=np.asarray(ubindseq)
    return ubindseq

def atom_list_generator(struct,siteinfo,siteseq,HasSite):
    if HasSite:
        catseq, bindseq = get_cat_bind(siteinfo,siteseq)
        ubindseq = unique_bindseq(catseq, bindseq)
    else:
        catseq=[[],[],[]]
        ubindseq=[[],[],[]]
    atom_list = get_atom_list(struct, catseq, ubindseq)
    return atom_list

def get_atom_list(struct, catseq, ubindseq):
    #iterate all residues, decide if each functions as cat, bind, het, or struct and extract its atom info into these catagories
    #cat = catalytic;
    #bind = binding sites that interact with metals,ligands,etc;
    #het = non amino acid residues (cofactors, metals, ligands)
    #struct = no documented function
    cat_atom_list = []
    bind_atom_list = []
    het_atom_list = []
    struct_atom_list =[]
    for residue in struct[0].get_residues():
        becat = False
        bebind = False
        behet = False
        for icat in range(len(catseq[0])): #first examine if the residue belongs to cat
            if residue.get_id()[1] == int(catseq[2][icat]) and residue.get_parent().get_id() == catseq[1][icat] and residue.get_resname() == catseq[0][icat]:
                becat = True
                for atom in residue.get_atoms():
                    #store the info of this atom, including its xyz coordinates, category, and its ID, which provides the entry to read this atom from pdf file.
                    cat_atom_list.append([atom.coord[0],atom.coord[1],atom.coord[2],atom.element,'cat',atom.get_id(),residue.get_id(),residue.get_parent().get_id()])
        if not becat: # then examine if the residue belongs to bind
            for ibind in range(len(ubindseq[0])):
                if residue.get_id()[1] == int(ubindseq[2][ibind]) and residue.get_parent().get_id() == ubindseq[1][ibind] and residue.get_resname() == ubindseq[0][ibind]:
                    if residue.get_id()[0] == ' ' or residue.get_id()[0] == "W":# to exclude non-water het residues
                        bebind = True
                        for atom in residue.get_atoms():
                            bind_atom_list.append([atom.coord[0],atom.coord[1],atom.coord[2],atom.element,'bind',atom.get_id(),residue.get_id(),residue.get_parent().get_id()])
        if (not becat) and (not bebind): # then examine if the residue belongs to het
            if residue.get_id()[0][0] == "H":
                behet = True
                for atom in residue.get_atoms():
                    het_atom_list.append([atom.coord[0],atom.coord[1],atom.coord[2],atom.element,'het',atom.get_id(),residue.get_id(),residue.get_parent().get_id()])
        if (not becat) and (not bebind) and (not behet): # then it must be struct
            for atom in residue.get_atoms():
                struct_atom_list.append([atom.coord[0],atom.coord[1],atom.coord[2],atom.element,'struct',atom.get_id(),residue.get_id(),residue.get_parent().get_id()])

    all_atom_list = cat_atom_list + bind_atom_list + het_atom_list + struct_atom_list
    df_labels = ['x','y','z','element','function','atom id','residue id','chain id']
    atom_list = pd.DataFrame(all_atom_list, columns = df_labels)
    return atom_list

def write_atom_list(atom_list,label):
    os.chdir(TOP_DIR)
    make_change(ATOM_DIR)
    #export the atom info of this enzyme into csv file
    atom_list.to_csv(label+'_atom_list.csv',sep='\t')

def get_atom_pocket(atom_list):
    atom_list_bind = atom_list.loc[atom_list['function'] == 'bind']
    atom_list_cat = atom_list.loc[atom_list['function'] == 'cat']
    atom_list_het = atom_list.loc[atom_list['function'] == 'het']
    atom_list_struct = atom_list.loc[atom_list['function']=='struct']
    atom_list_poc = pd.concat([atom_list_bind,atom_list_cat,atom_list_het])
    return atom_list_poc

def atom_clustering(atom_list, distance_thre=8): # do the clustering of atoms. atoms are clustered if their distance is below distance threshold.
    #return a list of the cluster id of each atom
    neigh_dis = radius_neighbors_graph(atom_list[['x','y','z']], distance_thre, mode='distance')
    db = DBSCAN(eps=8, min_samples=1,metric='precomputed').fit_predict(neigh_dis)
    return db

def pocket_summary(atom_list): # return the summary info of each pocket
    pocket_summary = []
    pocket_num = atom_list.loc[:, 'pocket'].max()+1
    for i in range(pocket_num):
        atom_list_pocket = atom_list.loc[atom_list.loc[:, 'pocket'] == i, :]
        pocket_info = [i] + atom_list_summary(atom_list_pocket)
        pocket_summary.append(pocket_info)
    return pd.DataFrame(pocket_summary, columns=['pocket','cat','bind','het','struct','sum'])

def pocket_selection(pocket_summary):# return the id of the pocket, which has the most number of 'cat' atoms.
    #If pockets have the same number of 'cat' atoms, then compare their number of atoms that are either 'het' ot 'bind'
    pocket_summary= pocket_summary.assign(bindhet = pocket_summary['bind']+pocket_summary['het'])
    pocket_summary_sort = pocket_summary.sort_values(by=['cat','bindhet'], ascending = False)
    return pocket_summary_sort.iloc[0,0]

def atom_list_summary(atom_list):
    cat_num = 0
    bind_num = 0
    het_num = 0
    struct_num = 0
    for j in range(len(atom_list)):
        if atom_list.iloc[j,4] == 'cat':
            cat_num+=1
        else:
            if atom_list.iloc[j,4] == 'bind':
                bind_num+=1
            else:
                if atom_list.iloc[j,4] == 'het':
                    het_num+=1
                else:
                    if atom_list.iloc[j,4] == 'struct':
                        struct_num+=1
    sum_num = cat_num+bind_num+het_num+struct_num
    return [cat_num,bind_num,het_num,struct_num,sum_num]

def atom_list_summary(atom_list):
    cat_num = 0
    bind_num = 0
    het_num = 0
    struct_num = 0
    for j in range(len(atom_list)):
        if atom_list.iloc[j,4] == 'cat':
            cat_num+=1
        else:
            if atom_list.iloc[j,4] == 'bind':
                bind_num+=1
            else:
                if atom_list.iloc[j,4] == 'het':
                    het_num+=1
                else:
                    if atom_list.iloc[j,4] == 'struct':
                        struct_num+=1
    sum_num = cat_num+bind_num+het_num+struct_num
    return [cat_num,bind_num,het_num,struct_num,sum_num]

def calc_center(atom_list): # return the cooridnate of the center of the atom space
    x_center = atom_list['x'].mean()
    y_center = atom_list['y'].mean()
    z_center = atom_list['z'].mean()
    center = [x_center, y_center, z_center]
    return center

def get_atom_center_distance(atom_list, i, center): # return the Euclidean distance between the atom and the center
    x_distance = atom_list.iloc[i,0] - center[0]
    y_distance = atom_list.iloc[i,1] - center[1]
    z_distance = atom_list.iloc[i,2] - center[2]
    E_distance = np.sqrt(x_distance**2 + y_distance**2 + z_distance**2)
    return E_distance

def add_atom_struct_in_pocket(atom_list, atom_list_pocket, pocket_radius=16): # atom 'struct' atoms into the pocket,
    #which are located within a threshold distance to the center of the pocket.
    atom_list_struct = atom_list.loc[atom_list['function']=='struct']
    atom_index=[]
    center = calc_center(atom_list_pocket)
    for i in range(len(atom_list_struct)):
        if get_atom_center_distance(atom_list_struct, i, center) < pocket_radius:
            atom_index.append(i)
    atom_list_pocket_struct = atom_list_struct.iloc[atom_index, :]
    pocket_id = atom_list_pocket.iloc[0,8]
    pocket_array = np.full(len(atom_list_pocket_struct),pocket_id)
    atom_list_pocket_struct=atom_list_pocket_struct.assign(pocket = pocket_array)
    atom_list_pocket_full = pd.concat([atom_list_pocket,atom_list_pocket_struct])
    return atom_list_pocket_full,center

def build_atom_list_pocket(atom_list):
    atom_list_poc = get_atom_pocket(atom_list)
    pocket_idlist = atom_clustering(atom_list_poc) #perform clustering of 'poc' atoms, identify the best pocket
    atom_list_poc['pocket'] = pocket_idlist
    pocket_summary_info = pocket_summary(atom_list_poc)
    pocket_id = pocket_selection(pocket_summary_info)
    atom_list_pocket =atom_list_poc.loc[atom_list_poc.loc[:,'pocket'] == pocket_id, :] #build an atom list of those 'poc' atoms in the best pocket
    atom_list_pocket, center = add_atom_struct_in_pocket(atom_list, atom_list_pocket) # add 'struct' atoms that are close to the pocket center into the pocket atom list. export pocket_atom_list
    return atom_list_pocket, center

def shift_center(atom_list,center): #shift the coordinate of atoms be a vector: -center
    atom_list.loc[:, 'x'] -= center[0]
    atom_list.loc[:, 'y'] -= center[1]
    atom_list.loc[:, 'z'] -= center[2]
    return atom_list

def trunc_box(atom_list, size=16): # truncate atom space into a box with the set size. remove atoms beyond the box space
    atom_index = []
    for i in range(len(atom_list)):
        if abs(atom_list.iloc[i,0]) < size and abs(atom_list.iloc[i,1]) < size and abs(atom_list.iloc[i,2]) < size:
            atom_index.append(i)
    atom_list_trunc = atom_list.iloc[atom_index, :]
    return atom_list_trunc

def grid_atom(atom_list, v_size=64, grid_distance=0.5):
    atom_list_centered = shift_center(atom_list, [-v_size*grid_distance/2,-v_size*grid_distance/2,-v_size*grid_distance/2])
    atom_list_grid = atom_list_centered.copy()
    for j in range(len(atom_list_centered)):
        x = float(atom_list_centered.iloc[j,0] // grid_distance)
        y = float(atom_list_centered.iloc[j,1] // grid_distance)
        z = float(atom_list_centered.iloc[j,2] // grid_distance)
        atom_list_grid.iloc[j,0] = x
        atom_list_grid.iloc[j,1] = y
        atom_list_grid.iloc[j,2] = z
    return atom_list_grid

def locate_channel_function(atom):
    if atom['function'] == 'cat':
        channel = 0
    else:
        if atom['function'] == 'bind':
            channel = 1
        else:
            if atom['function'] == 'het':
                channel = 2
            else:
                channel =3
    return channel

def locate_channel_element(atom):
    channel = -1
    if atom['element'] == 'C':
        channel = 4
    else:
        if atom['element'] == 'N':
            channel = 5
        else:
            if atom['element'] == 'O':
                channel = 6
            else:
                if atom['element'] == ('S' or 'SE'):
                    channel = 7
                else:
                    if atom['element'] == 'P':
                        channel = 8
                    if atom['element'] == ('CL' or 'BR' or 'F' or 'I'):
                        channel = 9
                    if atom['element'] == ('LI' or ' NA' or 'K' or 'CS' or 'RB'):
                        channel = 10
                    if atom['element'] == ('MG' or ' CA' or 'SR' or 'BA' or 'AL' or 'B' or 'TI' or 'ZR'):
                        channel = 11
                    if atom['element'] == 'ZN':
                        channel = 12
                    if atom['element'] == 'CU':
                        channel = 13
                    if atom['element'] == ('FE' or 'CO' or 'Ni'):
                        channel = 14
                    if atom['element'] == ('MN' or 'Cr' or 'V' or 'MO' or 'NB'):
                        channel = 15
                    if atom['element'] == ('RU' or 'RH' or 'PD' or 'AG' or 'W' or 'RE' or 'OS' or 'IR' or 'PT' or 'AU'):
                        channel = 16
                    if channel == -1:
                        channel = 17
    return channel

def channel_generator (atom_list, v_size=64, grid_distance = 0.5):
    X = np.zeros((v_size,v_size,v_size,18))
    for j in range(len(atom_list)):
        atom = atom_list.iloc[j,:]
        channel_function = locate_channel_function(atom)
        channel_element = locate_channel_element(atom)
        X[int(atom['x']), int(atom['y']), int(atom['z']), channel_function] = 1
        X[int(atom['x']), int(atom['y']), int(atom['z']), channel_element] = 1
    return X

def write_X(X,label):
    os.chdir(TOP_DIR)
    make_change(Xs_DIR)
    np.save(label+'.npy', X)

def write_batch_summary(batch_summary):
    df_labels = ['label','type','cat','bind','het','struct','sum']
    batch_summary = pd.DataFrame(batch_summary, columns =df_labels)
    os.chdir(TOP_DIR)
    batch_summary.to_csv('batch_summary.csv',sep='\t')
