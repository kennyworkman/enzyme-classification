from Xs_generator_functions import *


batch = read_batch()
labels = list(batch.label)
Types = list(batch.type)
download_raw_structure(labels)
batch_summary = []
Xs=[]
#batch info stores the information of each enzyme, including label, type, number of atoms function for cat(catalysis),
#bind(binding metals, cofactors, ligands, and etc.), het(metals, ligands, cofactors), and struct (structural, no documented function).
#Enzymes that do not have documented site information are removed.

for i_enzyme in range(len(labels)):
    label = labels[i_enzyme]
    Type = Types[i_enzyme]
    struct, HasHet = parse_pdb(label)
    siteseq, siteinfo, HasSite = parse_cif(label)
    if HasSite or HasHet:
        atom_list = atom_list_generator(struct,siteinfo,siteseq,HasSite)
        write_atom_list(atom_list,label)
        print('enzyme:', label, i_enzyme+1,"/", len(labels))
        print("   finish parsing the pdb file: site information acquired")
        atom_list_pocket, center = build_atom_list_pocket(atom_list) #the atom list of pocket
        atom_list_shift = shift_center(atom_list_pocket, center) # shift the atom coordinates, such that they are centered at [0,0,0]
        atom_list_trunc = trunc_box(atom_list_shift) # truncate the atoms into a box
        batch_summary.append([label, Type] + atom_list_summary(atom_list_trunc))
        print("   finish generating the pocket")
        atom_list_grid = grid_atom(atom_list_trunc)
        X = channel_generator(atom_list_grid)
        Xs.append(X)
        write_X(X,label)
        print("   finish generating the 3D atomic space with channels")
    else:
        print('enzyme:', label, i_enzyme+1,"/", len(labels))
        print("   finish parsing the pdb file: no site information")

write_batch_summary(batch_summary)
Xs = np.asarray(Xs)
print("finish generating Xs")
