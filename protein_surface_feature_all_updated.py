import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFreeSASA
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

# folder_path = './PDBfiles/'  # 替换为您的文件夹路径
# folder_path = './PDBfiles-20240920/'  # 替换为您的文件夹路径
folder_path = './PDBfiles-20240924/'  # 替换为您的文件夹路径
pdb_file_list = []

for filename in os.listdir(folder_path):
    if filename.endswith('.pdb'):
        pdb_file_list.append(filename)

print(len(pdb_file_list))
types = []
_l = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
for i in range(100):
    types.append(_l[i // 10] + _l[i % 10])

vis_p = False
ph = 7.5
pka = {'CYS': 8.18, 'ASP': 3.65, 'GLU': 4.25, 'LYS': 10.53, 'ARG': 12.48, 'HIS': 6.00}
# amino acid side chain hydrophobicity
shb = {
    'TYR': 0.96,  # kcal per mol
    'TRP': 2.25,
    'PHE': 1.79,
    'ARG': -1.01,
    'LYS': -0.99,
    'HIS': 0.13,
    'PRO': 0.72,
    'GLY': 0.0,
    'ALA': 0.31,
    'SER': -0.04,
    'CYS': 1.54,
    'MET': 1.23,
    'VAL': 1.22,
    'LEU': 1.70,
    'ILE': 1.80,
    'THR': 0.26,
    'GLN': -0.22,
    'ASN': -0.6,
    'GLU': -0.64,
    'ASP': -0.77
}

neg_shb = {
    'Ile': 	-1.56,
    'Val': 	-0.78,
    'Leu': 	-1.81,
    'Phe': 	-2.20,	
    'Cys': 	0.49,
    'Met': 	-0.76,
    'Ala': 	0.0,
    'Gly': 	1.72,
    'Thr': 	1.78,
    'Ser': 	1.83,
    'Trp': 	-0.38,
    'Tyr': 	-1.09,
    'Pro': 	-1.52,
    'His': 	4.76,
    'Glu':  1.64,
    'Gln': 	3.01,
    'Asp': 	2.95,
    'Asn': 	3.47,
    'Lys': 	5.39,
    'Arg': 	3.71,
}

for key in shb:
    shb[key] = 4120 * shb[key] / 8.314 / 310.  # in kBT

p_table = Chem.GetPeriodicTable()


protein_mass = []
gene_name = []
delta_list = []

total_charge = []
total_num_pos = []
total_area_pos = []
total_charge_pos = []
total_residue_pos = []
max_area_pos = []
min_area_pos = []
max_charge_pos = []
min_charge_pos = []

total_num_neg = []
total_area_neg = []
total_charge_neg = []
total_residue_neg = []
max_area_neg = []
min_area_neg = []
max_charge_neg = []
min_charge_neg = []

total_num_neutral = []
total_area_neutral = []
total_charge_neutral = []
total_residue_neutral = []
max_area_neutral = []
min_area_neutral = []
max_charge_neutral = []
min_charge_neutral = []

pdb_dt = pd.DataFrame({})
fails = []
for per_pdb in pdb_file_list:

    #print(per_pdb)
    if per_pdb.split('_')[0].split('.')[0] in gene_name:
        continue

    if 'ALDOC' in per_pdb:
        print('+++++')
    rdmol: Chem.Mol = Chem.MolFromPDBFile(folder_path + per_pdb, removeHs=False, sanitize=False)
    if 'ALDOC' not in per_pdb:
        try: 
            sub_mols = Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
            _mols = [mol for mol in sub_mols if mol.GetNumAtoms() > 200]
            if False:
                proteins = Chem.CombineMols(_mols[0], _mols[1])
                for i in range(2, len(_mols)):
                    proteins = Chem.CombineMols(proteins, _mols[i])
            else:
                proteins = _mols[0]
            Chem.SanitizeMol(proteins)
            rdmol = proteins
            #print(f"Molecules lesser than 10 atoms will be removed.")
        except:
            fails.append(per_pdb)
            continue
            #print(per_pdb, "failed")




    res_chg = []
    radii = []
    res_idx = []
    res_idx_name = {}
    atom: Chem.Atom

    res_atom_num = {}
    if not rdmol:
        continue

    #print()
    #print(per_pdb)
    gene_name.append(per_pdb.split('_')[0].split('.')[0])
    #print(gene_name)
    for atom in rdmol.GetAtoms():
        pr: Chem.AtomPDBResidueInfo = atom.GetPDBResidueInfo()
        if res_atom_num.get(pr.GetResidueNumber()) is None:
            res_atom_num[pr.GetResidueNumber()] = 0
        res_atom_num[pr.GetResidueNumber()] += 1

    mass_ = 0

    for atom in rdmol.GetAtoms():
        pr: Chem.AtomPDBResidueInfo = atom.GetPDBResidueInfo()
        pka_ = pka.get(pr.GetResidueName())
        if not pka_:
            chg = 0
        else:
            pn = np.sign(pka_ - ph)
            chg = pn / (1 + 10 ** (pn * (ph - pka_))) / res_atom_num[pr.GetResidueNumber()]
            mass_ += atom.GetMass()
        res_chg.append(chg)
        radii.append(p_table.GetRvdw(atom.GetAtomicNum()))
        res_idx.append(pr.GetResidueNumber())
        res_idx_name[pr.GetResidueNumber()] = pr.GetResidueName()

    res_chg = np.array(res_chg)
    res_idx = np.array(res_idx)
    total_charge.append(res_chg.sum())
    sasa_all = rdFreeSASA.CalcSASA(rdmol, radii)
    sasa = np.asarray([float(_.GetProp("SASA")) for _ in rdmol.GetAtoms()])

    x_ = rdmol.GetConformer(0).GetPositions()
    # Get surface atom positions
    surf_lb = sasa > 3.
    surf_idx = np.arange(x_.shape[0])[surf_lb]
    surf_chg = res_chg[surf_lb]
    surf_sa = sasa[surf_lb]
    x_surf = x_[surf_lb]
    res_idx_surf = res_idx[surf_lb]
    delta = 0.
    cnt_ = 0.
    for res_id in res_idx_surf:
        res_name = res_idx_name.get(res_id)
        if shb.get(res_name) is not None:
            delta += shb[res_name]
        cnt_ += 1.
    delta_list.append(delta/cnt_ * (mass_/18.) ** (2/3.))
    #print(f"Hydrophobicity of protein is {delta/cnt_:.4f} k_BT")

    for _i in range(3):
        if _i == 0:
            _idx = surf_chg > 0
            tar = 'positive'
        elif _i == 1:
            _idx = surf_chg < 0
            tar = 'negative'
        else:
            _idx = np.isclose(surf_chg, 0)
            tar = 'neutral'
        x_ch = x_surf[_idx]
        surf_sa_ch = surf_sa[_idx]
        res_idx_surf_ch = res_idx_surf[_idx]
        surf_chg_ch = surf_chg[_idx]
        surf_idx_ch = surf_idx[_idx]
        gr, r_ = np.histogram(distance.pdist(x_ch), bins=50, range=(0, 20))
        gr[0] = 0
        gr = gr / np.diff(4 * np.pi * r_ ** 3)
        rc = r_[np.argmax(gr)]
        clf = DBSCAN(eps=rc * 4., min_samples=3)
        clf.fit(x_ch)
        #print(f'Total num of patches of {tar} is: {len(set(clf.labels_) - {-1}):d}')
        #print(f"Total num of (noise) atoms of {tar} is: {clf.labels_.shape[0]} ({np.sum(clf.labels_ == -1)})")
        total_area = []
        for lb in set(clf.labels_) - {-1}:
            total_area.append(np.sum(surf_sa_ch[clf.labels_ == lb]))
        total_chg = []
        for lb in set(clf.labels_) - {-1}:
            total_chg.append(np.sum(surf_chg_ch[clf.labels_ == lb]))
        #print(f'Total area of patches of {tar} is: {np.sum(total_area) / 100:.4f} nm^2')
        #print(f"Total charge of patches of {tar} is: {np.sum(total_chg):.4f} e.")
        #print(f"Max/Min area/charge of patches of {tar} are "
        #      f"{np.max(total_area) / 100:.4f} {np.min(total_area) / 100:.4f} nm^2 "
        #      f"{total_chg[np.argmax(total_area)]:.4f} {total_chg[np.argmin(total_area)]:.4f} e.")
        _res_idx = set()
        for lb in set(clf.labels_) - {-1}:
            for _id in res_idx_surf_ch[clf.labels_ == lb]:
                _res_idx.add(_id)
        #print(f"Total num of residues of {tar} is: {len(_res_idx):d}")
        if vis_p:
            o = open(f"debug_xyz_{tar}.xyz", 'w')
            o.write(f'{np.sum(clf.labels_ != -1)}\nMeta\n')
            for lb in set(clf.labels_) - {-1}:
                pos = x_ch[clf.labels_ == lb]
                for p in pos:
                    o.write(f'{types[lb]} {p[0]} {p[1]} {p[2]}\n')
            o.close()


        if _i == 0:
            total_num_pos.append(len(set(clf.labels_) - {-1}))
            total_area_pos.append(np.sum(total_area) / 100)
            total_charge_pos.append(np.sum(total_chg))
            total_residue_pos.append(len(_res_idx))
            max_area_pos.append(np.max(total_area) / 100)
            min_area_pos.append(np.min(total_area) / 100)
            max_charge_pos.append(total_chg[np.argmax(total_area)])
            min_charge_pos.append(total_chg[np.argmin(total_area)])
        elif _i == 1:
            total_num_neg.append(len(set(clf.labels_) - {-1}))
            total_area_neg.append(np.sum(total_area) / 100)
            total_charge_neg.append(np.sum(total_chg))
            total_residue_neg.append(len(_res_idx))
            max_area_neg.append(np.max(total_area) / 100)
            min_area_neg.append(np.min(total_area) / 100)
            max_charge_neg.append(total_chg[np.argmax(total_area)])
            min_charge_neg.append(total_chg[np.argmin(total_area)])
        else:
            total_num_neutral.append(len(set(clf.labels_) - {-1}))
            total_area_neutral.append(np.sum(total_area) / 100)
            total_charge_neutral.append(np.sum(total_chg))
            total_residue_neutral.append(len(_res_idx))
            max_area_neutral.append(np.max(total_area) / 100)
            min_area_neutral.append(np.min(total_area) / 100)
            max_charge_neutral.append(total_chg[np.argmax(total_area)])
            min_charge_neutral.append(total_chg[np.argmin(total_area)])

    protein_mass.append((mass_/18)**(2./3))
    #print(f"Chi_N for effective chain {delta / cnt_ * (mass_/18) ** (2./3):.4f} kBT")

pdb_dt['gene'] = gene_name
pdb_dt['MV'] = protein_mass
pdb_dt['zi'] = total_charge
pdb_dt['delta'] = delta_list
pdb_dt['total_num_pos'] = total_num_pos
pdb_dt['total_area_pos'] = total_area_pos
pdb_dt['total_charge_pos'] = total_charge_pos
pdb_dt['total_residue_pos'] = total_residue_pos
pdb_dt['max_area_pos'] = max_area_pos
pdb_dt['min_area_pos'] = min_area_pos
pdb_dt['max_charge_pos'] = max_charge_pos
pdb_dt['min_charge_pos'] = min_charge_pos
pdb_dt['total_num_neg'] = total_num_neg
pdb_dt['total_area_neg'] = total_area_neg
pdb_dt['total_charge_neg'] = total_charge_neg
pdb_dt['total_residue_neg'] = total_residue_neg
pdb_dt['max_area_neg'] = max_area_neg
pdb_dt['min_area_neg'] = min_area_neg
pdb_dt['max_charge_neg'] = max_charge_neg
pdb_dt['min_charge_neg'] = min_charge_neg
pdb_dt['total_num_neutral'] = total_num_neutral
pdb_dt['total_area_neutral'] = total_area_neutral
pdb_dt['total_charge_neutral'] = total_charge_neutral
pdb_dt['total_residue_neutral'] = total_residue_neutral
pdb_dt['max_area_neutral'] = max_area_neutral
pdb_dt['min_area_neutral'] = min_area_neutral
pdb_dt['max_charge_neutral'] = max_charge_neutral
pdb_dt['min_charge_neutral'] = min_charge_neutral

print(pdb_dt)

# pdb_dt.to_csv('protein_surface_data.csv', index=False)
pdb_dt.to_csv('protein_surface_data_20250101.csv', index=False)
# pdb_dt.to_csv('protein_surface_data_20250101-2.csv', index=False)
print(fails)
print(pdb_dt['delta'].mean(), pdb_dt['delta'].std())
