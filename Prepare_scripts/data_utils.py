import importlib
import re
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pprint import pformat, pprint
import contextlib
import contextlib
import numpy as np
import torch as th
from distutils.util import strtobool
import joblib
import tqdm
import os
import errno
import signal
from functools import wraps
from timeit import default_timer as timer
from datetime import timedelta
import sys
import pandas as pd


def none_or_nan(thing):
    if thing is None:
        return True
    elif isinstance(thing, float) and np.isnan(thing):
        return True
    elif pd.isnull(thing):
        return True
    else:
        return False

@contextlib.contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

# ELEMENT_LIST = ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F']
ELEMENT_LIST = ['F', 'Hg', 'Cl', 'Pt', 'As', 'I', 'Co', 'C', 'Se', 'Gd', 'Au', 'Si', 'H', 'P', 'V', 'O', 'T', 'Sb', 'Cu', 'Sn', 'Ag', 'N', 'Cr', 'S', 'B', 'Fe', 'Br']
TWO_LETTER_TOKEN_NAMES = [
    'Al',
    'Ce',
    'Co',
    'Ge',
    'Gd',
    'Cs',
    'Th',
    'Cd',
    'As',
    'Na',
    'Nb',
    'Li',
    'Ni',
    'Se',
    'Sc',
    'Sb',
    'Sn',
    'Hf',
    'Hg',
    'Si',
    'Be',
    'Cl',
    'Rb',
    'Fe',
    'Bi',
    'Br',
    'Ag',
    'Ru',
    'Zn',
    'Te',
    'Mo',
    'Pt',
    'Mn',
    'Os',
    'Tl',
    'In',
    'Cu',
    'Mg',
    'Ti',
    'Pb',
    'Re',
    'Pd',
    'Ir',
    'Rh',
    'Zr',
    'Cr',
    '@@',
    'se',
    'si',
    'te']

LC_TWO_LETTER_MAP = {
    "se": "Se", "si": "Si", "te": "Te"
}

# these are all (exact) atomic masses
H_MASS = 1.007825  # 1.008
O_MASS = 15.994915  # 15.999
NA_MASS = 22.989771  # 22.990
N_MASS = 14.003074  # 14.007
C_MASS = 12.  # 12.011

JOBLIB_BACKEND = "loky" 
JOBLIB_N_JOBS = joblib.cpu_count() 
JOBLIB_TIMEOUT = None  


def rdkit_import(*module_strs): 
 
    RDLogger = importlib.import_module("rdkit.RDLogger") 
    RDLogger.DisableLog('rdApp.*') 
    modules = []
    for module_str in module_strs:
        modules.append(importlib.import_module(module_str))
    return tuple(modules) 


def normalize_ints(ints):

    total_ints = sum(ints)
    ints = [ints[i] / total_ints for i in range(len(ints))]
    return ints


def randomize_smiles(smiles, rseed, isomeric=False, kekule=False):
    """Perform a randomization of a SMILES string must be RDKit sanitizable"""
    if rseed == -1:
        return smiles
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    m = Chem.MolFromSmiles(smiles)
    assert not (m is None)
    ans = list(range(m.GetNumAtoms()))
    with np_temp_seed(rseed):
        np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(
        nm,
        canonical=False,
        isomericSmiles=isomeric,
        kekuleSmiles=kekule)
    assert not (smiles is None)
    return smiles


def split_smiles(smiles_str):

    token_list = []
    ptr = 0

    while ptr < len(smiles_str):
        if smiles_str[ptr:ptr + 2] in TWO_LETTER_TOKEN_NAMES:
            smiles_char = smiles_str[ptr:ptr + 2]
            if smiles_char in LC_TWO_LETTER_MAP:
                smiles_char = LC_TWO_LETTER_MAP[smiles_char]
            token_list.append(smiles_char)
            ptr += 2
        else:
            smiles_char = smiles_str[ptr]
            token_list.append(smiles_char)
            ptr += 1

    return token_list


def list_replace(l, d):
    return [d[data] for data in l]


def mol_from_inchi(inchi):
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        mol = Chem.MolFromInchi(inchi)
    except BaseException:
        mol = np.nan
    if none_or_nan(mol):
        mol = np.nan
    return mol


def rdkit_standardize(mol):

    modules = rdkit_import(
        "rdkit.Chem",
        "rdkit.Chem.MolStandardize",
        "rdkit.Chem.MolStandardize.rdMolStandardize")
    rdMolStandardize = modules[-1]
    mol = rdMolStandardize.Cleanup(mol)
    te = rdMolStandardize.TautomerEnumerator()
    mol = te.Canonicalize(mol)
    return mol


def mol_from_smiles(smiles, standardize=True):

    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if standardize:
            mol = rdkit_standardize(mol)
    except BaseException:
        mol = np.nan
    if none_or_nan(mol):
        mol = np.nan
    return mol


def mol_to_smiles(
        mol,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False): 
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        smiles = Chem.MolToSmiles(
            mol,
            canonical=canonical,
            isomericSmiles=isomericSmiles,
            kekuleSmiles=kekuleSmiles)
    except BaseException:
        smiles = np.nan
    return smiles


def mol_to_formula(mol):

    modules = rdkit_import("rdkit.Chem.AllChem") 
    AllChem = modules[0]
    try:
        formula = AllChem.CalcMolFormula(mol)
    except BaseException:
        formula = np.nan
    return formula


def mol_to_inchikey(mol):
    modules = rdkit_import("rdkit.Chem.inchi")
    inchi = modules[0]
    try:
        inchikey = inchi.MolToInchiKey(mol)
    except BaseException:
        inchikey = np.nan
    return inchikey


def mol_to_inchikey_s(mol):

    modules = rdkit_import("rdkit.Chem.inchi")
    inchi = modules[0]
    try:
        inchikey = inchi.MolToInchiKey(mol)
        inchikey_s = inchikey[:14]
    except BaseException:
        inchikey_s = np.nan
    return inchikey_s


def mol_to_inchi(mol):
  
    modules = rdkit_import("rdkit.Chem.rdinchi")
    rdinchi = modules[0]
    try:
        
        inchi = rdinchi.MolToInchi(mol, options='-SNon')[0] 
    except BaseException:
        inchi = np.nan
    return inchi


def mol_to_mol_weight(mol, exact=True):
    
    modules = rdkit_import("rdkit.Chem.Descriptors")
    Desc = modules[0]
    try:
        if exact:
            mol_weight = Desc.ExactMolWt(mol)
        else:
            mol_weight = Desc.MolWt(mol)
    except BaseException:
        mol_weight = np.nan
    return mol_weight


def mol_to_charge(mol):
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        charge = rdmolops.GetFormalCharge(mol)
    except BaseException:
        charge = np.nan
    return charge


def check_neutral_charge(mol):
    
    valid = mol_to_charge(mol) == 0
    return valid


def check_single_mol(mol):
   
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        num_frags = len(rdmolops.GetMolFrags(mol))
    except BaseException:
        num_frags = np.nan
    valid = num_frags == 1
    return valid


def inchi_to_smiles(inchi):
    try:
        mol = mol_from_inchi(inchi)
        smiles = mol_to_smiles(mol)
    except BaseException:
        smiles = np.nan
    return smiles


def smiles_to_selfies(smiles):
    sf, Chem = rdkit_import("selfies", "rdkit.Chem")
    try:
        # canonicalize, strip isomeric information, kekulize
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            isomericSmiles=False,
            kekuleSmiles=True)
        selfies = sf.encoder(smiles)
    except BaseException:
        selfies = np.nan
    return selfies


def make_morgan_fingerprint(mol, radius=3):
    
    modules = rdkit_import("rdkit.Chem.rdMolDescriptors", "rdkit.DataStructs")
    rmd = modules[0]
    ds = modules[1]
    
    fp = rmd.GetHashedMorganFingerprint(mol, radius) 
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr) 
    return fp_arr


def make_rdkit_fingerprint(mol):
   
    chem, ds = rdkit_import("rdkit.Chem", "rdkit.DataStructs")
    fp = chem.RDKFingerprint(mol)
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def make_maccs_fingerprint(mol):
   
    maccs, ds = rdkit_import("rdkit.Chem.MACCSkeys", "rdkit.DataStructs")
    fp = maccs.GenMACCSKeys(mol)
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def split_selfies(selfies_str):
    selfies = importlib.import_module("selfies")
    selfies_tokens = list(selfies.split_selfies(selfies_str))
    return selfies_tokens


def seq_apply(iterator, func):
   
    result = []
    for i in iterator:
        result.append(func(i))
    return result


def par_apply(iterator, func):
   
    n_jobs = joblib.cpu_count()
    par_func = joblib.delayed(func) 
    parallel = joblib.Parallel(
        backend=JOBLIB_BACKEND,
        n_jobs=JOBLIB_N_JOBS,
        timeout=JOBLIB_TIMEOUT
    )
    result = parallel(par_func(i) for i in iterator)
    return result 


def par_apply_series(series, func):
   
    series_iter = tqdm(
        series.items(),
        desc=pformat(func),
        total=series.shape[0])

    def series_func(tup): return func(tup[1]) 
    result_list = par_apply(series_iter, series_func)
    result_series = pd.Series(result_list, index=series.index) 
    return result_series


def seq_apply_series(series, func):

    series_iter = tqdm(
        series.iteritems(),
        desc=pformat(func),
        total=series.shape[0])

    def series_func(tup): return func(tup[1])
    result_list = seq_apply(series_iter, series_func)
    result_series = pd.Series(result_list, index=series.index)
    return result_series


def par_apply_df_rows(df, func):
    
    df_iter = tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0])  
    def df_func(tup): return func(tup[1]) 
    result_list = par_apply(df_iter, df_func)
    if isinstance(result_list[0], tuple):
        result_series = tuple([pd.Series(rl, index=df.index)
                              for rl in zip(*result_list)])
    else:
        result_series = pd.Series(result_list, index=df.index)
    return result_series


def seq_apply_df_rows(df, func):
   
    df_iter = tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0])
    def df_func(tup): return func(tup[1]) 
    result_list = seq_apply(df_iter, df_func) 
    if isinstance(result_list[0], tuple):
        result_series = tuple([pd.Series(rl, index=df.index) for rl in zip(*result_list)]) 
    else:
        result_series = pd.Series(result_list, index=df.index) 
    return result_series


def parse_ace_str(ce_str):

    if none_or_nan(ce_str):
        return np.nan
    matches = {

        r"^[\d]+[.]?[\d]*$": lambda x: float(x), 
        r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")), 
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")), 

        r"^[\d]+[.]?[\d]*HCD$": lambda x: float(x.rstrip("HCD")), 
        r"^CE [\d]+[.]?[\d]*$": lambda x: float(x.lstrip("CE ")), 
        r"^[\d]+HCD$": lambda x: float(x.rstrip("HCD")),  

        r"^[\d]+[.]?[\d]*V$": lambda x: float(x.rstrip("V")),  
        r"^[\d]+[.]?[\d]* [Vv]$": lambda x: float(x.rstrip(" Vv")),
    }
    for k, v in matches.items(): 
        if re.match(k, ce_str): 
            return v(ce_str)
    return np.nan


def parse_nce_str(ce_str):
    """这段代码的功能是解析碰撞能量字符串（ce_str），提取其中的相对碰撞能量（Normalized Collision Energy，NCE），并将其标准化为浮点数"""

    if none_or_nan(ce_str):
        return np.nan
    matches = {

        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[0].lstrip("NCE=").rstrip("%")), 
        r"^NCE=[\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("NCE=").rstrip("%")), 

        r"^[\d]+[.]?[\d]*$": lambda x: 100. * float(x) if float(x) < 2. else np.nan, 
        r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(x.rstrip(" %(nominal)")), 
        r"^HCD [\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("HCD ").rstrip("%")),
        r"^[\d]+[.]?[\d]* NCE$": lambda x: float(x.rstrip("NCE")), 
        r"^[\d]+[.]?[\d]*\(NCE\)$": lambda x: float(x.rstrip("(NCE)")), 
        r"^[\d]+[.]?[\d]*[ ]?%$": lambda x: float(x.rstrip(" %")), 
        r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(x.lstrip("HCD (NCE").rstrip("%)")), 
        r"^[\d]+[.]?[\d]* \(nominal\)$": lambda x: float(x.rstrip(" (nominal)")),  
    }
    for k, v in matches.items(): 
        if re.match(k, ce_str): 
            return v(ce_str)
    return np.nan


def parse_inst_info(df):
   
    inst_type_str = df["inst_type"] 
    inst_str = df["inst"]
    frag_mode_str = df["frag_mode"] 
    col_energy_str = df["col_energy"] 
    
    if inst_type_str == "EI":
        assert inst_str == "EI"
        assert frag_mode_str == "EI"
        assert col_energy_str == "100"
        return "EI", "EI"
    if none_or_nan(inst_type_str):
     
        inst_map = {
            "Maxis II HD Q-TOF Bruker": "QTOF",
            "qToF": "QTOF",
            "Orbitrap": "FT"
        }
        if none_or_nan(inst_str):
            inst_type = np.nan 
        elif inst_str in inst_map:
            inst_type = inst_map[inst_str]
        else:
            inst_type = "Other"
    else:
        inst_type_map = { 
            "QTOF": "QTOF",
            "FT": "FT",
            "Q-TOF": "QTOF",
            "HCD": "FT",
            "QqQ": "QQQ",
            "QqQ/triple quadrupole": "QQQ",
            "IT/ion trap": "IT",
            "IT-FT/ion trap with FTMS": "FT",
            "Q-ToF (LCMS)": "QTOF",
            "Bruker Q-ToF (LCMS)": "QTOF",
            "ESI-QTOF": "QTOF",
            "ESI-QFT": "FT",
            "ESI-ITFT": "FT",
            "Linear Ion Trap": "IT",
            "LC-ESI-QTOF": "QTOF",
            "LC-ESI-QFT": "FT",
            "LC-ESI-QQQ": "QQQ",
            "LC-ESI-QQ": "QQQ",
            "LC-ESI-QIT": "IT",
            "LC-Q-TOF/MS": "QTOF",
            "LC-ESI-ITFT": "FT",
            "LC-ESI-ITTOF": "IT",
            "LC-ESI-IT": "IT",
            "LC-QTOF": "QTOF",
            "LC-APPI-QQ": "QQQ",
            "qToF": "QTOF",
            "MALDI-TOFTOF": "Other",
            "FAB-EBEB": "Other",
        }
        if inst_type_str in inst_type_map:
            inst_type = inst_type_map[inst_type_str]
        else:
            inst_type = "Other"
   
    if inst_type_str == "HCD":
        frag_mode = "HCD"
    elif isinstance(col_energy_str, str) and "HCD" in col_energy_str:
        frag_mode = "HCD"
    elif none_or_nan(frag_mode_str) or frag_mode_str == "CID":
        frag_mode = "CID"
    elif frag_mode_str == "HCD":
        frag_mode = "HCD"
    else:
        frag_mode = np.nan
    return inst_type, frag_mode


def parse_ion_mode_str(ion_mode_str):
    
    if none_or_nan(ion_mode_str):
        return np.nan
    if ion_mode_str in ["P", "N", "E", "EI"]:
        return ion_mode_str
    elif ion_mode_str == "POSITIVE" or ion_mode_str =='Positive': 
        return "P"
    elif ion_mode_str == "NEGATIVE" or ion_mode_str == 'Negative':
        return "N"
    else:
        return np.nan


def parse_ri_str(ri_str):
    
    if none_or_nan(ri_str):
        return np.nan
    else:
        return float(ri_str)


def parse_prec_type_str(prec_type_str):
  
    if none_or_nan(prec_type_str):
        return np.nan
    if prec_type_str == "EI":
        return "EI"
    elif prec_type_str.endswith("1+"):
        return prec_type_str.replace("1+", "+")
    elif prec_type_str.endswith("1-"):
        return prec_type_str.replace("1-", "-")
    else:
        return prec_type_str


def parse_peaks_str(peaks_str):
   
    if none_or_nan(peaks_str):
        return np.nan
   
    lines = peaks_str.strip().split("\n")
    peaks = []
    for line in lines:
        if len(line) == 0:
            continue
     
        line = line.split()
        mz = line[0]
        ints = line[1]
        peaks.append((mz, ints)) 
    return peaks


def convert_peaks_to_float(peaks):

    float_peaks = []
    for peak in peaks:
        float_peaks.append((float(peak[0]), float(peak[1])))
    return float_peaks


def get_res(peaks):

    ress = [] 
    for mz, ints in peaks:
        dec_idx = mz.find(".")
        if dec_idx == -1:
            res = 0
        else:
            res = len(mz) - (dec_idx + 1) 
        ress.append(res)
    highest_res = max(ress)
    return highest_res


def get_murcko_scaffold(mol, output_type="smiles", include_chirality=False):
    
    if none_or_nan(mol):
        return np.nan
    MurckoScaffold = importlib.import_module(
        "rdkit.Chem.Scaffolds.MurckoScaffold")
    if output_type == "smiles":
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality)
    else:
        raise NotImplementedError
    return scaffold


def atom_type_one_hot(atom):

    chemutils = importlib.import_module("dgllife.utils")
    return chemutils.atom_type_one_hot(
        atom, allowable_set=ELEMENT_LIST, encode_unknown=True
    )


def atom_bond_type_one_hot(atom):

    chemutils = importlib.import_module("dgllife.utils")
    bs = atom.GetBonds() 
    if not bs:
        return [False, False, False, False] 
   
    bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
    
    return [any(bt[:, i]) for i in range(bt.shape[1])]


def analyze_mol(mol):

    import rdkit
    from rdkit.Chem.Descriptors import MolWt
    import rdkit.Chem as Chem
    mol_dict = {}
    mol_dict["num_atoms"] = mol.GetNumHeavyAtoms()
    mol_dict["num_bonds"] = mol.GetNumBonds(onlyHeavy=True)
    mol_dict["mol_weight"] = MolWt(mol)
    mol_dict["num_rings"] = len(list(Chem.GetSymmSSSR(mol)))
    mol_dict["max_ring_size"] = max(
        [-1] + [len(list(atom_iter)) for atom_iter in Chem.GetSymmSSSR(mol)])
    cnops_counts = {
        "C": 0,
        "N": 0,
        "O": 0,
        "P": 0,
        "S": 0,
        "Cl": 0,
        "other": 0}
    bond_counts = {"single": 0, "double": 0, "triple": 0, "aromatic": 0}
    cnops_bond_counts = {"C": [-1], "N": [-1],
                         "O": [-1], "P": [-1], "S": [-1], "Cl": [-1]}
    h_counts = 0
    p_num_bonds = [-1]
    s_num_bonds = [-1]
    other_atoms = set()
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol in cnops_counts:
            cnops_counts[atom_symbol] += 1
            cnops_bond_counts[atom_symbol].append(len(atom.GetBonds()))
        else:
            cnops_counts["other"] += 1
            other_atoms.add(atom_symbol)
        h_counts += atom.GetNumImplicitHs()
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == rdkit.Chem.rdchem.BondType.SINGLE:
            bond_counts["single"] += 1
        elif bond_type == rdkit.Chem.rdchem.BondType.DOUBLE:
            bond_counts["double"] += 1
        elif bond_type == rdkit.Chem.rdchem.BondType.TRIPLE:
            bond_counts["triple"] += 1
        else:
            assert bond_type == rdkit.Chem.rdchem.BondType.AROMATIC
            bond_counts["aromatic"] += 1
    mol_dict["other_atoms"] = ",".join(sorted(list(other_atoms)))
    mol_dict["H_counts"] = h_counts
    for k, v in cnops_counts.items():
        mol_dict[f"{k}_counts"] = v
    for k, v in bond_counts.items():
        mol_dict[f"{k}_counts"] = v
    for k, v in cnops_bond_counts.items():
        mol_dict[f"{k}_max_bond_counts"] = max(v)
    return mol_dict


def check_atoms(mol, element_list=ELEMENT_LIST):
   
    rdkit = importlib.import_module("rdkit")
    valid = all(a.GetSymbol() in element_list for a in mol.GetAtoms())
    return valid


def check_num_bonds(mol):
   
    rdkit = importlib.import_module("rdkit")
    valid = mol.GetNumBonds() > 0
    return valid


CHARGE_FACTOR_MAP = {
    1: 1.00,
    2: 0.90,
    3: 0.85,
    4: 0.80,
    5: 0.75,
    "large": 0.75
}


def get_charge(prec_type_str):

    if prec_type_str == "EI":
        return 1
    end_brac_idx = prec_type_str.index("]") 
    charge_str = prec_type_str[end_brac_idx + 1:]  

    if charge_str == "-":
        charge_str = "1-"
    elif charge_str == "+":
        charge_str = "1+"
    assert len(charge_str) >= 2
    sign = charge_str[-1] 
    assert sign in ["+", "-"]
    magnitude = int(charge_str[:-1]) 

    if sign == "+":
        charge = magnitude
    else:
        charge = -magnitude
    return charge


def nce_to_ace_helper(nce, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    ace = (nce * prec_mz * charge_factor) / 500.
    return ace


def ace_to_nce_helper(ace, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    nce = (ace * 500.) / (prec_mz * charge_factor)
    return nce


def nce_to_ace(row):

    prec_mz = row["prec_mz"]
    nce = row["nce"]  
    prec_type = row["prec_type"]  
    charge = np.abs(get_charge(prec_type)) 
    ace = nce_to_ace_helper(nce, charge, prec_mz)
    return ace


def ace_to_nce(row):

    prec_mz = row["prec_mz"]
    ace = row["ace"]
    prec_type = row["prec_type"]
    charge = np.abs(get_charge(prec_type))
    nce = ace_to_nce_helper(ace, charge, prec_mz)
    return nce


def parse_formula(formula):

    element_counts = {element: 0 for element in ELEMENT_LIST}
    cur_element = None
    cur_count = 1
    for token in re.findall('[A-Z][a-z]?|\\d+|.', formula):
        if token.isalpha():
            if cur_element is not None:
                assert element_counts[cur_element] == 0
                element_counts[cur_element] += cur_count
            cur_element = token
            cur_count = 1
        elif token.isdigit():
            cur_count = int(token)
        else:
            raise ValueError(f"Invalid token {token}")
    assert element_counts[cur_element] == 0
    element_counts[cur_element] += cur_count
    return element_counts


def check_mol_props(df):

    valid_atoms = par_apply_series(df["mol"], check_atoms)
    valid_num_bonds = par_apply_series(df["mol"], check_num_bonds)
    valid_charge = par_apply_series(df["mol"], check_neutral_charge)
    valid_single_mol = par_apply_series(df["mol"], check_single_mol)
    print(
        f"mol filters: atoms={valid_atoms.sum()}, num_bonds={valid_num_bonds.sum()}, charge={valid_charge.sum()}, single_mol={valid_single_mol.sum()}")
    df = df[valid_atoms & valid_num_bonds & valid_charge & valid_single_mol] 
    return df


EPS = np.finfo(np.float32).eps


def flatten_lol(list_of_list):

    flat_list = []
    for ll in list_of_list:
        flat_list.extend(ll)
    return flat_list




@contextlib.contextmanager
def th_temp_seed(seed):
    state = th.get_rng_state()
    th.manual_seed(seed)
    try:
        yield
    finally:
        th.set_rng_state(state)


def np_scatter_add(input, axis, index, src):
    """ numpy wrapper for scatter_add """

    th_input = th.as_tensor(input, device="cpu")
    th_index = th.as_tensor(index, device="cpu")
    th_src = th.as_tensor(src, device="cpu")
    dim = axis
    th_output = th.scatter_add(th_input, dim, th_index, th_src)
    output = th_output.numpy()
    return output


def np_one_hot(input, num_classes=None):
    """ numpy wrapper for one_hot """

    th_input = th.as_tensor(input, device="cpu")
    th_oh = th.nn.functional.one_hot(th_input, num_classes=num_classes)
    oh = th_oh.numpy()
    return oh


def list_dict_to_dict_array(list_dict):

    dict_keys = list_dict[0].keys()
    dict_list = {k: [] for k in dict_keys}
    for d in list_dict:
        for k, v in d.items():
            dict_list[k].append(v)
    dict_arr = {}
    for k, v in dict_list.items():
        dict_arr[k] = np.stack(v, axis=0)
    return dict_arr


def params_to_str(params):

    _params = {}
    for k, v in params.items():
        if isinstance(v, list):
            _params[k] = str(sorted(v))
        else:
            assert isinstance(v, str), v
            _params[k] = v
    params_str = str(_params)
    return _params, params_str


def booltype(x):
    return bool(strtobool(x))




@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def time_function(func, *args, num_reps=100):

    deltas = []
    for i in range(num_reps):
        start = timer()
        func(*args)
        end = timer()
        delta = timedelta(seconds=end - start)
        deltas.append(delta)
    avg_delta = np.mean(deltas)
    print(avg_delta)


def sharpen(x, p):
    return x**p / th.sum(x**p, dim=1, keepdim=True)




@contextlib.contextmanager
def suppress_output(stdout=True, stderr=True):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if stdout:
            sys.stdout = devnull
        if stderr:
            sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def list_str2float(str_list):
    return [float(str_item) for str_item in str_list]




class DummyContext:

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class DummyScaler:

    def scale(self, grad):
        return grad

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass


def sparse_to_dense(val, idx_0, idx_1, dim_0, dim_1):
    res = -th.ones((dim_0, dim_1), device=val.device, dtype=val.dtype)
    res[idx_0, idx_1] = val
    return res


def count_parameters(model, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_nograd_param_names(model):

    nograd_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            nograd_names.append(name)
    return nograd_names


def get_scaler(amp):

    if amp:
        scaler = th.cuda.amp.GradScaler() 
    else:
        scaler = DummyScaler() 
    return scaler


def get_pbar(iter, log_tqdm, **pbar_kwargs):
    if log_tqdm:
        return tqdm.tqdm(iter, **pbar_kwargs)
    else:
        if "desc" in pbar_kwargs:
            print(pbar_kwargs["desc"])
        return iter
