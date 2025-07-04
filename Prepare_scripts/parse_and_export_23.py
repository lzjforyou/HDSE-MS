from collections import Counter
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
import argparse
import glob
import sys
import requests
import json


from data_utils import rdkit_import, seq_apply, par_apply, seq_apply_series, par_apply_series, par_apply_df_rows, ELEMENT_LIST

from tqdm import tqdm

key_dict = {
    "Precursor_type": "prec_type",
    "Spectrum_type": "spec_type",
    "PrecursorMZ": "prec_mz",
    "Instrument_type": "inst_type",
    "Collision_energy": "col_energy",
    "Ion_mode": "ion_mode",
    "Ionization": "ion_type",
    "ID": "spec_id",
    "Collision_gas": "col_gas",
    "Pressure": "pressure",
    "Num peaks": "num_peaks",
    "MW": "mw",
    "ExactMass": "exact_mass",
    "CAS#": "cas_num",
    "NIST#": "nist_num",
    "Name": "name",
    "MS": "peaks",
    "SMILES": "smiles",
    "Rating": "rating",
    "Frag_mode": "frag_mode",
    "Instrument": "inst",
    "RI": "ri",
    "InChIKey": "inchi_key"
}


def inchi_key_to_smiles(inchi_key, local_mapping_file="", delay=1, missing_keys=None):
    """
    Query SMILES for a given InChIKey, first from a local file, then from the PubChem API if not found.
    :param inchi_key: str, InChIKey
    :param local_mapping_file: str, path to local InChIKey-SMILES JSON mapping
    :param delay: float, delay between API requests in seconds (default: 1)
    :param missing_keys: set, collect InChIKeys not found locally
    :return: str or None, SMILES or None if failed
    """
    if os.path.exists(local_mapping_file):
        try:
            with open(local_mapping_file, "r") as f:
                local_mapping = json.load(f)
            if inchi_key in local_mapping:
                return local_mapping[inchi_key]
        except Exception as e:
            print(f"Error reading local mapping file: {e}")

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=60)
        time.sleep(delay)
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        else:
            print(f"HTTP Error {response.status_code} for InChIKey {inchi_key}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"RequestException for InChIKey {inchi_key}: {e}")
        return None

def inchi_to_smiles(inchi_keys):
    """
    Sequentially convert a list of InChIKeys to SMILES, count failures.
    :param inchi_keys: list of InChIKeys
    :return: tuple, (smiles_list, failed_count)
    """
    failed_count = 0
    smiles_list = []
    for inchi_key in tqdm(inchi_keys, desc="Processing InChIKey to SMILES"):
        try:
            smiles = inchi_key_to_smiles(inchi_key)
            if smiles is None:
                failed_count += 1
            smiles_list.append(smiles)
        except Exception as e:
            failed_count += 1
            smiles_list.append(None)
    return smiles_list, failed_count

def extract_info_from_comments(comments, key):
    start_idx = comments.find(key)
    if start_idx == -1:
        return None
    start_idx += len(key) + 1  # skip '='
    end_idx = start_idx + 1
    cur_char = comments[end_idx]
    while cur_char != "\"":
        end_idx += 1
        cur_char = comments[end_idx]
    value = comments[start_idx:end_idx]
    return value

def validate_peaks(peaks_str, line_number):
    """
    Validate peaks data format.
    :param peaks_str: string of 'm/z intensity' peaks
    :param line_number: line number in source file (for debug)
    :return: True if valid, False otherwise
    """
    if not peaks_str.strip():
        print(f"Empty peaks data at line {line_number}")
        return False
    for line in peaks_str.strip().split('\n'):
        parts = line.split()
        try:
            m_z, intensity = float(parts[0]), float(parts[1])
            if m_z < 0 or intensity < 0:
                print(f"Negative value in peaks at line {line_number}: {line}")
                return False
        except ValueError:
            print(f"Non-numeric peak data at line {line_number}: {line}")
            return False
    return True

def preproc_msp(msp_dirs, keys, num_entries):
    """
    Parse MSP files from multiple folders into a DataFrame.
    :param msp_dirs: list of directories containing MSP files
    :param keys: list of field names to extract
    :param num_entries: max number of entries to read, -1 for all
    """
    msp_files = []
    for msp_dir in msp_dirs:
        for root, _, files in os.walk(msp_dir):
            for file in files:
                if file.endswith(".MSPEC"):
                    msp_files.append(os.path.join(root, file))
    print(f"Found {len(msp_files)} MSP files to process.")

    raw_data_list = []
    sum_invalid = 0
    total_entries = 0

    for msp_fp in tqdm(msp_files, desc="Processing MSP files"):
        with open(msp_fp) as f:
            raw_data_lines = f.readlines()
        raw_data_item = {key: None for key in keys}
        read_ms = False
        line_number = 0

        for raw_l in raw_data_lines:
            if num_entries > -1 and total_entries >= num_entries:
                break
            raw_l = raw_l.replace('\n', '')
            line_number += 1
            if raw_l == '':
                if raw_data_item['MS'] and not validate_peaks(raw_data_item['MS'], line_number):
                    sum_invalid += 1
                    print(f"Invalid peaks data at line {line_number} in file {msp_fp}")
                    raw_data_item['MS'] = None
                    raw_data_item = {key: None for key in keys}
                raw_data_list.append(raw_data_item.copy())
                raw_data_item = {key: None for key in keys}
                read_ms = False
                total_entries += 1
            elif read_ms:
                raw_data_item['MS'] = raw_data_item['MS'] + raw_l + '\n'
            else:
                if "RI:" in raw_l:
                    raw_l_split = raw_l.split(':')
                else:
                    raw_l_split = raw_l.split(': ')
                assert len(raw_l_split) >= 2
                key = raw_l_split[0]
                if key == "Num peaks" or key == "Num Peaks":
                    assert len(raw_l_split) == 2, raw_l_split
                    value = raw_l_split[1]
                    raw_data_item['Num peaks'] = int(value)
                    raw_data_item['MS'] = ''
                    read_ms = True
                elif key == "Comments":
                    comments = ": ".join(raw_l_split[1:])
                    smiles = extract_info_from_comments(comments, "computed SMILES")
                    rating = extract_info_from_comments(comments, "MoNA Rating")
                    frag_mode = extract_info_from_comments(comments, "fragmentation mode")
                    if not (smiles is None):
                        raw_data_item["SMILES"] = smiles
                    if not (rating is None):
                        raw_data_item["Rating"] = rating
                    if not (frag_mode is None):
                        raw_data_item["Frag_mode"] = frag_mode
                elif key == "CAS#":
                    cas_nist_data = raw_l_split[1].split(";")
                    cas_value = cas_nist_data[0].strip()
                    nist_value = raw_l_split[2].strip()
                    raw_data_item["CAS#"] = cas_value
                    raw_data_item["NIST#"] = nist_value
                elif key in keys:
                    value = raw_l_split[1]
                    raw_data_item[key] = value

    msp_df = pd.DataFrame(raw_data_list)
    msp_df = msp_df.dropna(axis=0, how="all")
    print(f"{sum_invalid} invalid spectra filtered")
    print(f"Total {total_entries} valid entries processed")
    return msp_df

def preproc_nist_mol(mol_dp):
    """
    Read all .MOL files in a directory and return a DataFrame with spec_id and SMILES.
    """
    mol_fp_list = glob.glob(os.path.join(mol_dp, "*.MOL"))
    def proc_mol_file(mol_fp):
        modules = rdkit_import(
            "rdkit.Chem",
            "rdkit.Chem.rdinchi",
            "rdkit.Chem.AllChem")
        Chem = modules[0]
        rdinchi = modules[1]
        AllChem = modules[2]
        mol_fn = os.path.basename(os.path.normpath(mol_fp))
        spec_id = mol_fn.lstrip("ID").rstrip(".MOL")
        mol = Chem.MolFromMolFile(mol_fp, sanitize=True)
        if not (mol is None):
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = None
        entry = dict(
            spec_id=spec_id,
            smiles=smiles
        )
        return entry
    mol_df_entries = par_apply(mol_fp_list, proc_mol_file)
    mol_df = pd.DataFrame(mol_df_entries)
    return mol_df

def merge_and_check(msp_df, mol_df, rename_dict):
    """
    Clean and merge msp_df with mol_df, validate completeness.
    :param msp_df: DataFrame, raw msp data
    :param mol_df: DataFrame, optional, molecule data
    :param rename_dict: dict, column rename mapping
    :return: DataFrame, processed spec_df
    """
    msp_bad_cols = set(msp_df.columns) - set(rename_dict.keys())
    msp_df = msp_df.drop(columns=msp_bad_cols)
    msp_df = msp_df.rename(columns=rename_dict)

    if mol_df is None:
        assert not msp_df["smiles"].isna().all(), "All values in smiles column are null!"
        assert msp_df["spec_id"].isna().all(), "Non-null values in spec_id column, unexpected!"
        msp_df.loc[:, "spec_id"] = np.arange(msp_df.shape[0])
        spec_df = msp_df
    else:
        msp_df["spec_id"] = msp_df["nist_num"]
        print("Missing values per column:")
        print(msp_df.isna().sum())
        msp_df = msp_df[msp_df["inchi_key"].notna()]
        print("Missing values per column:")
        print(msp_df.isna().sum())
        smiles_list, failed_count = inchi_to_smiles(msp_df["inchi_key"])
        assert len(smiles_list) == len(msp_df["inchi_key"]), (
            f"Length mismatch: inchi_key ({len(msp_df['inchi_key'])}) and smiles_list ({len(smiles_list)})"
        )
        print(f"Failed to convert {failed_count} InChIKeys to SMILES")
        msp_df["smiles"] = smiles_list
        msp_df = msp_df.drop(columns=["inchi_key"])
        spec_df = msp_df

    print("Missing values per column:")
    print(spec_df.isna().sum())
    spec_df = spec_df.reset_index(drop=True)
    return spec_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--msp_file', type=str, required=False)
    parser.add_argument('--mol_dir', type=str, required=False)
    parser.add_argument('--output_name', type=str, default='mb_na_23_df')
    parser.add_argument('--raw_data_dp', type=str, default='data/raw')
    parser.add_argument('--output_dp', type=str, default='data/df')
    parser.add_argument('--num_entries', type=int, default=-1)
    parser.add_argument(
        '--output_type',
        type=str,
        default="json",
        choices=[
            "json",
            "csv"])
    args = parser.parse_args()
    msp_fp = [""]

    assert os.path.isfile(msp_fp)
    mol_df = None
    os.makedirs(args.output_dp, exist_ok=True)

    msp_df = preproc_msp(msp_fp, key_dict.keys(), args.num_entries)
    spec_df = merge_and_check(msp_df, mol_df, key_dict)
    spec_df_fp = os.path.join(args.output_dp,
                              f"{args.output_name}.{args.output_type}")
    if args.output_type == "json":
        spec_df.to_json(spec_df_fp)
    else:
        assert args.output_type == "csv"
        spec_df.to_csv(spec_df_fp, index=False)
