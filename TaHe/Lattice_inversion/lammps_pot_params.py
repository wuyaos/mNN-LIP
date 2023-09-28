import json
import datetime
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
from pathlib import Path
import pandas as pd


def get_atomic_info(output_file_table):

    intertype = ["He", "Ta", "TaHe"]
    header_str = f'''# DATE: {datetime.datetime.now().strftime('%Y_%m_%d')}
# please cite "Wu, F. et al. Modeling of metal nanoparticles: Development of neural-network interatomic potential inspired by features of the modified embedded-atom method. Phys. Rev. B 102, 144107 (2020)."
# Lattice inversion potential for TaHe
# Lammps script format:
'''
    context = ""

    fp = open(output_file_table, 'w')
    for indexi, i in enumerate(intertype):
        context += f"LI_{i}\n"
        data = pd.read_csv(f"./out/{i}.csv")
        # 行数
        length = data.shape[0]
        rmin = data['r'][0]
        rmax = data['r'][length-1]
        header_str += f'''# {i[0]}-{i[1]}: pair_style  table spline {length}
#        pair_coeff  * * {output_file_table.name} LI_{i} {rmax}\n'''
        context += f"N {length:>5} R {rmin:> .8e} {rmax:> .8e}\n\n"
        for indexj in range(length):
            context += f"{indexj+1:>5} {data['r'][indexj]:> .8e} {data['E'][indexj]:> .8e} {data['F'][indexj]:> .8e}\n"
            if indexj == length - 1:
                context += f"\n"
    fp.write(header_str)
    fp.write("\n\n")
    fp.write(context)
    fp.close()

get_atomic_info(Path("./out/TaHe_lipot.table"))
