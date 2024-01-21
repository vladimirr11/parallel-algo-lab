import sys, os
import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix


def generate_csr_matrix(rows: int, cols: int, density=0.25):
    rng = np.random.default_rng()
    spm = sp.sparse.random(rows, cols, density=density, random_state=rng, dtype=np.int8)
    return csr_matrix(spm)


csr_mat = generate_csr_matrix(1000, 1000)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

if not os.path.exists("data"):
    os.makedirs("data")
    
with open("data/csr_matrix_data.txt", "w") as writer:
    writer.writelines(str(csr_mat.data))

with open("data/csr_matrix_indices.txt", "w") as writer:
    writer.writelines(str(csr_mat.indices))

with open("data/csr_matrix_indptr.txt", "w") as writer:
    writer.writelines(str(csr_mat.indptr))