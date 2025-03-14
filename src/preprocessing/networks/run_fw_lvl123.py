import numpy as np
from numba import njit, prange

@njit(parallel=True)
def fw_lvl23(k, A, B, number_nodes):
    for i in prange(number_nodes):
        for j in range(number_nodes):
            tmp = A[i,k]+A[k,j]
            if tmp < A[i,j]:
                A[i,j] = tmp
                B[i,j] = B[i,k]+B[k,j]

def main():
    # Load the matrices and parameters from temporary files
    A = np.load('A.npy')
    B = np.load('B.npy')
    set_stop_nodes = set(np.load('set_stop_nodes.npy'))
    with open('params.txt', 'r') as f:
        number_nodes = int(f.readline().strip())

    # Run the Floyd-Warshall algorithm
    for k in range(number_nodes):
        # do not route through stop nodes!
        if k in set_stop_nodes:
            print(f"Skipping intermediary routing via stop node {k}")
            continue
        if k % 250 == 0:
            print(f"\t ... loop {k}/{number_nodes}")
        fw_lvl23(k, A, B, number_nodes)

    # Save the modified matrices
    np.save('A.npy', A)
    np.save('B.npy', B)

if __name__ == "__main__":
    main()