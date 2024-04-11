# import sys
# sys.path.append('../') # Add parent directory to the system path
# print(sys.path)
import interferometer as itf
# from direct_decomposition import direct_decomposition
import numpy as np

# type1fusion = 1/np.sqrt(2) * np.matrix([[1,0,0,1],[0,np.sqrt(2),0,0],[1,0,0,-1],[0,0,np.sqrt(2),0]])
# interferom = direct_decomposition(type1fusion, 2)
# print(interferom.depth())

M = 1/np.sqrt(2) * np.matrix([[-1,1],[1,1]])
I = itf.square_decomposition(M)
print(I.BS_list)