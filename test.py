from posixpath import dirname
import cv2
import os
import numpy as np
from sympy.utilities.iterables import multiset_permutations, variations

colors = np.array([[255, 218, 185], [47, 79, 79], [25, 25, 112], [123, 104, 238], [0, 191, 255]], dtype=int)

def getL(g, tao, sigmal, sigmad):
	height, width = g.shape
	t1 = g - np.concatenate((np.zeros((tao, width)), g[:height-tao, :]), axis=0)
	t2 = g - np.concatenate((g[tao:, :], np.zeros((tao, width))), axis=0)
	print(t1)
	print(t2)
	# l = np.zeros(g.shape)
	l = np.where((g>sigmal)&(t1>sigmad)&(t2>sigmad), 255, 0)
	print(l)

	t1 = g - np.concatenate((np.zeros((height, tao)), g[:, :width-tao]), axis=1)
	t2 = g - np.concatenate((g[:, tao:], np.zeros((height, tao))), axis=1)
	print(t1)
	print(t2)
	l = np.where((l==255)|((g>sigmal)&(t1>sigmad)&(t2>sigmad)), 255, 0)
	return l

def get_all_files(path):
	allfile = []
	for dirpath, dirnames, filenames in os.walk(path):
		print(dirpath)
		print(dirnames)
		print(filenames)
		for dir in dirnames:
			allfile.append(os.path.join(dirpath, dir))
		for name in filenames:
			allfile.append(os.path.join(dirpath, name))
	return allfile

af = get_all_files("./resources/raw")
print(af)
	
	

