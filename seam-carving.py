import numpy as np
import cv2
from sys import argv

class SeamCarving:
	def __init__(self, image, newimage, ratio, axis='x'):
		img = cv2.imread(image, 1)
		img1 = self.carve(ratio, img, axis)
		cv2.imwrite(newimage, img1)

	def carve(self, ratio, img, axis):
		h, w, c = img.shape
		CTR=(h,w)
		if axis=="y":
			img=cv2.transpose(img)
			img=cv2.flip(img,flipCode=0)
			a=w
			w=h
			h=a
		num = round(ratio*w)
		for i in range(0, num):
			print("{0}/{1}".format(i+1,num))
			img = self.carve_col_dp(img, axis)
		if axis=="y":
			img=cv2.transpose(img)
			img=cv2.flip(img,flipCode=1)
		return img

	def energy(self, img, axis):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#derv = cv2.Scharr(gray,cv2.CV_64F,1,0) / 64
		#derv = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
		derv = cv2.Laplacian(gray,cv2.CV_64F, ksize=7)
		return np.absolute(derv) # in order to not have large negative values be the minimum

	def carve_col_greedy(self, img, axis):
		#leads to bad results, dynamic programming is necessary
		h, w, c = img.shape
		e = self.energy(img, axis)
		mask = np.ones((h, w), dtype=np.bool)
		j = np.argmin(e[-1])
		for i in reversed(range(h)):
			mask[(i,j)]=False
			if i!=0:
				m=(i-1,j)
				if j!=0:
					if e[(i-1,j-1)]<e[m]:
						m=(i-1,j-1)
				if j!=w-1:
					if e[(i-1,j+1)]<e[m]:
						m=(i-1,j+1)
				j=m[1]
		return self.remove_seam(img, mask)

	def carve_col_dp(self, img, axis):
		mask = self.create_mask(self.energy(img, axis))
		return self.remove_seam(img, mask)

	def create_mask(self, energy):
		h, w = energy.shape
		seams = np.zeros((h, w), dtype=np.float64)
		mask = np.ones((h, w), dtype=np.bool)
		#Now first calculate from top to bottom in DP style, the total energy function
		for i in range(h):
			for j in range(w):
				if h==0:
					seams[i][j] = energy[i][j]
				else:
					l = [seams[i-1][j]]
					if j!=0:
						l.append(seams[i-1][j-1])
					if j!=w-1:
						l.append(seams[i-1][j+1])
					seams[i][j] = energy[i][j]+min(l)
		#then, find the pixel with a minimum energy function, and construct a seam by tracing back to the neighboring pixels on top which result in the least energy function
		killw = 0
		for i in reversed(range(h)):
			if i==h-1:
				killw = np.argmin(seams[h-1])
			else:
				amn = killw-1
				amx = killw+2
				if killw==0:
					amn=0
				if amx==w or amx==w-1:
					amx=w
				killw = amn+np.argmin(seams[i][amn:amx])
			mask[(i,killw)]=False
		return mask

	def remove_seam(self, img, mask):
		#Makes use of a binary mask to get rid of pixels marked with 0
		h, w, c = img.shape
		mask = np.stack([mask]*3, axis=2)
		img3 = img[mask].reshape((h, w-1, 3))
		return img3

	def add_seam(self, img, mask):
		# to be implemented one day
		pass

if __name__=="__main__":
	fileName = argv[1]
	resName = argv[1].split('.')[0]+"1."+argv[1].split('.')[1]
	ratio = 0.2
	if len(argv)>=3:
		ratio = float(argv[2])
	axis = 'x'
	if len(argv)>=4:
		axis = argv[3]
	SeamCarving(fileName, resName, ratio, axis)

	# example: python seamcarving.py cat.png 0.3 x
