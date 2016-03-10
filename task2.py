import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('beautiful.png')
[r,g,b] = [img[:,:,i] for i in range(3)]

fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img)
ax2.imshow(r, cmap = 'Reds')
ax3.imshow(g, cmap = 'Greens')
ax4.imshow(b, cmap = 'Blues')
plt.show()

#extract r into UsV
rU, rs, rV = np.linalg.svd(r, full_matrices = False)
#extract g into UsV
gU, gs, gV = np.linalg.svd(g, full_matrices = False)
#extract b into UsV
bU, bs, bV = np.linalg.svd(b, full_matrices = False)

#calculate non-zero element in matrix s of r
red = np.count_nonzero(rs)
#calculate non-zero element in matrix s of g
green = np.count_nonzero(gs)
#calculate non-zero element in matrix s of b
blue = np.count_nonzero(bs)

#display the number of non-zero element of r, g, b
print("Total number of non-zero element in S of r:", red)
print("Total number of non-zero element in S of g:", green)
print("Total number of non-zero element in S of b:", blue)

#create 3 different matrices, named rs1, gs1, bs1
rs1 = [0 for x in range (0,red)]
gs1 = [0 for x in range (0,green)]
bs1 = [0 for x in range (0,blue)]

#change the matrix s of r where only keep the first 30 non-zero elements
#change other elements as 0, assigned to matrix rs1
for i in range (0, red):
    if (i < 30):
        rs1[i] = rs[i]
    else:
        rs1[i] = 0
 
#change the matrix s of g where only keep the first 30 non-zero elements
#change other elements as 0, assigned to matrix gs1       
for i in range (0, green):
    if (i < 30):
        gs1[i] = gs[i]
    else:
        gs1[i] = 0

#change the matrix s of b where only keep the first 30 non-zero elements
#change other elements as 0, assigned to matrix bs1
for i in range (0, blue):
    if (i < 30):
        bs1[i] = bs[i]
    else:
        bs1[i] = 0

#convert matrix rs1 into a diagonal matrix rS1
rS1 = np.diag(rs1)
#convert matrix gs1 into a diagonal matrix gS1
gS1 = np.diag(gs1)
#convert matrix bs1 into a diagonal matrix bS1
bS1 = np.diag(bs1)

#create new matrices, named newR, newG, newB
#by multiply matrix U, new matrix S1 and matrix V of r, g, b
newR = np.dot(rU, np.dot(rS1,rV))
newG = np.dot(gU, np.dot(gS1,gV))
newB = np.dot(bU, np.dot(bS1,bV))

#create image from new matrices and display
newImage = np.dstack([newR, newG, newB])
plt.imsave('lowerImage.png', newImage)
plt.imshow(newImage)
plt.show()

#below same with the scenario above
#but just keep the first 200 non-zero elements instead of 30
rs2 = [0 for x in range (0,red)]
gs2 = [0 for x in range (0,green)]
bs2 = [0 for x in range (0,blue)]

for i in range (0, red):
    if (i < 200):
        rs2[i] = rs[i]
    else:
        rs2[i] = 0
        
for i in range (0, green):
    if (i < 200):
        gs2[i] = gs[i]
    else:
        gs2[i] = 0

for i in range (0, blue):
    if (i < 200):
        bs2[i] = bs[i]
    else:
        bs2[i] = 0        

rS2 = np.diag(rs2)
gS2 = np.diag(gs2)
bS2 = np.diag(bs2)

newR1 = np.dot(rU, np.dot(rS2,rV))
newG1 = np.dot(gU, np.dot(gS2,gV))
newB1 = np.dot(bU, np.dot(bS2,bV))

newImage1 = np.dstack([newR1, newG1, newB1])
plt.imsave('betterImage', newImage1)
plt.imshow(newImage1)
plt.show()