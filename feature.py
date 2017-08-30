'''
@author : wanglang
'''
from PIL import Image
import os
import numpy as np

def getBinaryPix(im):
    im = Image.open(im)
    img = np.array(im)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if(img[i,j] <= 128):
                img[i,j] = 0
            else:
                img[i,j] = 1;
    binpix = np.ravel(img)
    return binpix

def getfiles(dirs):
    fs = []
    for fr in os.listdir(dirs):
        f = dirs + fr
        if f.rfind(u'.DS_Store') == -1:
            fs.append(f)
    return fs

def writefile(content):
    with open('/Users/wanglang/Desktop/pic4/train_data.txt','a+') as f:
        f.write(content)
        f.write('\n')
        f.close()

if __name__ == '__main__':
    dirs = '/Users/wanglang/Desktop/pic3/%s/'
    for i in range(9):
        for f in getfiles(dirs % (i)):
            pixs = getBinaryPix(f).tolist()
            pixs.append(i)
            pixs = [ str(i) for i in pixs ]
            content = ','.join(pixs)
            writefile(content)
