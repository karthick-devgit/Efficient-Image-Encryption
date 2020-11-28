import numpy as np
import random
import math
from PIL import Image
import time


class Substitution:
  
    def __init__(self,image,image_rgb):
      self.image = image
      self.image_rgb = image_rgb
      self.xdim,self.ydim=image.shape


    def logistic(self,z,u):
      z = z*u*(1-z)
      return z


    def fourBits(self,num):
      y = num & 15
      x = (num & 240)>>4
      return (x,y)


    def keyScheming(self,zk,uk):
      imageR = self.image_rgb[:,:,0]
      imageG = self.image_rgb[:,:,1]
      imageB = self.image_rgb[:,:,2]
      imageRGB = [imageR,imageG,imageB]
      xdim,ydim = imageR.shape
      z = zk
      u = uk
      w1 = list()
      w2 = list()
      for i in range(500):
        z = self.logistic(z,u)
      for i in range(xdim):
        z = self.logistic(z,u)
        w1.append(z)
      for i in range(ydim):
        z = self.logistic(z,u)
        w2.append(z) 
      
      sRwi = np.dot(w1,imageRGB[0])
      sRwi = np.dot(sRwi,w2)
      sGwi = np.dot(w1,imageRGB[1])
      sGwi = np.dot(sGwi,w2)
      sBwi = np.dot(w1,imageRGB[2])
      sBwi = np.dot(sBwi,w2)
      z = self.logistic(z,u)
      wr1 = z
      z = self.logistic(z,u)
      wr2 = z
      z = self.logistic(z,u)
      wr3 = z
      swi = np.dot([wr1,wr2,wr3],[sRwi,sGwi,sBwi])
      z01 = random.random()
      z02 = random.random()
      z03 = random.random()
      decPart = swi-math.floor(swi)
      z1 = z01 + decPart - math.floor(z01 + decPart)
      z2 = z02 + decPart - math.floor(z02 + decPart)
      z3 = z03 + decPart - math.floor(z03 + decPart)
      return [[z1,u],[z2,u],[z3,u]]

    def mapGenerator(self,rx,ry,l1,l2,d,si):    
      
      '''
      rx = 5
      ry = 4
      l1 = [4,3,1,2,5]
      l2 = [3,1,4,2]
      d = 0
      si = (3,4)
      ''' 
      n1 = list()

      if d == 0:
        ends = []
        for i in range(rx-1,-1,-1):
          ends.append((i,0))
        for i in range(1,ry):
          ends.append((0,i))
        for x,y in ends:
          while x<rx and y<ry :
            n1.append((l1[x],l2[y]))
            x = x+1
            y = y+1

      if d == 1:
        ends = []
        for i in range(ry-1,-1,-1):
          ends.append((0,i))
        for i in range(1,rx):
          ends.append((i,0))
        for x,y in ends:
          while x<rx and y<ry :
            n1.append((l1[x],l2[y]))
            x = x+1
            y = y+1

      if d == 2:
        ends = []
        for i in range(0,ry):
          ends.append((0,i))
        for i in range(1,rx,):
          ends.append((i,ry-1))
        
        for x,y in ends:
          while x<rx and y>-1 :
            n1.append((l1[x],l2[y]))
            x = x+1
            y = y-1


      if d == 3:
        ends = []
        for i in range(rx-1,-1,-1):
          ends.append((i,ry-1))
        for i in range(ry-2,-1,-1):
          ends.append((0,i))
        
        for x,y in ends:
          while x<rx and y>-1 :
            n1.append((l1[x],l2[y]))
            x = x+1
            y = y-1

      if d == 4:
        ends = []
        for i in range(0,rx):
          ends.append((i,0))
        for i in range(1,ry):
          ends.append((rx-1,i))
        
        for x,y in ends:
          while x>-1 and y<ry :
            n1.append((l1[x],l2[y]))
            x = x-1
            y = y+1

      if d == 5:
        ends = []
        for i in range(ry-1,-1,-1):
          ends.append((rx-1,i))
        for i in range(rx-2,-1,-1):
          ends.append((i,0))
        
        
        for x,y in ends:
          while x>-1 and y<ry :
            n1.append((l1[x],l2[y]))
            x = x-1
            y = y+1

      if d == 6:
        ends = []
        for i in range(0,ry):
          ends.append((rx-1,i))
        for i in range(rx-2,-1,-1):
          ends.append((i,ry-1))
        
        
        for x,y in ends:
          while x>-1 and y>-1 :
            n1.append((l1[x],l2[y]))
            x = x-1
            y = y-1

      if d == 7:
        ends = []
        for i in range(0,rx):
          ends.append((i,ry-1))
        for i in range(ry-2,-1,-1):
          ends.append((rx-1,i))
        
        for x,y in ends:
          while x>-1 and y>-1 :
            n1.append((l1[x],l2[y]))
            x = x-1
            y = y-1

      f = n1.index(si)
      n1 = n1[f:] + n1[:f] 
      f_list = list()
      j = 0
      for i in range(rx):
        f_list.append(n1[j:j+ry])
        j = j+ry
      return f_list



    def obtainMap(self,z1,u):
      k = z1
      chaoticSequenceM = list()
      chaoticSequenceN = list()
      for i in range(500):
          k = self.logistic(k,u)
      for i in range(self.xdim):
          k = self.logistic(k,u)
          chaoticSequenceM.append((k,i))
      for i in range(30):
          k = self.logistic(k,u)
      for i in range(self.ydim):
          k = self.logistic(k,u)
          chaoticSequenceN.append((k,i))
      sr =  self.logistic(k,u)
      sc =  self.logistic(k,u)
      d = self.logistic(k,u)
      chaoticSequenceM.sort()
      chaoticSequenceN.sort()
      sr = math.floor(sr*(10**14)) % self.xdim
      sc = math.floor(sc*(10**14)) % self.ydim
      d = math.floor(d*(10**14))%8

      r = list()
      c = list()
      for val,pos in chaoticSequenceM:
        r.append(pos)
      for val,pos in chaoticSequenceN:
        c.append(pos)

      sr = r.index(sr)
      sc = c.index(sc)
      map = self.mapGenerator(self.xdim,self.ydim,r,c,d,(sr,sc))
      return map

    def randomGrouping(self,k,u):
      map = self.obtainMap(k,u)
      grouped_image = np.arange(self.xdim*self.ydim).reshape(self.xdim,self.ydim)
      for i in range(self.xdim):
        for j in range(self.ydim):
          index = map[i][j]
          grouped_image[i][j] = self.image[index[0]][index[1]]
      self.image = grouped_image

    def reGrouping(self,k,u):
      map = self.obtainMap(k,u)
      regrouped_image = np.arange(self.xdim*self.ydim).reshape(self.xdim,self.ydim)
      for i in range(self.xdim):
        for j in range(self.ydim):
          index = map[i][j]
          regrouped_image[index[0]][index[1]] = self.image[i][j]
      return regrouped_image
      


    def sBoxConstruction(self,k,u):
      sBoxKey = list()
      for i in range(16):
        for j in range(15):
          k = self.logistic(k,u)
        k = self.logistic(k,u)
        sBoxKey.append(k)
      sBox = list()
      count = 0
      for i in range(16):
        chaoticSequence = list()
        k = sBoxKey[i]
        for j in range(500):
          k = self.logistic(k,u)
        for j in range(256):
          k = self.logistic(k,u)
          chaoticSequence.append((k,j))
        chaoticSequence.sort()
        for j in range(30):
          k = self.logistic(k,u)
        k = self.logistic(k,u)
        p = k
        k = self.logistic(k,u)
        q = k
        p1 = math.floor(p*(10**14))%16
        q1 = math.floor(q*(10**14))%16
        
        w, h = 16, 16;
        sBoxI = [[0 for x in range(w)] for y in range(h)]
        
        for val,pos in chaoticSequence:
          x,y = self.fourBits(pos)
          x1 = (x + p1*y)%16
          y1 = (q1*x + p1*q1*y +y)%16
          sBoxI[x1][y1] = pos
        sBox.append(sBoxI)
      return sBox



    def modifySBox(self,sBox):
      sBoxcopy = list()
      for i in range(16):
        sBoxIcopy = np.arange(self.xdim*self.ydim).reshape(self.xdim,self.ydim)
        for j in range(len(sBox[i])):
          for k in range(len(sBox[i][j])):
            index = self.fourBits(sBox[i][j][k])
            sBoxIcopy[index[0]][index[1]] = (j<<4) | k 
        sBoxcopy.append(sBoxIcopy)
      return sBoxcopy


    def randomSubstitution(self,k,u,sBox):
      cipher_image = np.arange(self.xdim*self.ydim).reshape(self.xdim,self.ydim)
      p = list()
      for i in range(500):
        k = self.logistic(k,u)
      for i in range(self.xdim):
        temp = list()
        for j in range(3):
          temp.append(k)
          k = self.logistic(k,u)
        p.append(temp)
        for k in range(10):
          k = self.logistic(k,u)
      for i in range(self.xdim):
        iter = []
        if i==0:
          iter.append(math.floor(p[i][0]*(10**14))%16 )
          iter.append(math.floor(p[i][1]*(10**14))%16 )
          iter.append(math.floor(p[i][2]*(10**14))%256)
        else:
          iter.append(((math.floor(p[i][0]*(10**14))%256)^cipher_image[i-1][self.ydim-1])%16 )
          iter.append(((math.floor(p[i][1]*(10**14))%256)^cipher_image[i-1][self.ydim-1])%16 )
          iter.append((math.floor(p[i][2]*(10**14))%256)^cipher_image[i-1][self.ydim-1])
        for j in range(self.ydim):
            if j==0:
              randBits = self.fourBits(int(iter[2]))
              imgbits = self.fourBits(self.image[i][j] ^ sBox[iter[1]][randBits[0]][randBits[1]])
              cipher_image[i][j] =sBox[iter[0]][imgbits[0]][imgbits[1]]
            else:
              randBits = self.fourBits(cipher_image[i][j-1])
              imgbits = self.fourBits(self.image[i][j] ^ sBox[iter[1]][randBits[0]][randBits[1]])
              cipher_image[i][j] =sBox[iter[0]][imgbits[0]][imgbits[1]]
      self.image = cipher_image


    def reverseSubstitution(self,k,u,sBox,sBoxReversed):
      plain_image = np.arange(self.xdim*self.ydim).reshape(self.xdim,self.ydim)
      p = list()
      for i in range(500):
        k = self.logistic(k,u)
      for i in range(self.xdim):
        temp = list()
        for j in range(3):
          temp.append(k)
          k = self.logistic(k,u)
        p.append(temp)
        for k in range(10):
          k = self.logistic(k,u)
      for i in range(self.xdim):
        iter = []
        if i==0:
          iter.append(math.floor(p[i][0]*(10**14))%16 )
          iter.append(math.floor(p[i][1]*(10**14))%16 )
          iter.append(math.floor(p[i][2]*(10**14))%256)
        else:
          iter.append(((math.floor(p[i][0]*(10**14))%256)^self.image[i-1][self.ydim-1])%16 )
          iter.append(((math.floor(p[i][1]*(10**14))%256)^self.image[i-1][self.ydim-1])%16 )
          iter.append((math.floor(p[i][2]*(10**14))%256)^self.image[i-1][self.ydim-1])
        for j in range(self.ydim):
            if j==0:
              randBits = self.fourBits(int(iter[2]))
              index = self.fourBits(self.image[i][j])
              imgbits =sBoxReversed[iter[0]][index[0]][index[1]]
              plain_image[i][j] = imgbits ^ sBox[iter[1]][randBits[0]][randBits[1]]
            else:
              randBits = self.fourBits(self.image[i][j-1])
              index = self.fourBits(self.image[i][j])
              imgbits =sBoxReversed[iter[0]][index[0]][index[1]]
              plain_image[i][j] = imgbits ^ sBox[iter[1]][randBits[0]][randBits[1]]
      self.image = plain_image
      
      
class SecureImage:

  sub_handle = None

  def __init__(self,image_path):
    image_object = Image.open(image_path).convert('RGB')
    self.image_rgb = np.asarray(image_object).copy()
    self.image_rgb.setflags(write=1)
    self.image_rgb[0][0][0] +=10
    print(self.image_rgb.shape)
    self.xdim,self.ydim,_ = self.image_rgb.shape
    self.image = np.concatenate((self.image_rgb[:,:,0],self.image_rgb[:,:,1],self.image_rgb[:,:,2]),axis=0)
    self.sub_handle = Substitution(self.image,self.image_rgb)
  
  
  def getKeys(self,zk,uk):
    return self.sub_handle.keyScheming(zk,uk)
  
  
  def encrypt(self,keys):
    self.sub_handle.randomGrouping(keys[0][0],keys[0][1])
    s_box = self.sub_handle.sBoxConstruction(keys[1][0],keys[1][1])
    self.sub_handle.randomSubstitution(keys[2][0],keys[2][1],s_box)
    cipher_image_array = self.sub_handle.reGrouping(keys[0][0],keys[0][1])
    cipher_image_array = np.uint8(cipher_image_array)
    print(cipher_image_array.shape)
    cipher_image_list = np.split(cipher_image_array,3)
    cipher_image_array = np.stack(cipher_image_list,axis=-1)
    print(cipher_image_array.shape)
    cipher_image_object = Image.fromarray(cipher_image_array)
    return cipher_image_object
  
  
  def decrypt(self,keys):
    self.sub_handle.randomGrouping(keys[0][0],keys[0][1])
    s_box = self.sub_handle.sBoxConstruction(keys[1][0],keys[1][1])
    s_box_reversed = self.sub_handle.modifySBox(s_box)
    self.sub_handle.reverseSubstitution(keys[2][0],keys[2][1],s_box,s_box_reversed)
    plain_image_array = self.sub_handle.reGrouping(keys[0][0],keys[0][1])
    plain_image_array = np.uint8(plain_image_array)
    plain_image_list = np.split(plain_image_array,3)
    plain_image_array = np.stack(plain_image_list,axis=-1)
    plain_image_object = Image.fromarray(plain_image_array)
    return plain_image_object
