#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
L = [] 
n = [16,18,9,25,1]
N = (len(n)-1)//2
n = np.arange(-N,N+1)
l = []
tim = [i for i in np.linspace(-3,3,1000)]
for i in n:   
    for t in tim:
        term =2.5
        for k in range(-N, N+1): 
            if k!=i:
                term = term*((t-k)/(i-k)) 
        l.append(term)
    L.append(l)
    l=[]
for i in range(len(L)):
    plt.plot(tim , L[i])
plt.show()


# In[16]:


import thinkdsp as t
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import thinkplot as d
from IPython.display import Audio
whis = t.read_wave('C:/Users/This PC/Downloads/my.wav')
ALLE= whis.segment(start = 0, duration = 5)
SPECTRUM = ALLE.make_spectrum()
SPECTRUM.plot(high = 1500)
SPECTRUM.low_pass(300)
SPECTRUM.make_wave().make_audio()


# In[5]:


import matplotlib.pyplot as p
import numpy as np

w1 = [5* np.sin(2*np.pi*16*t) for t in np.linspace(0,1,1000)]
w2 = [5* np.cos(2*np.pi*6*t) for t in np.linspace(0,1,1000)]
w3 = [5* np.sin(2*np.pi*2000*t) for t in np.linspace(0,1,1000)]
sig=[]

for i,j,k in zip(w1,w2,w3):
    sig.append(i+j+k)
    


gt = np.fft.fft(sig)
p.plot(gt)


# In[ ]:





# In[ ]:




