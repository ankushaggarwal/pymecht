import matplotlib.pylab as plt
import numpy as np
from pymecht import *
from matplotlib import cm
from itertools import cycle

data = np.load('biax-data.npz')
inp = data['inp']
out = data['out']
protocols = data['protocols']

mat = 'ls'
if mat=='yeoh':
    material = MatModel('yeoh')
elif mat == 'goh':
    material = MatModel('goh','nh')
elif mat == 'ls':
    material = MatModel('ls','nh')
elif mat == 'mn':
    material = MatModel('mn','nh')
elif mat == 'hy':
    material = MatModel('hy','expI1')
elif mat == 'holzapfel':
    material = MatModel('Holzapfel','nh')
elif mat == 'hgo':
    material = MatModel('hgo','nh')
elif mat == 'hgo2':
    material = MatModel('hgo','expI1')

mm = material.models
mm[0].fiber_dirs = [np.array([1,0,0])]
sample = PlanarBiaxialExtension(material,disp_measure='stretch',force_measure='1pk')
sample_value_bounds = sample.parameters_wbounds()

if mat=='yeoh':
    sample_value_bounds[2]['c1_0']=40.
    sample_value_bounds[2]['c2_0']=40.
    sample_value_bounds[2]['c3_0']=40.
elif mat=='goh':
    sample_value_bounds[2]['k1_0']=100.
    sample_value_bounds[2]['k2_0']=40.
    sample_value_bounds[2]['mu_1']=30.
elif mat == 'ls':
    sample_value_bounds[2]['k2_0']=40.
    sample_value_bounds[2]['k3_0']=40.
    sample_value_bounds[2]['mu_1']=30.
elif mat == 'mn':
    sample_value_bounds[2]['k2_0']=40.
    sample_value_bounds[2]['k3_0']=40.
    sample_value_bounds[2]['mu_1']=30.
elif mat == 'hy':
    sample_value_bounds[2]['k2_1']=40.
    sample_value_bounds[2]['k4_0']=40.
elif mat == 'holzapfel':
    sample_value_bounds[2]['k2_0']=40.
    sample_value_bounds[2]['mu_1']=30.
elif mat == 'hgo':
    sample_value_bounds[2]['k3_0']=40.
    sample_value_bounds[2]['mu_1']=30.
elif mat == 'hgo2':
    sample_value_bounds[2]['k2_1']=40.
    sample_value_bounds[2]['k4_0']=40.

c_all,c_low,c_high = sample_value_bounds

c_fix  = c_all.copy()
for key, value in c_all.items():
    if key not in ['L10','L20','thick']:
        c_fix[key]=False
    else:
        c_fix[key]=True

def complete_params(cval,c_all,c_fix):
    i=0
    for key,value in c_all.items():
        if not c_fix[key]:
            try:
                c_all[key] = cval[i]
                i += 1
            except IndexError as err:
                print("Non-fixed parameters and cval are of different length",err)
    return

def residual(c,c_all,c_fix,measure):
    complete_params(c,c_all,c_fix)
    x = sample.disp_controlled(inp,c_all)
    return (x-measure).flatten()

from scipy.optimize import least_squares

c0 = np.array([value for key, value in c_all.items() if not c_fix[key]])
low  = np.array([value for key, value in c_low.items() if not c_fix[key]])
high = np.array([value for key, value in c_high.items() if not c_fix[key]])
bounds = (low,high)
result = least_squares(residual,x0=c0,args=(c_all,c_fix,out),bounds=bounds)

res = sample.disp_controlled(inp,c_all)

colors = cycle(cm.rainbow(np.linspace(0, 1,len(set(protocols)))))
fig,(ax1,ax2) = plt.subplots(2,1)
for i in set(protocols):
    cl = next(colors)
    subset = protocols==i
    ax1.plot(inp[subset][:,0],out[subset][:,0],'o',color=cl)
    ax1.plot(inp[subset][:,0],res[subset][:,0],'-',color=cl)
    ax2.plot(inp[subset][:,1],out[subset][:,1],'o',color=cl)
    ax2.plot(inp[subset][:,1],res[subset][:,1],'-',color=cl)

plt.show()
