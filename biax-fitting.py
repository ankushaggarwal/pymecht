import matplotlib.pylab as plt
import numpy as np
from pymecht import *
from matplotlib import cm
from itertools import cycle

def initialiseVals(_var, _init_guess=1., _lower_bound=-1000., _upper_bound=1000.):
    global init_guess_string, lower_bound_string, upper_bound_string
    init_guess_string += _var + " = %s, " %(_init_guess)
    lower_bound_string += _var + " = %s, " %(_lower_bound)
    upper_bound_string += _var + " = %s, " %(_upper_bound)

x = "(I1-3.)"
y = "(sqrt(I4)-1.)"

data = np.load('biax-data.npz')
inp = data['inp'] # stretches
out = data['out'] # stresses
protocols = data['protocols']

# SHOULD MAKE A LIBRARY THAT INCLUDES:
# 1, x, y, x*y, x^2, y^2, x^3, y^3, etc.
# STANDARD MODELS (YEOH & NH ALREADY THERE)
# GOH   = (k1GOH/(2.*k2GOH))*(exp(k2GOH*(k3GOH*I1+(1.-3.*k3GOH)*I4-1)**2.)-1.)
# HGO   = (k1HGO/(2.*k2HGO))*(exp(k2HGO*(I4-1.)**2.)-1.)
# expI1 = (k3HGO/(2.*k4HGO))*(exp(k4HGO*(I1-3.))-1.)
# Holz. = (k1Holz/(2.*k2Holz))*(exp(k2Holz*(k3Holz*(I1-3.)**2.+(1.-k3Holz)*I4**2.))-1.)
# HY    = k3HY/k4HY*(exp(k4HY*(sqrt(I4)-1.)**2.)-1.))
# LS    = k1LS/2./(k4LS*k2LS+(1.0-k4LS)*k3LS)*(k4LS*exp(k2LS*(I1-3.0)**2.)+(1.0-k4LS)*exp(k3LS*(I4-1.0)**2.0)-1.0)
# MN    = k1MN/(k2MN+k3MN)*(exp(k2MN*(I1-3.)**2.+k3MN*(sqrt(I4)-1.)**4.)-1.)
# AND SOME NEW FORMS
new_forms = []
# ["ltheta*x**nltheta", "ltheta*y**nltheta", "ltheta*(x*y)**nltheta", \
#              "ltheta*x**nltheta*y", "ltheta*x*y**nltheta", \
#              "ltheta*exp(nltheta*x**nltheta)", "ltheta*exp(nltheta*y**nltheta)", \
#              "ltheta*exp(nltheta*x**nltheta)"]
# 

# ltheta_00=1., ltheta_01=1., ltheta_02=1., theta_03=1., theta_10=1., theta_11=1., theta_12=1., theta_13=1., theta_20=1., theta_21=1., theta_22=1., theta_23=1., theta_30=1., theta_31=1., theta_32=1., theta_33=1., k1GOH=1., k2GOH=1., k3GOH=1., k1HGO=1., k2HGO=1., k3HGO=1., k4HGO=1., k1Holz=1., k2Holz=1., k3Holz=1., k3HY=1., k4HY=1., k1LS=1., k2LS=1., k3LS=1., k4LS=1., k1MN=1., k2MN=1., k3MN=1.

W_string = ""
init_guess_string = ""
lower_bound_string = ""
upper_bound_string = ""

for i in range(0,4):
    for j in range(0,4):
        W_string += "ltheta_%s%s*x**%s*y**%s + " %(i,j,i,j)
        initialiseVals("ltheta_%s%s"%(i,j))

W_string = W_string.replace("*x**0","")
W_string = W_string.replace("*y**0","")
W_string = W_string.replace("x**1","x")
W_string = W_string.replace("y**1","y")
W_string = W_string.replace("x",x)
W_string = W_string.replace("y",y)

print("Initial params = ", init_guess_string[17:-2] +"\n")
print("Lower bound = ", lower_bound_string[21:-2] +"\n")
print("Upper bound = ", upper_bound_string[20:-2] +"\n")
print("Energy form = ", W_string[12:-3] +"\n")

mat = 'sparse_fit'
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
elif mat == 'arb':
    material = MatModel('arb')
elif mat == 'sparse_fit':
    material = MatModel('goh','hgo','expI1','Holzapfel','hy','ls','mn','arb')

import time
start = time.time()

mm = material.models
for model in mm:
    model.fiber_dirs = [np.array([1,0,0])]
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

res = sample.disp_controlled(inp,c_all) # For stretches in

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

end = time.time()
print("Time spent evaluating: ",end - start)