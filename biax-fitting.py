import matplotlib.pylab as plt
import numpy as np
from pymecht import *
from matplotlib import cm
from itertools import cycle
import pandas as pd

def initialiseVals(_var, _init_guess=1., _lower_bound=-1000., _upper_bound=1000.):
    global init_guess_string, lower_bound_string, upper_bound_string
    init_guess_string += _var + " = %s, " %(_init_guess)
    lower_bound_string += _var + " = %s, " %(_lower_bound)
    upper_bound_string += _var + " = %s, " %(_upper_bound)

biax_data = False
constI1 = True
constI4 = True
constI6 = False

tol = 0.1 # tolerance of biggest linear parameter to smallest. Removes lowest (tol)*100% of terms

X = "(I1-3.)"
Y = "(sqrt(I4)-1.)"

# SHOULD MAKE A LIBRARY THAT INCLUDES:
# X, Y, X*Y, X^2, Y^2, X^3, Y^3, etc.
# STANDARD MODELS (YEOH & NH ALREADY THERE)
# GOH   = (k1GOH/(2.*k2GOH))*(exp(k2GOH*(k3GOH*I1+(1.-3.*k3GOH)*I4-1)**2.)-1.)
# HGO   = (k1HGO/(2.*k2HGO))*(exp(k2HGO*(I4-1.)**2.)-1.)
# expI1 = (k3HGO/(2.*k4HGO))*(exp(k4HGO*(I1-3.))-1.)
# Holz. = (k1Holz/(2.*k2Holz))*(exp(k2Holz*(k3Holz*(I1-3.)**2.+(1.-k3Holz)*I4**2.))-1.)
# HY    = k3HY/k4HY*(exp(k4HY*(sqrt(I4)-1.)**2.)-1.))
# LS    = k1LS/2./(k4LS*k2LS+(1.0-k4LS)*k3LS)*(k4LS*exp(k2LS*(I1-3.0)**2.)+(1.0-k4LS)*exp(k3LS*(I4-1.0)**2.0)-1.0)
# MN    = k1MN/(k2MN+k3MN)*(exp(k2MN*(I1-3.)**2.+k3MN*(sqrt(I4)-1.)**4.)-1.)
# AND SOME NEW FORMS
new_forms = ["ltheta*exp(nltheta*X)*Y**2.", "ltheta*exp(nltheta*X)*Y", "ltheta*exp(nltheta*X)",\
             "ltheta*exp(nltheta*Y)*X**2.", "ltheta*exp(nltheta*Y)*X", "ltheta*exp(nltheta*Y)",\
             "ltheta*X**nlthetap",\
             "ltheta*X**nlthetap*Y**nlthetap",\
             "ltheta*X**nlthetap*Y",\
             "ltheta*X*Y**nlthetap",\
             "ltheta*exp(nltheta*X**nlthetap)",\
             "ltheta*exp(nltheta*Y**nlthetap)"]

# ltheta_00=1., ltheta_01=1., ltheta_02=1., theta_03=1., theta_10=1., theta_11=1., theta_12=1., theta_13=1., theta_20=1., theta_21=1., theta_22=1., theta_23=1., theta_30=1., theta_31=1., theta_32=1., theta_33=1., k1GOH=1., k2GOH=1., k3GOH=1., k1HGO=1., k2HGO=1., k3HGO=1., k4HGO=1., k1Holz=1., k2Holz=1., k3Holz=1., k3HY=1., k4HY=1., k1LS=1., k2LS=1., k3LS=1., k4LS=1., k1MN=1., k2MN=1., k3MN=1.

W_string = ""
init_guess_string = ""
lower_bound_string = ""
upper_bound_string = ""
L_params = []

for i in range(0,2):
    for j in range(0,4):
        if (i==0) and (j==0):
            pass
        else:
            W_string += "ltheta_%s_%s*X**%s*Y**%s + " %(i,j,i,j)
            initialiseVals("ltheta_%s_%s"%(i,j))
            L_params += ["ltheta_%s_%s"%(i,j)]

L_params = L_params[1:]

for counter, form in enumerate(new_forms):
    if "nltheta*" in form:
        initialiseVals("nltheta%s"%(counter))
        form = form.replace("nltheta*","nltheta%s*" %(counter))
    if "**nlthetap" in form:
        initialiseVals("nlthetap%s"%(counter),2.0,1.0)
        form = form.replace("**nlthetap","**nlthetap%s" %(counter))
    # if "X**nlthetap" in form:
    #     initialiseVals("nlthetap%s"%(counter),2.0,1.0)
    #     form = form.replace("**nlthetap","**nlthetap%s" %(counter))
    # if "Y**nlthetap" in form:
    #     initialiseVals("nlthetap%s"%(counter),2.5,2.0)
    #     form = form.replace("**nlthetap","**nlthetap%s" %(counter))
    if "ltheta*" in form:
        initialiseVals("ltheta%s"%(counter))
        form = form.replace("ltheta*","ltheta%s*" %(counter))
        L_params += ["ltheta%s" %(counter)]
    W_string += form+" + "

W_string = W_string.replace("*X**0","")
W_string = W_string.replace("*Y**0","")
W_string = W_string.replace("X**1","X")
W_string = W_string.replace("Y**1","Y")
W_string = W_string.replace("X",X)
W_string = W_string.replace("Y",Y)

# W_string += "(k1GOH/(2.*k2GOH))*(exp(k2GOH*(k3GOH*I1+(1.-3.*k3GOH)*I4-1)**2.)-1.)"
# W_string += "(k1HGO/(2.*k2HGO))*(exp(k2HGO*(I4-1.)**2.)-1.)"
# W_string += "(k3HGO/(2.*k4HGO))*(exp(k4HGO*(I1-3.))-1.)"
# W_string += "(k1Holz/(2.*k2Holz))*(exp(k2Holz*(k3Holz*(I1-3.)**2.+(1.-k3Holz)*I4**2.))-1.)"
# W_string += "k3HY/k4HY*(exp(k4HY*(sqrt(I4)-1.)**2.)-1.))"
# W_string += "k1LS/2./(k4LS*k2LS+(1.0-k4LS)*k3LS)*(k4LS*exp(k2LS*(I1-3.0)**2.)+(1.0-k4LS)*exp(k3LS*(I4-1.0)**2.0)-1.0)"
# W_string += "k1MN/(k2MN+k3MN)*(exp(k2MN*(I1-3.)**2.+k3MN*(sqrt(I4)-1.)**4.)-1.)"

print("Initial params = ", init_guess_string[:-2] +"\n")
print("Lower bound = ", lower_bound_string[:-2] +"\n")
print("Upper bound = ", upper_bound_string[:-2] +"\n")
print("Energy form = ", W_string[:-3] +"\n")

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

df = pd.read_excel(io='/mnt/WD_Black/Aggarwal_postdoc/ross-temp/ConstantInvariant_DrAggarwal_0517212.xlsx',sheet_name='TVAL1',header=[0,1])
df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')

inp = np.empty((0,2))
out = np.empty((0,2))
protocols = np.array([])

ranges = []
if biax_data == True:
    ranges+= [1,2,3,4,5,6,7]
if constI4 == True:
    ranges+= [8,9,10,11]
if constI6 == True:
    ranges+= [12,13,14,15]
if constI1 == True:
    ranges+= [16,17,18,19]

for protocol in ranges:
    subset = (df['Protocol']== protocol) & (df["L/U"] == 1)
    inp = np.append(inp, np.transpose([df['Tine']['λ_1'][subset].to_numpy(),df['Tine']['λ_2'][subset].to_numpy()]), axis=0)
    out = np.append(out, np.transpose([df['Applied']['P11'][subset].to_numpy(),df['Applied']['P22'][subset].to_numpy()]), axis=0)
    protocols = np.append(protocols, np.transpose(df['Protocol'][subset].to_numpy()))

# inp = data['inp'] # stretches
# out = data['out'] # stresses
# protocols = data['protocols']

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
elif mat == 'sparse_fit':
    #SORT BOUNDS
    L_params += ['k1_0']
    L_params += ['k3_1']
    L_params += ['k1_2']
    L_params += ['k1_3']
    L_params += ['k3_4']
    L_params += ['k1_5']
    L_params += ['k1_6']

c_all,c_low,c_high = sample_value_bounds

fixed_params = ['L10','L20','thick'] # WILL NEED OTHER LINEAR PARAMS AFTER SPARSITY CHECK
c_fix  = c_all.copy()

params_removed = 1

from scipy.optimize import least_squares

# WILL START LOOP HERE
while params_removed != 0:
    for key, value in c_all.items():
        if key not in fixed_params:
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
    
    c0 = np.array([value for key, value in c_all.items() if not c_fix[key]])
    low  = np.array([value for key, value in c_low.items() if not c_fix[key]])
    high = np.array([value for key, value in c_high.items() if not c_fix[key]])
    bounds = (low,high)
    result = least_squares(residual,x0=c0,args=(c_all,c_fix,out),bounds=bounds)
    
    res = sample.disp_controlled(inp,c_all) # For stretches in
    
    #%%
    
    colors = cycle(cm.rainbow(np.linspace(0, 1,len(set(protocols)))))
    fig,(ax1,ax2) = plt.subplots(2,1)
    for i in set(protocols):
        cl = next(colors)
        subset = protocols==i
        ax1.plot(inp[subset][:,0],out[subset][:,0],'o',color=cl)
        ax1.plot(inp[subset][:,0],res[subset][:,0],'-',color=cl)
        ax2.plot(inp[subset][:,1],out[subset][:,1],'o',color=cl)
        ax2.plot(inp[subset][:,1],res[subset][:,1],'-',color=cl)
    
    ax1.set(xlabel='$\lambda_1$', ylabel='$P_{11}$')
    ax2.set(xlabel='$\lambda_2$', ylabel='$P_{22}$')
    
    fig.tight_layout()
    
    plt.show()
    
    #%%
    
    fixed_params_len_old = len(fixed_params)
    
    for var in c_all:
        if (abs(c_all[var]) < tol*abs(max(c_all.values(),key=abs))) and (c_fix[var]==False):
            fixed_params += [var]
            c_all[var] = 0.0
            # update bounds?
    params_removed = len(fixed_params) - fixed_params_len_old
    # REPEAT LOOP HERE

end = time.time()
print("Time spent evaluating: ",end - start)