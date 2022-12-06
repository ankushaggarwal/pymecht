from SampleExperiment import *
from MatModel import *

import numpy as np
from math import cos,sin,exp,sqrt,pi,tan,log
import scipy.optimize as opt

#For the DOE experiment
def DOEobs(npoints,theta):
    smax = 200.
    times = np.linspace(0.,1.,npoints)

    if theta<pi/4.:
        s11max,s22max = smax,smax*tan(theta)
    else:
        s11max,s22max = smax*tan(pi/2.-theta),smax

    stresses = []
    for i in range(npoints):
        stresses += [s11max*times[i],s22max*times[i]]
    return np.array(stresses)

material = MatModel('goh','nh')
mm = material.models
mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]

print("Biaxial")
sample = PlanarBiaxialExtension(material,disp_measure='Delta L')
params = sample.parameters
print(params)
params['k1_0']=5.
params['k2_0']=15.
params['k3_0']=0.1
params['mu_1']=1
print("Force controlled test")
forces = DOEobs(100,pi/4.).reshape(-1,2)
print(forces,sample.force_controlled(forces,params))


mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]
sample = UniformAxisymmetricTubeInflationExtension(material,force_measure='pressure')
print(sample.disp_controlled([1.1],sample.parameters))
print(sample.force_controlled(np.array([-0.29167718]),sample.parameters))


from math import *
l=1.1
result = lambda l: log(l)-1./2/l**2
parameters = sample.parameters
parameters['k1_0']=5.
parameters['k2_0']=15.
parameters['k3_0']=0.
parameters['mu_1']=2.
#compare with the analytical solution
Hbar=parameters['thick']/parameters['Ri']
l2 = sqrt(1+(l**2-1)/(1+Hbar)**2)
l1 = lambda l: sqrt(parameters['k2_0'])*(l**2-1)*cos(0.1)**2
l12 = lambda l: sqrt(parameters['k2_0'])*(l**2-1)*cos(0.1)**2/(1+Hbar)**2
from scipy.special import erfi
print((erfi(l12(l))-erfi(l1(l)))*4*parameters['k1_0']*sqrt(pi)*cos(0.1)**2/4./sqrt(parameters['k2_0']) + (result(l2)-result(l))*parameters['mu_1']) #instead of a factor of 2 for 2 fibers, I had to use double (=4). Not sure why.
print((result(l2)-result(l))*parameters['mu_1'])

#material = MatModel('nh')
sample = UniformAxisymmetricTubeInflationExtension(material,force_measure='pressure')
print(sample.disp_controlled([1.1],parameters))
