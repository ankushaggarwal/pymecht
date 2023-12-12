import matplotlib
matplotlib.use('Agg')
import multiprocessing as mp
from MatModel import *
from SampleExperiment import *
from RandomParameters import *
import sys
import scipy.stats
R1=scipy.stats.norm(0.95, 0.05)
F1=scipy.stats.norm(0.1,0.03)
R1=scipy.stats.norm(0.95, 0.03)
F2=scipy.stats.norm(0.05,0.02)

outfile = sys.argv[-1]

def cal_prob(r1,r2,r3,r4):
    f1,f2=r2/r1-1.,r4/r3-1.
    return R1.pdf(r1)*F1.pdf(f1)*F2.pdf(f2)

diasP=0.03
sysP=0.08

r0=1.2
def cal_output(params):
    #print(params)
    #t = params['phi']
    #mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]
    try:
        r1,r2 = sample.force_controlled(np.array([diasP,sysP]),params,params['Ri'])
    except RuntimeError:
        print(params,"Not converged, returning nan probability")
        return np.nan
    r3 = r0
    pstent = sample.disp_controlled([r0],params)
    try:
        r4 = sample.force_controlled(pstent+sysP-diasP,params,x0=r0)[0]
    except RuntimeError:
        print(params,"Not converged, return nan probability")
        return np.nan
    return cal_prob(r1,r2,r3,r4)

material = MatModel('yeoh')
mm = material.models

sample = TubeInflation(material,force_measure='pressure')
#params = sample.parameters
Theta = RandomParameters(*sample.parameters_wbounds())
#Theta.add('phi',0.1,0,pi/2.,'uniform')

total = 0.
N=2**18
theta_samples = Theta.sample(N,'sobol')

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
#result = pool.imap(cal_output, theta_samples)
results_all = pool.map(cal_output,theta_samples)

results_all = np.array(results_all)
theta_samples = np.array(theta_samples)

theta_samples_nonconverged = theta_samples[np.isnan(results_all)]
theta_samples = theta_samples[~np.isnan(results_all)]
results_all = results_all[~np.isnan(results_all)]


print(N-len(results_all),'did not converge. Ignoring them')
N = len(results_all)
print(np.cumsum(results_all)[-1]/N)

#plot the von-Mises stress
from matplotlib import pyplot as plt
plt.plot(np.cumsum(results_all)/np.arange(1,N+1))
plt.gca().set_xlim(left=10000)
plt.xlabel('Samples')
plt.ylabel('Probability')
plt.savefig(outfile+'.pdf')

np.savez(outfile,theta=theta_samples,theta_nonconverged=theta_samples_nonconverged,results=results_all)
