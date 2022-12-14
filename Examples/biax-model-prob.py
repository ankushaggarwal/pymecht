from MatModel import *
from SampleExperiment import *
###############################################################################
############################# ARTIFICIAL DATA #################################
###############################################################################
material = MatModel('goh','nh')
mm = material.models
mm[0].fiber_dirs = [np.array([1,0,0])]
sample = PlanarBiaxialExtension(material,disp_measure='length')
params = sample.parameters
params['k1']=5.
params['k2']=15.
params['k3']=0.1
params['mu']=1

#create stretches for 5 protocol angles
deltalmax=0.1
npoints=30
stretches = []
nprotocol = 4
times = np.linspace(0.,1.,npoints+1)[1:]

def add_stretches(theta,stretches):
    if theta<pi/4.:
        l1max,l2max = deltalmax,deltalmax*tan(theta)
    else:
        l1max,l2max = deltalmax*tan(pi/2.-theta),deltalmax
    for i in range(npoints):
        stretches += [l1max*times[i],l2max*times[i]]
    return stretches

for theta in np.linspace(0,pi/2.,nprotocol):
    stretches = add_stretches(theta,stretches)

stretches = np.array(stretches)+1
#calculate the stresses
stress = sample.disp_controlled(stretches,params)

from RandomParameters import *
var=0.1
params_low = params.copy()
params_low['k1'] *= (1-var) 
params_low['k2'] *= (1-var) 
params_low['k3'] *= (1-var) 
params_low['mu'] *= (1-var) 
params_up = params.copy()
params_up['k1'] *= (1+var) 
params_up['k2'] *= (1+var) 
params_up['k3'] *= (1+var) 
params_up['mu'] *= (1+var) 
Theta = RandomParameters(params,params_low,params_up)
Theta.fix('L10')
Theta.fix('L20')
Theta.fix('thick')
#Theta.make_normal('k1')
#Theta.make_normal('k2')
#Theta.make_normal('k3')
#Theta.make_normal('mu')
#print(Theta)
N=10
theta_samples = Theta.sample(N)

#generate artificial data
stresses=[]
for i in range(N):
    stresses.append(sample.disp_controlled(stretches,theta_samples[i]))

#calculate mean and covariance matrix
mean_stress = sum(stresses)/len(stresses)
cov_stress = np.cov(np.array(stresses).T)
noise = np.diag(mean_stress*0.05)
cov_stress += noise

#create observations
from scipy.stats import multivariate_normal
obs = multivariate_normal(mean_stress,cov_stress)

print("Created observations. Calculate model probability")
###############################################################################
############################# MODEL PROB CALC #################################
###############################################################################
material = MatModel('yeoh')
#material = MatModel('goh','nh')
mm = material.models
mm[0].fiber_dirs = [np.array([1,0,0])]
sample = PlanarBiaxialExtension(material,disp_measure='length')
Theta = RandomParameters(*sample.parameters_wbounds())
Theta.fix('L10')
Theta.fix('L20')
Theta.fix('thick')
#Theta.add('phi',0.1,0,pi/2.,'uniform')

def cal_prob(params):
   stress = sample.disp_controlled(stretches,params)
   return obs.pdf(stress)

total = 0.
N=1000
theta_samples = Theta.sample(N,'sobol')

#print(mp.cpu_count())
#pool = mp.Pool(mp.cpu_count())
##result = pool.imap(cal_output, theta_samples)
#results_all = pool.map(cal_output,theta_samples)
results_all = []
for i in range(N):
    print(i)
    results_all.append(cal_prob(theta_samples[i]))

results_all = np.array(results_all)
theta_samples = np.array(theta_samples)

theta_samples_nonconverged = theta_samples[np.isnan(results_all)]
theta_samples = theta_samples[~np.isnan(results_all)]
results_all = results_all[~np.isnan(results_all)]

print(N-len(results_all),'did not converge. Ignoring them')
N = len(results_all)
print(np.cumsum(results_all)[-1]/N)
