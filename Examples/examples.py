from pymecht import *
from scipy.special import erfi
from matplotlib import pyplot as plt
import numpy as np

##################### Create material in different ways ########
def mat_creation():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    #mm[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]
    #material.models[0].fiber_dirs
    return material

###################### Uniaxial 
def unixex():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    print("Uniaxial")
    sample = UniaxialExtension(material,force_measure='force')
    params = sample.parameters
    print("Displacement controlled test")
    l_disp=np.linspace(1,2,10)
    print(l_disp,sample.disp_controlled(l_disp,params))
    l_for=np.linspace(0,2,10)
    print("Force controlled test")
    print(l_for,sample.force_controlled(l_for,params))
    return l_disp,sample.disp_controlled(l_disp,params),l_for,sample.force_controlled(l_for,params)

###################### Biaxial 
def biaxex():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    print("Biaxial")
    l_disp=np.linspace(1,2,10)
    sample=PlanarBiaxialExtension(material)
    params = sample.parameters
    print("Displacement controlled test")
    print(l_disp,sample.disp_controlled(l_disp,params))
    l_for=np.linspace(0,2,10)
    print("Force controlled test")
    print(l_for,sample.force_controlled(l_for,params))
    return l_disp,sample.disp_controlled(l_disp,params),l_for,sample.force_controlled(l_for,params)

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

    params['k1_0']=5.
    params['k2_0']=15.
    params['k3_0']=0.1
    params['mu_1']=1
    print("Force controlled test")
    forces = DOEobs(100,pi/4.).reshape(-1,2)
    print(forces,sample.force_controlled(forces,params))

######################## Validation of tube ###########
def validate_tube():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]
    sample = TubeInflation(material,force_measure='pressure')
    force_sol = sample.force_controlled(np.array([-0.29167718]),sample.parameters)
    print(force_sol)

    l=1.1
    result = lambda l: log(l)-1./2/l**2
    parameters = sample.parameters._val()
    parameters['k1_0']=5.
    parameters['k2_0']=15.
    parameters['k3_0']=0.
    parameters['mu_1']=2.
    #compare with the analytical solution
    Hbar=parameters['thick']/parameters['Ri']
    l2 = sqrt(1+(l**2-1)/(1+Hbar)**2)
    l1 = lambda l: sqrt(parameters['k2_0'])*(l**2-1)*cos(0.1)**2
    l12 = lambda l: sqrt(parameters['k2_0'])*(l**2-1)*cos(0.1)**2/(1+Hbar)**2
    analytical_sol = (erfi(l12(l))-erfi(l1(l)))*4*parameters['k1_0']*sqrt(pi)*cos(0.1)**2/4./sqrt(parameters['k2_0']) + (result(l2)-result(l))*parameters['mu_1'] #instead of a factor of 2 for 2 fibers, I had to use double (=4). Not sure why.
    print(analytical_sol)
    print((result(l2)-result(l))*parameters['mu_1'])

    #material = MatModel('nh')
    pymecht_sol = sample.disp_controlled([1.1],parameters)
    print(pymecht_sol)
    return abs(analytical_sol), pymecht_sol[0], force_sol[0]

############################# Using tube to calculate and plot stresses
def artery0Dmodel():
    def von_mises(sigma_list):
        return [sqrt(3./2.)*np.linalg.norm(sigma-np.trace(sigma)/3.*np.eye(3)) for sigma in sigma_list]

    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]

    sample = TubeInflation(material,force_measure='pressure')
    params = sample.parameters
    params['k1_0']=5.
    params['k2_0']=15.
    params['k3_0']=0.1
    params['mu_1']=1.
    print(sample.disp_controlled([1.1],params))
    print(sample.force_controlled(np.array([0.27494258]),params))

    xi,stress = sample.cauchy_stress([1.1],params,n=11)
    print(params)
    #print(list(zip(xi,von_mises(stress))))

    plt.plot(xi,von_mises(stress))
    plt.xlabel('Normalized thickness')
    plt.ylabel('von-Mises stress')
    plt.show()

    intima = TubeInflation(material)
    media = TubeInflation(material)
    artery = LayeredTube(intima,media)
    combined_parameters = artery.parameters
    combined_parameters[0]['Ri']=1.0
    combined_parameters[0]['thick']=0.01
    combined_parameters[1]['Ri']=1.0
    combined_parameters[1]['thick']=0.05
    xi,stress = artery.cauchy_stress(np.array([1.1]),combined_parameters)

    plt.plot(xi,von_mises(stress),'-o')
    plt.xlabel('Normalized thickness')
    plt.ylabel('von-Mises stress')
    plt.show()

###################### Creating random parameters
def randomex():
    model = MatModel('goh','nh')
    mm = model.models
    Theta = RandomParameters(model.parameters)
    Theta.make_normal('mu_1')
    #Theta.fix('k2',10)
    #Theta.fix('k1',10)
    #Theta.parameters
    #Theta.sample(10)

    #Theta.sample(10)
    #Theta.sample(10,'lhcube')
    #Theta.sample(10,'sobol')
    t = Theta.sample(1)
    print(Theta.prob(t[0]))

if __name__=="__main__":
    mat_creation()
    unixex()
    biaxex()
    validate_tube()
    #artery0Dmodel()
    randomex()
