from SampleExperiment import *
from MatModel import *
from LinearFEM import *
from RandomParameters import *
from scipy.special import erfi
from matplotlib import pyplot as plt

##################### Create material in different ways ########
def mat_creation():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    #mm[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]
    #material.models[0].fiber_dirs

###################### Uniaxial 
def unixex():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    print("Uniaxial")
    sample = UniaxialExtension(material)
    params = sample.parameters
    print("Displacement controlled test")
    l=np.linspace(1,2,10)
    print(l,sample.disp_controlled(l,params))
    l=np.linspace(0,2,10)
    print("Force controlled test")
    print(l,sample.force_controlled(l,params))

###################### Biaxial 
def biaxex():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    print("Biaxial")
    l=np.linspace(1,2,10)
    sample=PlanarBiaxialExtension(material)
    params = sample.parameters
    print("Displacement controlled test")
    print(l,sample.disp_controlled(l,params))
    l=np.linspace(0,2,10)
    print("Force controlled test")
    print(l,sample.force_controlled(l,params))

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

    params['k1']=5.
    params['k2']=15.
    params['k3']=0.1
    params['mu']=1
    print("Force controlled test")
    forces = DOEobs(100,pi/4.).reshape(-1,2)
    print(forces,sample.force_controlled(forces,params))

######################## Validation of tube ###########
def validate_tube():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]
    sample = UniformAxisymmetricTubeInflationExtension(material,force_measure='pressure')
    print(sample.disp_controlled([1.1],sample.parameters))
    print(sample.force_controlled(np.array([-0.29167718]),sample.parameters))

    l=1.1
    result = lambda l: log(l)-1./2/l**2
    parameters = sample.parameters
    parameters['k1']=5.
    parameters['k2']=15.
    parameters['k3']=0.
    parameters['mu']=2.
    #compare with the analytical solution
    Hbar=parameters['thick']/parameters['Ri']
    l2 = sqrt(1+(l**2-1)/(1+Hbar)**2)
    l1 = lambda l: sqrt(parameters['k2'])*(l**2-1)*cos(0.1)**2
    l12 = lambda l: sqrt(parameters['k2'])*(l**2-1)*cos(0.1)**2/(1+Hbar)**2
    print((erfi(l12(l))-erfi(l1(l)))*4*parameters['k1']*sqrt(pi)*cos(0.1)**2/4./sqrt(parameters['k2']) + (result(l2)-result(l))*parameters['mu']) #instead of a factor of 2 for 2 fibers, I had to use double (=4). Not sure why.
    print((result(l2)-result(l))*parameters['mu'])

    #material = MatModel('nh')
    sample = UniformAxisymmetricTubeInflationExtension(material,force_measure='pressure')
    print(sample.disp_controlled([1.1],parameters))

############################# Using tube to calculate and plot stresses
def artery0Dmodel():
    def von_mises(sigma_list):
        return [sqrt(3./2.)*np.linalg.norm(sigma-np.trace(sigma)/3.*np.eye(3)) for sigma in sigma_list]

    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]

    sample = UniformAxisymmetricTubeInflationExtension(material,force_measure='pressure')
    params = sample.parameters
    params['k1']=5.
    params['k2']=15.
    params['k3']=0.1
    params['mu']=1.
    print(sample.disp_controlled([1.1],params))
    print(sample.force_controlled(np.array([0.27494258]),params))

    xi,stress = sample.cauchy_stress([1.1],params,n=11)
    print(params)
    #print(list(zip(xi,von_mises(stress))))

    plt.plot(xi,von_mises(stress))
    plt.xlabel('Normalized thickness')
    plt.ylabel('von-Mises stress')
    plt.show()

    intima = UniformAxisymmetricTubeInflationExtension(material)
    media = UniformAxisymmetricTubeInflationExtension(material)
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
    t,t1,t2 = model.parameters_wbounds()
    Theta = RandomParameters(t,t1,t2)
    Theta.make_normal('mu')
    #Theta.fix('k2',10)
    #Theta.fix('k1',10)
    #Theta.parameters
    #Theta.sample(10)

    #Theta.sample(10)
    #Theta.sample(10,'lhcube')
    #Theta.sample(10,'sobol')
    t = Theta.sample(1)
    print(Theta.prob(t[0]))

######################### FEM example
def femex():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([1,0,0])]
    sample = UniaxialExtension(material,disp_measure='stretch',force_measure='1stPK')
    params = sample.parameters
    def uniax(dy,**extra_args):
        P = sample.disp_controlled([dy],params)[0]
        return 0,0,P+1

    x=np.linspace(0,1,10)
    femodel = LinearFEM1D(x,DOF='equal')
    femodel.compute=uniax
    femodel.assemble()
    femodel.fglobal
    femodel.Kglobal
    BC=np.zeros_like(x,dtype=bool)
    BC[0]=True
    #BC[-1]=True
    for i in range(5):
        femodel.newton_step(BC)
        print(femodel.dof)


if __name__=="__main__":
    mat_creation()
    unixex()
    biaxex()
    validate_tube()
    #artery0Dmodel()
    randomex()
    femex()
