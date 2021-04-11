import numpy as np
from math import sqrt, pi
from functools import partial
import scipy.optimize as opt
import warnings
from scipy.integrate import quad

class SampleExperiment:
    '''
    An abstract class, which defines the sample geometry and experimental conditions.
    It also defines the coordinate system to be used.
    '''
    def __init__(self,mat_model,disp_measure,force_measure):
        self.mat_model = mat_model
        self.inp = disp_measure.replace(" ","").lower()
        self.output = force_measure.replace(" ","").lower()
        params = {}

    def disp_controlled(self,input_,params):
        self.update(**params)
        output = [self.observe(self.compute(F,params)) for F in self.F(input_)]
        return np.array(output).reshape(np.shape(input_))

    def force_controlled(self,forces,params):
        self.update(**params)
        
        def compare(displ,ybar,params):
            return self.disp_controlled([displ],params)[0]-ybar

        #solve for the input_ by solving the disp_controlled minus desired output
        forces_temp = forces.reshape(-1,self.ndim)
        ndata = len(forces_temp)
        y=[]
        x0=self.x0 + 1e-5
        for i in range(ndata):
            sol = opt.root(compare,x0,args=(forces_temp[i],params)).x
            x0 = sol.copy()
            y.append(sol)
        return np.array(y).reshape(np.shape(forces))

    @property
    def parameters(self):
        theta = self.param_default.copy()
        mat_theta = self.mat_model.parameters
        if len(theta.keys() & mat_theta.keys())>0:
                raise ValueError("Same parameter names in the model and the sample were used. You must modify the parameter names in the classes to avoid conflicts")
        theta.update(mat_theta)
        return theta

    @parameters.setter
    def parameters(self,theta):
        raise ValueError("The dictionary of parameters should not be changed in this way")

class UniaxialExtension(SampleExperiment):
    def __init__(self,mat_model,disp_measure='stretch',force_measure='force'):
        super().__init__(mat_model,disp_measure,force_measure)
        self.param_default  = dict(L0=1.,A0=1.)
        self.param_low_bd   = dict(L0=0.0001,A0=0.0001)
        self.param_up_bd    = dict(L0=1000.,A0=1000.)
        self.update(**self.param_default)
        #check the fibers in mat_model and set their directions to [1,0,0]
        for mm in mat_model.models:
            F = mm.fiber_dirs
            if F is None:
                continue
            for f in F:
                if f[1]!=0 or f[2]!= 0:
                    warnings.warn("The UniaxialExtension assumes that fibers are aligned along the first direction. This is not satisfied and the results may be spurious.")
        if self.output == 'force':
            self.compute = partial(self.mat_model.stress,stresstype='1pk',incomp=True,Fdiag=True)
        else:
            self.compute = partial(self.mat_model.stress,stresstype=force_measure,incomp=True,Fdiag=True)
        if self.inp == 'stretch':
            self.x0 = 1.
        elif self.inp == 'strain':
            self.x0 = 0.
        elif self.inp == 'deltall':
            self.x0 = 0.
        elif self.inp == 'length':
            self.x0 = self.L0
        self.ndim=1

    def update(self,L0,A0,**extra_args):
        self.L0 = L0
        self.A0 = A0

    def F(self,input_):
        #converts the input into 3X3 F tensors
        F = []
        for i in input_:
            i = self.stretch(i)
            F.append(np.diag([i,1./sqrt(i),1./sqrt(i)]))
        return F

    def stretch(self,l):
        if type(l) is np.ndarray or isinstance(l,list):
            if len(l)>1:
                raise ValueError("The length of stretch vector should be one")
            l = l[0]

        #converts the input into stretch
        if self.inp == 'stretch':
            return l
        if self.inp == 'strain':
            return sqrt(l)+1
        if self.inp == 'deltal':
            return l/self.L0+1
        if self.inp == 'length':
            return l/self.L0

    def observe(self,stress):
        #converts the output into force
        s1 = stress[0,0]
        if self.output=='force':
            return s1*self.A0
        return s1

class PlanarBiaxialExtension(SampleExperiment):
    def __init__(self,mat_model,disp_measure='stretch',force_measure='cauchy'):
        super().__init__(mat_model,disp_measure,force_measure)
        self.param_default  = dict(L10=1.,L20=1.,thick=1.)
        self.param_low_bd   = dict(L10=0.0001,L20=0.0001,thick=0.0001)
        self.param_up_bd    = dict(L10=1000.,L20=1000.,thick=1000.)
        self.update(**self.param_default)
        #check the fibers in mat_model 
        for mm in mat_model.models:
            F = mm.fiber_dirs
            if F is None:
                continue
            for f in F:
                if f[2]!= 0:
                    warnings.warn("The PlanarBiaxialExtension assumes that fibers are in the plane. This is not satisfied and the results may be spurious.")

        if self.output == 'force' or self.output == 'tension':
            self.compute = partial(self.mat_model.stress,stresstype='1pk',incomp=True,Fdiag=True)
        else:
            self.compute = partial(self.mat_model.stress,stresstype=force_measure,incomp=True,Fdiag=True)
        if self.inp == 'stretch':
            self.x0 = np.array([1.,1.])
        elif self.inp == 'strain':
            self.x0 = np.zeros(2)
        elif self.inp == 'deltal':
            self.x0 = np.zeros(2)
        elif self.inp == 'length':
            self.x0 = self.L0.copy()
        self.ndim=2

    def update(self,L10,L20,thick,**extra_args):
        self.L0 = np.array([L10,L20])
        self.thick = thick

    def F(self,input_):
        if type(input_).__module__ == np.__name__:
            input_ = input_.reshape(-1,2)
        F = []
        for i in input_:
            i = self.stretch(i)
            F.append(np.diag([i[0],i[1],1./i[0]/i[1]]))
        return F

    def stretch(self,l):
        if len(l) != 2:
            raise ValueError("Inputs to PlanarBiaxialExtension must have two components")
        if self.inp == 'stretch':
            return l
        if self.inp == 'strain':
            return np.sqrt(l)+1
        if self.inp == 'deltal':
            return l/self.L0+1
        if self.inp == 'length':
            return l/self.L0

    def observe(self,stress):
        s1,s2 = stress[0,0],stress[1,1]
        if self.output=='force':
            return np.array([s1*self.L0[1],s2*self.L0[0]])*self.thick
        if self.output=='tension':
            return np.array([s1,s2])*self.thick
        return np.array([s1,s2])

class UniformAxisymmetricTubeInflationExtension(SampleExperiment):
    def __init__(self,mat_model,disp_measure='radius',force_measure='force'):
        super().__init__(mat_model,disp_measure,force_measure)
        self.param_default  = dict(Ri=1., thick=1., omega=0., L0=1.,lambdaZ=1.)
        self.param_low_bd   = dict(Ri=1., thick=1., omega=0., L0=1.,lambdaZ=1.)
        self.param_up_bd    = dict(Ri=1., thick=1., omega=0., L0=1.,lambdaZ=1.)
        self.update(**self.param_default)
        #check the fibers in mat_model
        for mm in mat_model.models:
            F = mm.fiber_dirs
            if F is None:
                continue
            if len(F)%2 !=0:
                warnings.warn("Even number of fiber families are expected. The results may be spurious")
            for f in F:
                if f[0]!=0:
                    warnings.warn("The UniformAxisymmetricTubeInflationExtension assumes that fibers are aligned in a helical direction. This is not satisfied and the results may be spurious.")
            for f1, f2 in zip(*[iter(F)]*2):
                if (f1+f2)[1] != 0. and (f1+f2)[2] != 0.:
                    warnings.warn("The UniformAxisymmetricTubeInflationExtension assumes that fibers are symmetric. This is not satisfied and the results may be spurious.")
                    print(f1,f2)
        self.compute = partial(self.mat_model.stress,stresstype='cauchy',incomp=True,Fdiag=True)
        if self.inp == 'stretch':
            self.x0 = 1.
        elif self.inp == 'deltalr':
            self.x0 = 0.
        elif self.inp == 'radius':
            self.x0 = self.Ri
        elif self.inp == 'area':
            self.x0 = self.Ri**2*pi
        else:
            raise ValueError("Unknown disp_measure", disp_measure)
        self.ndim=1

    def update(self,Ri,thick,omega,L0,lambdaZ,**extra_args):
        self.Ri,self.thick,self.k,self.L0,self.lambdaZ = Ri,thick,2*pi/(2*pi-omega),L0,lambdaZ

    def F(self,r,R):
        return np.diag([R/r/self.k/self.lambdaZ,self.k*r/R,self.lambdaZ])

    def disp_controlled(self,input_,params):
        self.update(**params)

        def integrand(xi,ri,params):
            R = self.Ri+xi*self.thick
            r = sqrt((R**2-self.Ri**2)/self.k/self.lambdaZ+ri**2)
            F = self.F(r,R)
            sigma = self.compute(F,params) 
            return R/self.lambdaZ/r**2*self.thick*(sigma[0,0]-sigma[1,1])
        
        if self.output=='pressure':
            output = [quad(integrand,0,1,args=(ri,params))[0] for ri in self.stretch(input_)]
        elif self.output =='force':
            output = [quad(integrand,0,1,args=(ri,params))[0]*self.L0*self.lambdaZ*pi*ri**2 for ri in self.stretch(input_)]
        return np.array(output).reshape(np.shape(input_))

    def stretch(self,l): #this returns internal radius instead
        if self.inp == 'stretch':
            return l*self.Ri
        if self.inp == 'strain':
            return np.sqrt(l)+1
        if self.inp == 'deltar':
            return l+self.Ri
        if self.inp == 'radius':
            return l
        if self.inp == 'area':
            return np.sqrt(l/pi)

