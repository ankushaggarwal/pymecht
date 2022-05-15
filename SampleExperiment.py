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

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        self.update(**params)
        output = [self.observe(self.compute(F,params)) for F in self.F(input_)]
        return np.array(output).reshape(np.shape(input_))

    def force_controlled(self,forces,params=None,x0=None):
        if params is None:
            params = self.parameters
        self.update(**params)
        
        def compare(displ,ybar,params):
            return self.disp_controlled([displ],params)[0]-ybar

        #solve for the input_ by solving the disp_controlled minus desired output
        forces_temp = forces.reshape(-1,self.ndim)
        ndata = len(forces_temp)
        y=[]
        if x0 is None:
            x0=self.x0 + 1e-5
        for i in range(ndata):
            sol = opt.root(compare,x0,args=(forces_temp[i],params))
            if not sol.success or any(np.abs(sol.r)>1e5):
                if ndata==1:
                    raise RuntimeError('force_controlled: Solution not converged',forces_temp[i],params)
                NIter=[5,10,20]
                for niter in NIter:
                    df = (forces_temp[i]-forces_temp[i-1])/niter
                    x0j = x0.copy()
                    for j in range(niter):
                        sol = opt.root(compare,x0j,args=(forces_temp[i-1]+(j+1)*df,params))
                        #print('subiter',j,'/',niter,forces_temp[i-1]+(j+1)*df,params,x0,sol)
                        if not sol.success or any(np.abs(sol.r)>1e5):
                            break
                        x0j = sol.x.copy()
                    if sol.success:
                        break
            if not sol.success:
                raise RuntimeError('force_controlled: Solution not converged',forces_temp[i],params)
            x0 = sol.x.copy()
            y.append(sol.x)
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

    def parameters_wbounds(self):
        theta = self.param_default.copy()
        theta_low = self.param_low_bd.copy()
        theta_up = self.param_up_bd.copy()
        mat_theta,mat_theta_low,mat_theta_up = self.mat_model.parameters_wbounds()
        if len(theta.keys() & mat_theta.keys())>0:
                raise ValueError("Same parameter names in the model and the sample were used. You must modify the parameter names in the classes to avoid conflicts")

        theta.update(mat_theta)
        theta_low.update(mat_theta_low)
        theta_up.update(mat_theta_up)

        return theta, theta_low, theta_up

class LinearSpring(SampleExperiment):
    def __init__(self,mat_model,disp_measure='stretch',force_measure='force'):
        super().__init__(mat_model,disp_measure,force_measure)
        self.param_default = dict(L0=1.,f0=0.,k0=1.,A0=1.,thick=0.)
        self.param_low_bd  = dict(L0=0.0001,f0=-100., k0=0.0001,A0=1.,thick=0.)
        self.param_up_bd   = dict(L0=1000., f0= 100., k0=1000.,A0=1.,thick=0.)
        self.update(**self.param_default)
        if self.inp == 'stretch':
            self.x0 = 1.
            self.compute1 = lambda x: self.k0*(x-1)*self.L0-self.f0
        elif self.inp == 'deltal':
            self.x0 = 0.
            self.compute1 = lambda x: self.k0*x-self.f0
        elif self.inp == 'length' or self.inp == 'radius':
            self.x0 = self.L0
            self.compute1 = lambda x: self.k0*(x-self.L0)-self.f0
        self.ndim = 1
        self.F = lambda x: x

    def update(self,L0,f0,k0,A0,**extra_args):
        self.L0 = L0
        self.f0 = f0
        self.k0 = k0
        self.A0 = A0
    
    def compute(self,x,params):
        self.update(**params)
        return self.compute1(x)

    def observe(self,force):
        if self.output=='pressure' or self.output=='stress':
            return force/self.A0
        return force

    def outer_radius(self,x,params):
        self.update(**params)
        if self.inp == 'stretch':
            return x*self.L0
        elif self.inp == 'deltal':
            return x+self.L0
        elif self.inp == 'length' or self.inp == 'radius':
            return x

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
        elif self.inp == 'deltal':
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
        self.param_default  = dict(Ri=1., thick=0.1, omega=0., L0=1.,lambdaZ=1.)
        self.param_low_bd   = dict(Ri=0.5, thick=0., omega=0., L0=1.,lambdaZ=1.)
        self.param_up_bd    = dict(Ri=1.5, thick=1., omega=0., L0=1.,lambdaZ=1.)
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
        self.compute = partial(self.mat_model.stress,stresstype='cauchy',incomp=False,Fdiag=True)
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

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        self.update(**params)

        def integrand(xi,ri,params):
            R = self.Ri+xi*self.thick
            r = sqrt((R**2-self.Ri**2)/self.k/self.lambdaZ+ri**2)
            F = self.F(r,R)
            sigma = self.compute(F,params) 
            return R/self.lambdaZ/r**2*self.thick*(sigma[1,1]-sigma[0,0])
        
        if self.output=='pressure':
            output = [quad(integrand,0,1,args=(ri,params))[0] for ri in self.stretch(input_)]
        elif self.output =='force':
            output = [quad(integrand,0,1,args=(ri,params))[0]*self.L0*self.lambdaZ*pi*ri**2 for ri in self.stretch(input_)]
        return np.array(output).reshape(np.shape(input_))

    def outer_radius(self,input_,params):
        self.update(**params)

        Ro = self.Ri+self.thick
        ro = np.array([sqrt((Ro**2-self.Ri**2)/self.k/self.lambdaZ+ri**2) for ri in self.stretch(input_)])
        return ro.reshape(np.shape(input_))

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

    def cauchy_stress(self,input_,params,n=10,pressure=None):
        self.update(**params)
        ri = self.stretch(input_)

        if type(ri) is np.ndarray or isinstance(ri,list):
            if len(ri)>1:
                raise Warning("cauchy_stress uses only single input. Only the first value will be used")
            ri = ri[0]

        def integrand(xi,ri,params):
            R = self.Ri+xi*self.thick
            r = sqrt((R**2-self.Ri**2)/self.k/self.lambdaZ+ri**2)
            F = self.F(r,R)
            sigma = self.compute(F,params) 
            return R/self.lambdaZ/r**2*self.thick*(sigma[1,1]-sigma[0,0])

        Stresses = []
        if pressure is None:
            pressure = quad(integrand,0,1,args=(ri,params))[0]
        xi = np.linspace(0,1,n)
        for xii in xi:
            R = self.Ri+xii*self.thick
            r = sqrt((R**2-self.Ri**2)/self.k/self.lambdaZ+ri**2)
            F = self.F(r,R)
            sigmabar = self.compute(F,params)
            I = quad(integrand,0,xii,args=(ri,params))[0] #=sigmarr-sigmarr0=sigmarr+pressure=sigmabar-pi
            pi = sigmabar[0,0] + pressure - I
            Stresses += [sigmabar-pi*np.eye(3)]

        return xi,Stresses

class NonUniformTube(SampleExperiment):
    #This need not be derived from SampleExperiment, but I am keeping that for now. Will decide later.
    def __init__(self,mat_model,disp_measure='radius',force_measure='pressure'):
        super().__init__(mat_model,disp_measure,force_measure)
        self.param_default  = dict(Ri=1., Rip = 0., thick=0.1, thickp = 0., omega=0., L0=1.,lambdaZ=1.)
        self.param_low_bd   = dict(Ri=0.5, Rip = 0., thick=0., thickp = 0., omega=0., L0=1.,lambdaZ=1.)
        self.param_up_bd    = dict(Ri=1.5, Rip = 0., thick=1., thickp = 0., omega=0., L0=1.,lambdaZ=1.)
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
                    warnings.warn("The NonUniformTube assumes that fibers are aligned in a helical direction. This is not satisfied and the results may be spurious.")
            for f1, f2 in zip(*[iter(F)]*2):
                if (f1+f2)[1] != 0. and (f1+f2)[2] != 0.:
                    warnings.warn("The NonUniformTube assumes that fibers are symmetric. This is not satisfied and the results may be spurious.")
                    print(f1,f2)
        self.compute = partial(self.mat_model.stress,stresstype='1stPK',incomp=False,Fdiag=False)
        if self.inp == 'radius':
            self.x0 = [self.Ri,self.Rip]
        #elif self.inp == 'area':
        #    self.x0 = [self.Ri**2*pi,self.Ri*2*pi*self.Rip]
        else:
            raise ValueError("Unknown disp_measure", disp_measure)
        self.ndim=2

    def update(self,Ri,Rip,thick,thickp,omega,L0,lambdaZ,**extra_args):
        self.Ri,self.Rip,self.thick,self.thickp,self.k,self.L0,self.lambdaZ = Ri,Rip,thick,thickp,2*pi/(2*pi-omega),L0,lambdaZ

    #def F(self,r,R,rp,Rp):
    #    return 

    def calculate_terms(self,ri,rip,params,terms):
        self.update(**params)
        Ri = self.Ri
        Rip = self.Rip
        v2 = self.lambdaZ
        if type(terms) is str:
            terms = [terms]

        def integrand(xi,term):
            R = Ri+xi*self.thick
            r = sqrt((R**2-Ri**2)/self.k/v2+ri**2)
            Rp = Rip+xi*self.thickp
            rp = ri*rip/r + (R*Rp-Ri*Rip)/r/v2
            fac = 1.
            F = np.array([[R/r/self.k/v2,0, rp-Rp],[0, self.k*r/R,0],[fac*(Rp-rp),0,v2]])
            #NOTE: All terms exclude the 2*pi factor
            if term=='energy':
                return self.mat_model.energy(F,params)*R*self.thick
            P = self.compute(F,params)
            drdu1,drdv2 = ri/r, -(R*R-Ri*Ri)/(2*r*v2**2)
            if term=='du1':
                term02 = -(2*ri*rip+2*(R*Rp-Ri*Rip)/v2)/(2*r*r)*drdu1+rip/r
                dFdu1 = np.array([[-R*drdu1/(v2*r*r), 0, term02],[0,drdu1/R,0],[-term02,0,0]])
                return (P[0,0]*dFdu1[0,0]+P[1,1]*dFdu1[1,1]+P[0,2]*dFdu1[0,2]+fac*P[2,0]*dFdu1[2,0])*R*self.thick
            if term=='dv1':
                #dFdv1 = np.array([[0,0,ri/r],[0,0,0],[0,0,0]])
                return P[0,2]*ri/r*R*self.thick
            if term=='du2':
                return 0
            if term=='dv2':
                term02 = -(2*ri*rip + 2*(R*Rp-Ri*Rip)/v2)*drdv2/(2*r*r)+(-2*(R*Rp-Ri*Rip)/v2**2)/(2*r)
                dFdv2 = np.array([[-R/(v2*r*r)*drdv2-R/(v2**2*r), 0, term02], [0,drdv2/R,0],[-term02,0,1]])
                return (P[0,0]*dFdv2[0,0]+P[1,1]*dFdv2[1,1]+P[0,2]*dFdv2[0,2]+P[2,2] + fac*P[2,0]*dFdv2[2,0])*R*self.thick
            raise AttributeError("Not an acceptable value for terms")
        
        #return quad(integrand,0,1,args=(terms[0]))[0] 
        return [quad(integrand,0,1,args=(term))[0] for term in terms]

    def consistency_check(self,params):
        ri=params['Ri']*1.1
        rip=params['Rip']+0.1
        lambdaZ = params['lambdaZ']*1.1
        eps=1e-6
        params['lambdaZ']=lambdaZ
        du1 = self.calculate_terms(ri,rip,params,'du1')[0]
        du1n = (self.calculate_terms(ri+eps,rip,params,'energy')[0]-self.calculate_terms(ri-eps,rip,params,'energy')[0])/eps/2.
        #print(du1,du1n,abs(du1-du1n))
        #return
        dv1 = self.calculate_terms(ri,rip,params,'dv1')[0]
        dv1n = (self.calculate_terms(ri,rip+eps,params,'energy')[0]-self.calculate_terms(ri,rip-eps,params,'energy')[0])/eps/2.
        dv2 = self.calculate_terms(ri,rip,params,'dv2')[0]
        params['lambdaZ']=lambdaZ+eps
        a = self.calculate_terms(ri,rip,params,'energy')[0] 
        params['lambdaZ']=lambdaZ-eps
        b = self.calculate_terms(ri,rip,params,'energy')[0] 
        dv2n = (a-b)/2./eps
        print(abs(du1-du1n),abs(dv1-dv1n),abs(dv2-dv2n))

    def outer_radius(self,ri,rip,params):
        self.update(**params)
        Ro = self.Ri+self.thick
        ro = sqrt((Ro**2-self.Ri**2)/self.k/self.lambdaZ+ri**2) 
        Rp = self.Rip+self.thickp
        rp = ri*rip/ro + (Ro*Rp-self.Ri*self.Rip)/ro/self.lambdaZ
        return ro,rp

    #this is not used
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

    #TODO fix this, it was simply copied from uniformtube
    def cauchy_stress(self,input_,params,n=10,pressure=None):
        self.update(**params)
        ri = self.stretch(input_)

        if type(ri) is np.ndarray or isinstance(ri,list):
            if len(ri)>1:
                raise Warning("cauchy_stress uses only single input. Only the first value will be used")
            ri = ri[0]

        def integrand(xi,ri,params):
            R = self.Ri+xi*self.thick
            r = sqrt((R**2-self.Ri**2)/self.k/self.lambdaZ+ri**2)
            F = self.F(r,R)
            sigma = self.compute(F,params) 
            return R/self.lambdaZ/r**2*self.thick*(sigma[1,1]-sigma[0,0])

        Stresses = []
        if pressure is None:
            pressure = quad(integrand,0,1,args=(ri,params))[0]
        xi = np.linspace(0,1,n)
        for xii in xi:
            R = self.Ri+xii*self.thick
            r = sqrt((R**2-self.Ri**2)/self.k/self.lambdaZ+ri**2)
            F = self.F(r,R)
            sigmabar = self.compute(F,params)
            I = quad(integrand,0,xii,args=(ri,params))[0] #=sigmarr-sigmarr0=sigmarr+pressure=sigmabar-pi
            pi = sigmabar[0,0] + pressure - I
            Stresses += [sigmabar-pi*np.eye(3)]

        return xi,Stresses

class LayeredSamples:
    '''
    A class which can contain layers of samples
    '''
    def __init__(self,*samplesList):
        self._samples = samplesList
        self.nsamples = len(samplesList)
        #check that all the members are instance of SampleExperiment
        if not all([isinstance(s,SampleExperiment) for s in self._samples]):
            raise ValueError("The class only accepts objects of type SampleExperiment")
        outputs = [s.output for s in self._samples]
        inputs = [s.inp for s in self._samples]
        self.ndim = samplesList[0].ndim
        #check if all outputs and inputs are the same
        if len(set(outputs)) > 1:
            raise ValueError("The outputs for all the layers must be the same")
        if len(set(inputs)) > 1:
            raise ValueError("The inputs for all the layers must be the same")

    @property
    def parameters(self):
        return [s.parameters for s in self._samples]

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        if len(params) != self.nsamples:
            raise ValueError("The params argument is of different length than the number of layers. This is not allowed")
        total_force = 0.
        for i,s in enumerate(self._samples):
            total_force += s.disp_controlled(input_,params[i])

        return total_force

    def force_controlled(self,forces,params,x0=None):
        
        def compare(displ,ybar,params):
            return self.disp_controlled([displ],params)[0]-ybar

        #solve for the input_ by solving the disp_controlled minus desired output
        forces_temp = forces.reshape(-1,self.ndim)
        ndata = len(forces_temp)
        y=[]
        if x0 is None:
            x0=self._samples[0].x0 + 1e-5
        for i in range(ndata):
            sol = opt.root(compare,x0,args=(forces_temp[i],params))
            if not sol.success:
                raise RuntimeError('force_controlled: Solution not converged',forces_temp[i],params)
            x0 = sol.x.copy()
            y.append(sol.x)
        return np.array(y).reshape(np.shape(forces))

class LayeredUniaxial(LayeredSamples):
    def __init__(self,*samplesList):
        super().__init__(*samplesList)
        if not all([isinstance(s,UniaxialExtension) for s in self._samples]):
            raise ValueError("The class only accepts objects of type SampleExperiment")

class LayeredTube(LayeredSamples):
    def __init__(self,*samplesList):
        super().__init__(*samplesList)
        if not all([isinstance(s,UniformAxisymmetricTubeInflationExtension) or isinstance(s,LinearSpring) for s in self._samples]):
            raise ValueError("The class only accepts objects of type SampleExperiment")
        for i,s in enumerate(samplesList):
            if i==0:
                continue
            s.inp = 'radius' #except the first layer make other layers' input in terms of radius

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        if len(params) != self.nsamples:
            raise ValueError("The params argument is of different length than the number of layers. This is not allowed")
        total_force = 0.
        i_input = input_
        for i,s in enumerate(self._samples):
            total_force += s.disp_controlled(i_input,params[i])
            i_input = s.outer_radius(i_input,params[i])

        return total_force

    def cauchy_stress(self,input_,params=None,n=10):
        if params is None:
            params = self.parameters
        #temporarily change the output to pressure and calculate the pressure related to the input, which will be used for stress calculation
        temp = self._samples[0].output
        for s in self._samples:
            s.output = 'pressure'
        pressure = self.disp_controlled(input_,params)[0]

        total_thick = 0.
        for s in params: total_thick+=s['thick']

        XI, Stress = [],[]
        i_input = input_
        first_layer=True
        for i,s in enumerate(self._samples):
            if isinstance(s,LinearSpring):
                continue
            xi,stress = s.cauchy_stress(i_input,params[i],n,pressure=pressure)
            pressure -= s.disp_controlled(i_input,params[i])[0]
            if not first_layer:
                xi = [max(XI)+x*s.thick/total_thick for x in xi]
                first_layer=False
            else:
                xi = [x*s.thick/total_thick for x in xi]
            i_input = s.outer_radius(i_input,params[i])
            print(i_input)
            XI.extend(xi)
            Stress.extend(stress)
        for s in self._samples:
            s.output = temp
        return XI,Stress

class LayeredNonUniformTube(LayeredSamples):
    def __init__(self,*samplesList):
        super().__init__(*samplesList)
        if not all([isinstance(s,NonUniformTube) for s in self._samples]):
            raise ValueError("The class only accepts objects of type SampleExperiment")
        for i,s in enumerate(samplesList):
            if i==0:
                continue
            s.inp = 'radius' #except the first layer make other layers' input in terms of radius

    def calculate_terms(self,ri,rip,params,terms):
        if len(params) != self.nsamples:
            raise ValueError("The params argument is of different length than the number of layers. This is not allowed")
        if type(terms) is str:
            terms = [terms]
        results = np.zeros(len(terms))
        rii,ripi = ri,rip
        for i,s in enumerate(self._samples):
            results += s.calculate_terms(rii,ripi,params[i],terms)
            rii,ripi = s.outer_radius(rii,ripi,params[i])

        return results

    #TODO fix this, it was simply copied from uniformtube
    def cauchy_stress(self,input_,params,n=10):
        #temporarily change the output to pressure and calculate the pressure related to the input, which will be used for stress calculation
        temp = self._samples[0].output
        for s in self._samples:
            s.output = 'pressure'
        pressure = self.disp_controlled(input_,params)[0]

        total_thick = 0.
        for s in params: total_thick+=s['thick']

        XI, Stress = [],[]
        i_input = input_
        for i,s in enumerate(self._samples):
            xi,stress = s.cauchy_stress(i_input,params[i],n,pressure=pressure)
            pressure -= s.disp_controlled(i_input,params[i])[0]
            if i>0:
                xi = [max(XI)+x*s.thick/total_thick for x in xi]
            else:
                xi = [x*s.thick/total_thick for x in xi]
            i_input = s.outer_radius(i_input,params[i])
            print(i_input)
            XI.extend(xi)
            Stress.extend(stress)
        for s in self._samples:
            s.output = temp
        return XI,Stress
