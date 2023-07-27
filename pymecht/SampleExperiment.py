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
        self._mat_model = mat_model
        self._inp = disp_measure.replace(" ","").lower()
        self._output = force_measure.replace(" ","").lower()

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        self._update(**params)
        return_scalar, return_list = False, False
        if type(input_) is list:
            return_list = True
        elif type(input_) is float or type(input_) is int:
            input_ = [input_]
            return_scalar = True    
        elif type(input_) is not np.ndarray:
            raise ValueError("Input to disp_controlled should be a scalar, a list, or a numpy array")
        output = [self._observe(self._compute(F,params)) for F in self._defGrad(input_)]
        if return_scalar:
            return output[0]
        if return_list:
            return output
        return np.array(output).reshape(np.shape(input_))

    def force_controlled(self,forces,params=None,x0=None):
        if params is None:
            params = self.parameters
        self._update(**params)
        return_scalar, return_list = False, False
        if type(forces) is float or type(forces) is int:
            forces = np.array([forces])
            return_scalar = True
        elif type(forces) is list:
            forces = np.array(forces)
            return_list = True
        elif type(forces) is not np.ndarray:
            raise ValueError("Input to force_controlled should be a scalar, a list, or a numpy array")
        
        def compare(displ,ybar,params):
            return self.disp_controlled([displ],params)[0]-ybar

        #solve for the input_ by solving the disp_controlled minus desired output
        forces_temp = forces.reshape(-1,self._ndim)
        ndata = len(forces_temp)
        y=[]
        if x0 is None:
            x0=self._x0 + 1e-5
        for i in range(ndata):
            sol = opt.root(compare,x0,args=(forces_temp[i],params))
            if not sol.success or any(np.abs(sol.r)>1e5):
                if ndata==1 or i==0:
                    niter=10
                    for j in range(1,niter+1):
                        df = forces_temp[i]*j/niter
                        #print(df,x0)
                        sol = opt.root(compare,x0,args=(df,params))
                        if not sol.success or any(np.abs(sol.r)>1e5):
                            break
                        x0 = sol.x.copy()
                    #raise RuntimeError('force_controlled: Solution not converged',forces_temp[i],params)
                else:
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
        if return_scalar:
            return y[0]
        if return_list:
            return y
        return np.array(y).reshape(np.shape(forces))

    @property
    def parameters(self):
        theta = self._param_default.copy()
        mat_theta = self._mat_model.parameters
        if len(theta.keys() & mat_theta.keys())>0:
                raise ValueError("Same parameter names in the model and the sample were used. You must modify the parameter names in the classes to avoid conflicts")
        theta.update(mat_theta)
        return theta

    @parameters.setter
    def parameters(self,theta):
        mat = {}
        for k in theta.keys():
            if k in self._param_default:
                self._param_default[k] = theta[k]
            else:
                mat[k] = theta[k]
        self._update(**self._param_default)
        self._mat_model.parameters = mat

    def parameters_wbounds(self):
        theta = self._param_default.copy()
        theta_low = self._param_low_bd.copy()
        theta_up = self._param_up_bd.copy()
        mat_theta,mat_theta_low,mat_theta_up = self._mat_model.parameters_wbounds()
        if len(theta.keys() & mat_theta.keys())>0:
                raise ValueError("Same parameter names in the model and the sample were used. You must modify the parameter names in the classes to avoid conflicts")

        theta.update(mat_theta)
        theta_low.update(mat_theta_low)
        theta_up.update(mat_theta_up)

        return theta, theta_low, theta_up
    
    def __str__(self):
        out = "An object of type " + self.__class__.__name__ + "with " + self._inp + " as input, " + self._output + " as output, and the following material\n"
        out += self._mat_model.__str__()
        return out
    
    def __repr__(self):
        return self.__str__()

class LinearSpring(SampleExperiment):
    def __init__(self,mat_model,disp_measure='stretch',force_measure='force'):
        super().__init__(mat_model,disp_measure,force_measure)
        self._param_default = dict(L0=1.,f0=0.,k0=1.,A0=1.,thick=0.)
        self._param_low_bd  = dict(L0=0.0001,f0=-100., k0=0.0001,A0=1.,thick=0.)
        self._param_up_bd   = dict(L0=1000., f0= 100., k0=1000.,A0=1.,thick=0.)
        self._update(**self._param_default)
        if self._inp == 'stretch':
            self._x0 = 1.
            self._compute1 = lambda x: self._k0*(x-1)*self._L0-self._f0
        elif self._inp == 'deltal':
            self._x0 = 0.
            self._compute1 = lambda x: self._k0*x-self._f0
        elif self._inp == 'length' or self._inp == 'radius':
            self._x0 = self._L0
            self._compute1 = lambda x: self._k0*(x-self._L0)-self._f0
        else:
            raise ValueError(self.__class__.__name__,": Unknown disp_measure", disp_measure,". It should be one of stretch, deltal, length, or radius")
        if not self._output in ['force','stress','pressure']:
            raise ValueError(self.__class__.__name__,": Unknown force_measure", force_measure,". It should be one of force, stress, or pressure")
        self._ndim = 1
        self._defGrad = lambda x: x

    def _update(self,L0,f0,k0,A0,**extra_args):
        self._L0 = L0
        self._f0 = f0
        self._k0 = k0
        self._A0 = A0
        if self._inp == 'length' or self._inp == 'radius':
            self._x0 = self._L0
    
    def _compute(self,x,params):
        self._update(**params)
        return self._compute1(x)

    def _observe(self,force):
        if self._output=='pressure' or self._output=='stress':
            return force/self._A0
        return force

    def outer_radius(self,x,params):
        self._update(**params)
        if self._inp == 'stretch':
            return x*self._L0
        elif self._inp == 'deltal':
            return x+self._L0
        elif self._inp == 'length' or self._inp == 'radius':
            return x

class UniaxialExtension(SampleExperiment):
    def __init__(self,mat_model,disp_measure='stretch',force_measure='force'):
        super().__init__(mat_model,disp_measure,force_measure)
        self._param_default  = dict(L0=1.,A0=1.)
        self._param_low_bd   = dict(L0=0.0001,A0=0.0001)
        self._param_up_bd    = dict(L0=1000.,A0=1000.)
        self._update(**self._param_default)
        #check the fibers in mat_model and set their directions to [1,0,0]
        for mm in mat_model.models:
            F = mm.fiber_dirs
            if F is None:
                continue
            for f in F:
                if f[1]!=0 or f[2]!= 0:
                    warnings.warn("The UniaxialExtension assumes that fibers are aligned along the first direction. This is not satisfied and the results may be spurious.")

        if not self._output in ['force']+[item for sublist in mat_model._stressnames for item in sublist]:
            raise ValueError(self.__class__.__name__,": Unknown force_measure", force_measure,". It should be either force or one of the stress measures in the material model")

        if self._output == 'force':
            self._compute = partial(self._mat_model.stress,stresstype='1pk',incomp=True,Fdiag=True)
        else:
            self._compute = partial(self._mat_model.stress,stresstype=force_measure,incomp=True,Fdiag=True)
        if self._inp == 'stretch':
            self._x0 = 1.
        elif self._inp == 'strain':
            self._x0 = 0.
        elif self._inp == 'deltal':
            self._x0 = 0.
        elif self._inp == 'length':
            self._x0 = self._L0
        else:
            raise ValueError(self.__class__.__name__,": Unknown disp_measure", disp_measure,". It should be one of stretch, strain, length, or deltal")
        self._ndim=1

    def _update(self,L0,A0,**extra_args):
        self._L0 = L0
        self._A0 = A0
        if self._inp == 'length' or self._inp == 'radius':
            self._x0 = self._L0

    def _defGrad(self,input_):
        #converts the input into 3X3 F tensors
        F = []
        for i in input_:
            i = self._stretch(i)
            F.append(np.diag([i,1./sqrt(i),1./sqrt(i)]))
        return F

    def _stretch(self,l):
        if type(l) is np.ndarray or isinstance(l,list):
            if len(l)>1:
                raise ValueError("The length of stretch vector should be one")
            l = l[0]

        #converts the input into stretch
        if self._inp == 'stretch':
            return l
        if self._inp == 'strain':
            return sqrt(l)+1
        if self._inp == 'deltal':
            return l/self._L0+1
        if self._inp == 'length':
            return l/self._L0

    def _observe(self,stress):
        #converts the output into force
        s1 = stress[0,0]
        if self._output=='force':
            return s1*self._A0
        return s1

class PlanarBiaxialExtension(SampleExperiment):
    def __init__(self,mat_model,disp_measure='stretch',force_measure='cauchy'):
        super().__init__(mat_model,disp_measure,force_measure)
        self._param_default  = dict(L10=1.,L20=1.,thick=1.)
        self._param_low_bd   = dict(L10=0.0001,L20=0.0001,thick=0.0001)
        self._param_up_bd    = dict(L10=1000.,L20=1000.,thick=1000.)
        self._update(**self._param_default)
        #check the fibers in mat_model 
        for mm in mat_model.models:
            F = mm.fiber_dirs
            if F is None:
                continue
            for f in F:
                if f[2]!= 0:
                    warnings.warn("The PlanarBiaxialExtension assumes that fibers are in the plane. This is not satisfied and the results may be spurious.")

        if not self._output in ['force','tension']+[item for sublist in mat_model._stressnames for item in sublist]:
            raise ValueError(self.__class__.__name__,": Unknown force_measure", force_measure,". It should be either force, tension, or one of the stress measures in the material model")

        if self._output == 'force' or self._output == 'tension':
            self._compute = partial(self._mat_model.stress,stresstype='1pk',incomp=True,Fdiag=True)
        else:
            self._compute = partial(self._mat_model.stress,stresstype=force_measure,incomp=True,Fdiag=True)
        if self._inp == 'stretch':
            self._x0 = np.array([1.,1.])
        elif self._inp == 'strain':
            self._x0 = np.zeros(2)
        elif self._inp == 'deltal':
            self._x0 = np.zeros(2)
        elif self._inp == 'length':
            self._x0 = self._L0.copy()
        else:
            raise ValueError(self.__class__.__name__,": Unknown disp_measure", disp_measure,". It should be one of stretch, strain, length, or deltal")
        self._ndim=2

    def _update(self,L10,L20,thick,**extra_args):
        self._L0 = np.array([L10,L20])
        self._thick = thick
        if self._inp == 'length':
            self._x0 = self._L0.copy()

    def _defGrad(self,input_):
        if type(input_) is np.ndarray:
            input_ = input_.reshape(-1,2)
        F = []
        for i in input_:
            i = self._stretch(i)
            F.append(np.diag([i[0],i[1],1./i[0]/i[1]]))
        return F

    def _stretch(self,l):
        if len(l) != 2:
            raise ValueError("Inputs to PlanarBiaxialExtension must have two components")
        if self._inp == 'stretch':
            return l
        if self._inp == 'strain':
            return np.sqrt(l)+1
        if self._inp == 'deltal':
            return l/self._L0+1
        if self._inp == 'length':
            return l/self._L0

    def _observe(self,stress):
        s1,s2 = stress[0,0],stress[1,1]
        if self._output=='force':
            return np.array([s1*self._L0[1],s2*self._L0[0]])*self._thick
        if self._output=='tension':
            return np.array([s1,s2])*self._thick
        return np.array([s1,s2])

class UniformAxisymmetricTubeInflationExtension(SampleExperiment):
    def __init__(self,mat_model,disp_measure='radius',force_measure='force'):
        super().__init__(mat_model,disp_measure,force_measure)
        self._param_default  = dict(Ri=1., thick=0.1, omega=0., L0=1.,lambdaZ=1.)
        self._param_low_bd   = dict(Ri=0.5, thick=0., omega=0., L0=1.,lambdaZ=1.)
        self._param_up_bd    = dict(Ri=1.5, thick=1., omega=0., L0=1.,lambdaZ=1.)
        self._update(**self._param_default)
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
        self._compute = partial(self._mat_model.stress,stresstype='cauchy',incomp=False,Fdiag=True)
        if self._inp == 'stretch':
            self._x0 = 1.
        elif self._inp == 'deltalr':
            self._x0 = 0.
        elif self._inp == 'radius':
            self._x0 = self._Ri
        elif self._inp == 'area':
            self._x0 = self._Ri**2*pi
        else:
            raise ValueError(self.__class__.__name__,": Unknown disp_measure", disp_measure,". It should be one of stretch, deltar, radius, or area")
        if not force_measure in ['force','pressure']:
            raise ValueError(self.__class__.__name__,": Unknown force_measure", force_measure,". It should be one of force or pressure")
        self._ndim=1

    def _update(self,Ri,thick,omega,L0,lambdaZ,**extra_args):
        self._Ri,self._thick,self._k,self._L0,self._lambdaZ = Ri,thick,2*pi/(2*pi-omega),L0,lambdaZ
        if self._inp == 'radius':
            self._x0 = self._Ri
        elif self._inp == 'area':
            self._x0 = self._Ri**2*pi

    def _defGrad(self,r,R):
        return np.diag([R/r/self._k/self._lambdaZ,self._k*r/R,self._lambdaZ])

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        self._update(**params)
        output_scalar, output_list = False, False
        if type(input_) is float or type(input_) is int:
            input_ = [input_]
            output_scalar = True
        elif type(input_) is list:
            output_list = True
        elif type(input_) is not np.ndarray:
            raise ValueError("Input to disp_controlled should be a scalar, a list, or a numpy array")

        def integrand(xi,ri,params):
            R = self._Ri+xi*self._thick
            r = sqrt((R**2-self._Ri**2)/self._k/self._lambdaZ+ri**2)
            F = self._defGrad(r,R)
            sigma = self._compute(F,params) 
            return R/self._lambdaZ/r**2*self._thick*(sigma[1,1]-sigma[0,0])
        
        if self._output=='pressure':
            output = [quad(integrand,0,1,args=(ri,params))[0] for ri in self._stretch(input_)]
        elif self._output =='force':
            output = [quad(integrand,0,1,args=(ri,params))[0]*self._L0*self._lambdaZ*pi*ri*2 for ri in self._stretch(input_)]
        if output_scalar:
            return output[0]
        if output_list:
            return output
        return np.array(output).reshape(np.shape(input_))

    def outer_radius(self,input_,params):
        self._update(**params)

        Ro = self._Ri+self._thick
        ro = np.array([sqrt((Ro**2-self._Ri**2)/self._k/self._lambdaZ+ri**2) for ri in self._stretch(input_)])
        return ro.reshape(np.shape(input_))

    def _stretch(self,l): #this returns internal radius instead
        if self._inp == 'stretch':
            return l*self._Ri
        if self._inp == 'strain':
            return np.sqrt(l)+1
        if self._inp == 'deltar':
            return l+self._Ri
        if self._inp == 'radius':
            return l
        if self._inp == 'area':
            return np.sqrt(l/pi)

    def cauchy_stress(self,input_,params,n=10,pressure=None):
        self._update(**params)
        ri = self._stretch(input_)

        if type(ri) is np.ndarray or isinstance(ri,list):
            if len(ri)>1:
                raise Warning("cauchy_stress uses only single input. Only the first value will be used")
            ri = ri[0]

        def integrand(xi,ri,params):
            R = self._Ri+xi*self._thick
            r = sqrt((R**2-self._Ri**2)/self._k/self._lambdaZ+ri**2)
            F = self._defGrad(r,R)
            sigma = self._compute(F,params) 
            return R/self._lambdaZ/r**2*self._thick*(sigma[1,1]-sigma[0,0])

        Stresses = []
        if pressure is None:
            pressure = quad(integrand,0,1,args=(ri,params))[0]
        xi = np.linspace(0,1,n)
        for xii in xi:
            R = self._Ri+xii*self._thick
            r = sqrt((R**2-self._Ri**2)/self._k/self._lambdaZ+ri**2)
            F = self._defGrad(r,R)
            sigmabar = self._compute(F,params)
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
        self._nsamples = len(samplesList)
        #check that all the members are instance of SampleExperiment
        if not all([isinstance(s,SampleExperiment) for s in self._samples]):
            raise ValueError("The class only accepts objects of type SampleExperiment")
        outputs = [s._output for s in self._samples]
        inputs = [s._inp for s in self._samples]
        self._ndim = samplesList[0]._ndim
        #check if all outputs and inputs are the same
        if len(set(outputs)) > 1:
            raise ValueError("The outputs for all the layers must be the same")
        self._inp = inputs[0]
        if len(set(inputs)) > 1:
            raise ValueError("The inputs for all the layers must be the same")
        self._output = outputs[0]

    @property
    def parameters(self):
        return [s.parameters for s in self._samples]

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        if len(params) != self._nsamples:
            raise ValueError("The params argument is of different length than the number of layers. This is not allowed")
        total_force = 0.
        for i,s in enumerate(self._samples):
            total_force += s.disp_controlled(input_,params[i]) #TODO this would not be correct for stresses

        return total_force

    def force_controlled(self,forces,params,x0=None): #TODO update this based on SampleExperiment.force_controlled
        if params is None:
            params = self.parameters 
        def compare(displ,ybar,params):
            return self.disp_controlled(displ,params)[0]-ybar

        #solve for the input_ by solving the disp_controlled minus desired output
        forces_temp = forces.reshape(-1,self._ndim)
        ndata = len(forces_temp)
        y=[]
        if x0 is None:
            x0=self._samples[0]._x0 + 1e-5
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
            s._inp = 'radius' #except the first layer make other layers' input in terms of radius

    def disp_controlled(self,input_,params=None):
        if params is None:
            params = self.parameters
        if len(params) != self._nsamples:
            raise ValueError("The params argument is of different length than the number of layers. This is not allowed")
        return_scalar, return_list = False, False
        if type(input_) is list:
            return_list = True
        elif type(input_) is float or type(input_) is int:
            return_scalar = True
            input_ = [input_]
        elif type(input_) is not np.ndarray:
            raise ValueError("Input to disp_controlled should be a scalar, a list, or a numpy array")
        input_ = np.array(input_)
        total_force = np.zeros_like(input_)
        i_input = input_
        for i,s in enumerate(self._samples):
            total_force += s.disp_controlled(i_input,params[i])
            i_input = s.outer_radius(i_input,params[i])
        if return_scalar:
            return total_force[0]
        if return_list:
            return list(total_force)
        return total_force

    def cauchy_stress(self,input_,params=None,n=10):
        if params is None:
            params = self.parameters
        #temporarily change the output to pressure and calculate the pressure related to the input, which will be used for stress calculatioself._outputfor s in self._samples:
            s._output = 'pressure'
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
            s._output = self._output
        return XI,Stress

