import numpy as np
from math import cos,sin,exp,sqrt,pi,tan,log
import scipy.optimize as opt

class MatModel:
    '''
    A material model class that allows adding different models together
    For example:
    >>> from MatModel import *
    >>> mat1 = NH() #A neo-Hookean model
    >>> mat2 = GOH([np.array([1,0,0]),np.array([0.5,0.5,0])]) #A GOH model with two fiber families
    >>> model = MatModel(mat1,mat2) #can provide as many models as needed as long as their parameters are uniquely named
    >>> model = MatModel(mat1) + MatModel(mat2) #can also add them after creating the MatModel object
    >>> model.models #provides a list of the models included
    >>> model.parameters #provides a dictionary of parameters and their default values
    >>> model = MatModel('goh','nh') #can also provide a list of model names, however one has to assign fiber directions afterwards
    >>> mm = model.models
    >>> mm[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]

    To get an overview of all the classes in this file, try the following:
    >>> import importlib, inspect
    >>> for name, cls in inspect.getmembers(importlib.import_module("MatModel"), inspect.isclass):
    >>>    print(name,cls.__doc__)
    '''
    __stressnames = [['cauchy'],['1pk','1stpk','firstpk'],['2pk','2ndpk','secondpk']]

    def __init__(self,*modelsList):
        self._models = modelsList
        #check if any of the components is a string
        if any([isinstance(m,str) for m in modelsList]):
            self._models = list(self._models) #convert to list so that it can be modified
            for i,m in enumerate(modelsList):
                if isinstance(m,str):
                    try:
                        model_ = globals()[m.upper()]
                        self._models[i] = model_()
                    except KeyError:
                        try:
                            model_ = globals()[m]
                            self._models[i] = model_()
                        except KeyError:
                            print ('Unknown model: ', m)
            self._models = tuple(self._models)
        self.theta = dict([])
        for m in self._models:
            if len(self.theta.keys() & m.param_default.keys())>0:
                raise ValueError("Common parameter names in the models used. Must modify the code to avoid conflicts")
            self.theta.update(m.param_default)

    @property
    def parameters(self):
        return self.theta.copy()

    @property
    def models(self):
        return self._models

    @parameters.setter
    def parameters(self,theta):
        raise ValueError("The dictionary of parameters should not be changed in this way")

    def parameters_wbounds(self):
        theta_low = dict([])
        theta_up = dict([])
        for m in self._models:
            theta_low.update(m.param_low_bd)
            theta_up.update(m.param_up_bd)
        if set(theta_low.keys()) != set(self.theta.keys()) or set(theta_up.keys()) != set(self.theta.keys()):
            raise ValueError("The dictionaries of parameter default, upper, and lower values have different set of keys",theta_low,theta_up,self.theta)

        return self.theta.copy(), theta_low, theta_up

    @models.setter
    def models(self,modelsList):
        raise ValueError("The component models should not be changed in this way")

    def energy(self,F=np.identity(3),theta={}):
        en = 0.
        for m in self._models:
            en += m.energy(F,theta)

        return en

    def stress(self,F=np.identity(3),theta={},stresstype='cauchy',incomp=False,Fdiag=False):
        stresstype = stresstype.replace(" ", "").lower()
        stype = None
        for i in range(len(self.__stressnames)):
            if any(stresstype==x for x in self.__stressnames[i]):
                stype = i
        if stype is None:
            raise ValueError('Unknown stress type, only the following allowed:', self.__stressnames)

        #Calculate the second PK stress
        S = np.zeros([3,3])
        detF = 1.
        for m in self._models:
            S += m.secondPK(F,theta)
            detF = m.J

        #If incompressible, then impose 2,2 component of stress=0 to find the Lagrange multiplier
        #TODO find a better way to implement this?
        if incomp:
            C = F.T@F
            if Fdiag: #if F is diagonal, then inverse of C can be computed more easily
                invC = np.diag(1./np.diag(C))
            else:
                invC = np.linalg.inv(C)
            p = S[2,2]/invC[2,2]
            S -= p*invC

        if stype==0: #return Cauchy stress
            return 1./detF*F@S@F.T #convert to Cauchy stress
        if stype==1: #return 1st PK stress
            return F@S
        if stype==2: #return 2nd PK stress
            return S

    def energy_stress(self,F=np.identity(3),theta={},stresstype='cauchy',incomp=False,Fdiag=False):
        stresstype = stresstype.replace(" ", "").lower()
        stype = None
        for i in range(len(self.__stressnames)):
            if any(stresstype==x for x in self.__stressnames[i]):
                stype = i
        if stype is None:
            raise ValueError('Unknown stress type, only the following allowed:', self.__stressnames)

        en = 0.
        #Calculate the second PK stress
        S = np.zeros([3,3])
        detF = 1.
        for m in self._models:
            e,s = m.energy_stress(F,theta)
            en += e
            S += s
            detF = m.J

        #If incompressible, then impose 2,2 component of stress=0 to find the Lagrange multiplier
        #TODO find a better way to implement this?
        if incomp:
            C = F.T@F
            if Fdiag: #if F is diagonal, then inverse of C can be computed more easily
                invC = np.diag(1./np.diag(C))
            else:
                invC = np.linalg.inv(C)
            p = S[2,2]/invC[2,2]
            S -= p*invC

        if stype==0: #return Cauchy stress
            return en, 1./detF*F@S@F.T #convert to Cauchy stress
        if stype==1: #return 1st PK stress
            return en, F@S
        if stype==2: #return 2nd PK stress
            return en, S

    def __add__(self,other):
        return MatModel(*(self.models+other.models))

class InvariantHyperelastic:
    '''
    An abstract class from which all the invariant-based hyperelastic models should be derived.
    Currently, it allows for models that depend on I1, I2, J, and I4 with multiple fiber families
    '''
    def __init__(self):
        self.I1 = 3.
        self.I2 = 3.
        self.J = 1.
        self.I4 = None
        self.M = None

    def energy(self,F,theta): #the energy density
        self.invariants(F)
        return self._energy(**theta)

    def _energy(self,**theta): #the energy density, required from the derived class
        raise NotImplementedError()

    def partial_deriv(self,**theta): #derivatives of energy w.r.t. the invariants, required from the derived class
        raise NotImplementedError()

    def invariants(self,F):
        C=np.dot(F.transpose(),F)
        self.I1 = np.trace(C)
        self.I2 = 1./2.*(self.I1**2 - np.trace(np.dot(C,C)))
        self.J = np.linalg.det(F)
        if self.M is not None:
            self.I4 = np.array([np.dot(m,np.dot(C,m)) for m in self.M])
        return

    def update(self,F):
        self.invariants(F)
        return

    def test(self,theta): #to test the partial derivatives by comparing with finite difference
        self.I1 = 3.1
        self.I4 = np.array([1.2])
        dPsidI1, dPsidI2, dPsidJ, dPsidI4 = self.partial_deriv(**theta)
        delta = 1e-5
        if dPsidI1 is not None:
            self.I1 += delta
            eplus = self._energy(**theta)
            self.I1 -= 2*delta
            eminus = self._energy(**theta)
            dPsidI1FD = (eplus-eminus)/2/delta
            print(dPsidI1,dPsidI1FD,abs(dPsidI1-dPsidI1FD))
            self.I1 += delta
        if dPsidI4 is not None:
            self.I4 += delta
            eplus = self._energy(**theta)
            self.I4 -= 2*delta
            eminus = self._energy(**theta)
            dPsidI4FD = (eplus-eminus)/2/delta
            print(dPsidI4,dPsidI4FD,abs(dPsidI4-dPsidI4FD))
            self.I4 += delta

    def energy_stress(self,F,theta): #returns both energy and second PK stress
        self.update(F)
        e = self._energy(**theta)
        I=np.eye(3)
        dPsidI1, dPsidI2, dPsidJ, dPsidI4 = self.partial_deriv(**theta) #TODO make sure that there is no problem here when parallel computing
        #J23 = self.J**(-2./3.)
        S = np.zeros([3,3])
        if dPsidI1 is not None:
            S += 2.*dPsidI1*I #contribution from I1
        if dPsidI2 is not None:
            S += 2.*dPsidI2*(self.I1*I-np.dot(F.transpose(),F)) #contribution from I2
        if dPsidJ is not None:
            S += dPsidJ*self.J*np.linalg.inv(np.dot(F.transpose(),F)) #contribution from J
        if dPsidI4 is not None:
            for i,m in enumerate(self.M):
                S += 2.*dPsidI4[i]*np.outer(m,m) #contribution from I4
        return e,S

    def secondPK(self,F,theta):
        self.update(F)
        I=np.eye(3)
        dPsidI1, dPsidI2, dPsidJ, dPsidI4 = self.partial_deriv(**theta)
        #J23 = self.J**(-2./3.)
        S = np.zeros([3,3])
        if dPsidI1 is not None:
            S += 2.*dPsidI1*I #contribution from I1
        if dPsidI2 is not None:
            S += 2.*dPsidI2*(self.I1*I-np.dot(F.transpose(),F)) #contribution from I2
        if dPsidJ is not None:
            S += dPsidJ*self.J*np.linalg.inv(np.dot(F.transpose(),F)) #contribution from J
        if dPsidI4 is not None:
            for i,m in enumerate(self.M):
                S += 2.*dPsidI4[i]*np.outer(m,m) #contribution from I4
        return S

    def normalize(self): #normalize the fiber directions to identity magnitudes
        if self.M is not None:
            for i in range(len(self.M)):
                self.M[i] = self.M[i]/np.linalg.norm(self.M[i])
        return

    @property
    def fiber_dirs(self):
        #print("Getting fiber directions")
        return self.M

    @fiber_dirs.setter
    def fiber_dirs(self,M): #need this setter in order to ensure that fiber directions are unit vectors
        if len(M)>0:
            if isinstance(M,list) and type(M[0]) is np.ndarray:
                self.M = M
            else:
                self.M = [M]
        self.normalize()
        

class NH(InvariantHyperelastic):
    '''
    Neo-Hookean model
    Psi = mu/2.*(I1-3)
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(mu=1.)
        self.param_low_bd   = dict(mu=0.0001)
        self.param_up_bd    = dict(mu=100.)

    def _energy(self,mu,**extra_args):
        return mu/2.*(self.I1-3)

    def partial_deriv(self,mu,**extra_args):
        return mu/2., None, None, None

class YEOH(InvariantHyperelastic):
    '''
    Yeoh model
    Psi = c1*(I1-3)+c2*(I1-3)**2+c3*(I1-3)**3+c4*(I1-3)**4
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(c1=1.,c2=1.,c3=1.,c4=0.)
        self.param_low_bd   = dict(c1=0.0001,c2=0.,c3=0.,c4=0.)
        self.param_up_bd    = dict(c1=100.,c2=100.,c3=100.,c4=100.)

    def _energy(self,c1,c2,c3,c4,**extra_args):
        return c1*(self.I1-3)+c2*(self.I1-3)**2+c3*(self.I1-3)**3+c4*(self.I1-3)**4

    def partial_deriv(self,c1,c2,c3,c4,**extra_args):
        return c1+2*c2*(self.I1-3)+3*c3*(self.I1-3)**2+4*c4*(self.I1-3)**3, None, None, None

class LS(InvariantHyperelastic):
    '''
    Lee--Sacks model
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,k3,k4,I1bar,I4bar = symbols('k1 k2 k3 k4 I1bar I4bar')
    Psi = k1/2/(k4*k2+(1-k4)*k3)*(k4*exp(k2*(I1bar-3)**2) + (1-k4)*exp(k3*(I4bar-1)**2)-1)
    diff(Psi,I1bar)
    diff(Psi,I4bar)
    '''
    def __init__(self,M=[]):
        super().__init__()
        self.param_default  = dict(k1=10., k2=10., k3=10, k4=0.5)
        self.param_low_bd   = dict(k1=0.1, k2=0.1, k3=0.1, k4=0.)
        self.param_up_bd    = dict(k1=100., k2=30., k3=30, k4=1.)
        if len(M)>0:
            if isinstance(M,list):
                self.M = M
                if len(M)!=1:
                    print('Warning: LS model should be used with only one fiber.', \
                        'Other situations can give unexpected behavior and', \
                        'non-zero energy at identity deformation gradient')
            else:
                self.M = [M]
        self.normalize()

    def _energy(self,k1,k2,k3,k4,**extra_args):
        esum = k1/2/(k4*k2+(1-k4)*k3)*(k4*exp(k2*(self.I1-3)**2)-1)
        if self.I4 is not None:
            for i4 in self.I4:
                esum += k1/2/(k4*k2+(1-k4)*k3)*((1-k4)*exp(k3*(i4-1)**2))
        return esum

    def partial_deriv(self,k1,k2,k3,k4,**extra_args):
        dPsidI1 = k1/(k4*k2+(1-k4)*k3)*k2*k4*(2*self.I1 - 6)*np.exp(k2*(self.I1 - 3)**2)/2.
        if self.I4 is not None:
            dPsidI4 = k1/(k4*k2+(1-k4)*k3)*k3*(2*self.I4 - 2)*(-k4 + 1)*np.exp(k3*(self.I4 - 1)**2)/2.
        return dPsidI1, None, None, dPsidI4

class MN(InvariantHyperelastic):
    '''
    MayNewman model
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,k3,I1bar,I4bar = symbols('k1 k2 k3 I1bar I4bar')
    Psi = k1/(k2+k3)*(exp(k2*(I1bar-3)**2+k3*(sqrt(I4bar)-1)**4)-1)
    diff(Psi,I1bar)
    diff(Psi,I4bar)
    '''
    def __init__(self,M=[]):
        super().__init__()
        self.param_default  = dict(k1=10.,k2=10.,k3=10.)
        self.param_low_bd   = dict(k1=0.1,k2=0.1,k3=0.1)
        self.param_up_bd    = dict(k1=100.,k2=100.,k3=100.)
        if len(M)>0:
            if isinstance(M,list):
                self.M = M
                if len(M)!=1:
                    print('Warning: LS model should be used with only one fiber.', \
                        'Other situations can give unexpected behavior and', \
                        'non-zero energy at identity deformation gradient')
            else:
                self.M = [M]
        self.normalize()

    def _energy(self,k1,k2,k3,**extra_args):
        esum = 0.
        for i4 in self.I4:
            esum += k1/(k2+k3)*(exp(k2*(self.I1-3)**2+k3*(sqrt(i4)-1)**4)-1)
        return esum

    def partial_deriv(self,k1,k2,k3,**extra_args):
        dPsidI1 = sum(k1*k2/(k2+k3)*(2.*self.I1 - 6.)*np.exp(k2*(self.I1 - 3.)**2 + k3*(np.sqrt(self.I4) - 1.)**4))
        dPsidI4 = 2*k1*k3/(k2+k3)*(np.sqrt(self.I4) - 1.)**3*np.exp(k2*(self.I1 - 3)**2 + k3*(np.sqrt(self.I4) - 1.)**4)/np.sqrt(self.I4)
        return dPsidI1, None, None, dPsidI4

class GOH(InvariantHyperelastic):
    '''
    Gasser-Ogden-Holzapfel model
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,k3,I1bar,I4bar = symbols('k1 k2 k3 I1bar I4bar')
    Psi = k1/2/k2*(exp(k2*(k3*I1bar+(1-3*k3)*I4bar-1)**2)-1)
    diff(Psi,I1bar)
    diff(Psi,I4bar)
    '''
    def __init__(self,M=[]):
        super().__init__()
        self.param_default  = dict(k1=10., k2=10., k3=0.1)
        self.param_low_bd   = dict(k1=0.1, k2=0.1, k3=0.)
        self.param_up_bd    = dict(k1=30., k2=30., k3=1./3.)
        if len(M)>0:
            if isinstance(M,list):
                self.M = M
            else:
                self.M = [M]
        self.normalize()
    
    def _energy(self,k1,k2,k3,**extra_args):
        esum = 0.
        for i4 in self.I4:
            esum += k1/2./k2*(exp(k2*(k3*self.I1+(1-3*k3)*i4-1)**2)-1)
        return esum

    def partial_deriv(self,k1,k2,k3,**extra_args):
        Q = (self.I1*k3 + self.I4*(-3*k3 + 1) - 1)
        #Q = max(Q,np.zeros_like(Q))
        Q[Q<0]=0.
        #print(type(Q),Q)
        expt = np.exp(k2*Q**2)
        dPsidI1 = sum(k1*k3*Q*expt)
        dPsidI4 = k1*(1-3*k3)*Q*expt
        return dPsidI1, None, None, dPsidI4

class Holzapfel(InvariantHyperelastic):
    '''
    Holzapfel (2005) model
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,k3,I1bar,I4bar = symbols('k1 k2 k3 I1bar I4bar')
    Psi = k1/2/k2*(exp(k2*(k3*(I1bar-3)**2+(1-k3)*(I4bar-1)**2))-1)
    diff(Psi,I1bar)
    diff(Psi,I4bar)
    '''
    def __init__(self,M=[]):
        super().__init__()
        self.param_default  = dict(k1=10., k2=10., k3=0.5)
        self.param_low_bd   = dict(k1=0.1, k2=0.1, k3=0.)
        self.param_up_bd    = dict(k1=100., k2=30., k3=1.)
        if len(M)>0:
            if isinstance(M,list):
                self.M = M
            else:
                self.M = [M]
        self.normalize()
    
    def _energy(self,k1,k2,k3,**extra_args):
        esum = 0.
        I41 = self.I4 -1
        I41[I41<0]=0.
        for i4 in I41:
            esum += k1/2./k2*(exp(k2*(k3*(self.I1-3)**2+(1-k3)*i4**2))-1)
        return esum

    def partial_deriv(self,k1,k2,k3,**extra_args):
        I41 = self.I4 -1
        I41[I41<0]=0.
        Q = ((self.I1-3)**2*k3 + I41**2*(1-k3))
        expt = np.exp(k2*Q)
        dPsidI1 = sum(k1*k3*(self.I1-3)*expt)
        dPsidI4 = k1*(1-k3)*I41*expt
        return dPsidI1, None, None, dPsidI4

class expI1(InvariantHyperelastic):
    '''
    Model with exponential of I1
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,I1bar = symbols('k1 k2 I1bar')
    Psi = k1/k2*(exp(k2*(I1bar-3))-1) 
    diff(Psi,I1bar)
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(k1=10., k2=10.)
        self.param_low_bd   = dict(k1=0.1, k2=0.1)
        self.param_up_bd    = dict(k1=100., k2=100.)

    def _energy(self,k1,k2,**extra_args):
        return k1/k2*(exp(k2*(self.I1-3))-1) 

    def partial_deriv(self,k1,k2,**extra_args):
        dPsidI1 = k1*np.exp(k2*(self.I1 - 3)) 
        return dPsidI1, None, None, None

class HGO(InvariantHyperelastic):
    '''
    HGO 2000 model's fiber part
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,k3,k4,I1bar,I4bar = symbols('k1 k2 k3 k4 I1bar I4bar')
    Psi = k3/2./k4*(exp(k4*(I4bar-1)**2)-1)
    diff(Psi,I1bar)
    diff(Psi,I4bar)
    '''
    def __init__(self,M=[]):
        super().__init__()
        self.param_default  = dict(k3=10., k4=10.)
        self.param_low_bd   = dict(k3=0.1, k4=0.1)
        self.param_up_bd    = dict(k3=100., k4=100.)
        if len(M)>0:
            if isinstance(M,list):
                self.M = M
            else:
                self.M = [M]
        self.normalize()

    def _energy(self,k3,k4,**extra_args):
        return sum(k3/2./k4*(np.exp(k4*(self.I4-1)**2)-1))

    def partial_deriv(self,k3,k4,**extra_args):
        dPsidI4 = k3*(self.I4 - 1)*np.exp(k4*(self.I4 - 1)**2)
        return None, None, None, dPsidI4

class HY(InvariantHyperelastic):
    '''
    Humphrey-Yin model's fiber part
    The partial derivative expressions are obtained using the following code
    from sympy import *
    k1,k2,k3,k4,I1bar,I4bar = symbols('k1 k2 k3 k4 I1bar I4bar')
    Psi = k3/k4*(exp(k4*(sqrt(I4bar)-1)**2)-1)
    diff(Psi,I1bar)
    diff(Psi,I4bar)
    '''
    def __init__(self,M=[]):
        super().__init__()
        self.param_default  = dict(k3=10., k4=10.)
        self.param_low_bd   = dict(k3=0.1, k4=0.1)
        self.param_up_bd    = dict(k3=100., k4=100.)
        if len(M)>0:
            if isinstance(M,list):
                self.M = M
            else:
                self.M = [M]
        self.normalize()

    def _energy(self,k3,k4,**extra_args):
        return sum(k3/k4*(np.exp(k4*(np.sqrt(self.I4)-1)**2)-1))

    def partial_deriv(self,k3,k4,**extra_args):
        dPsidI4 = k3*(np.sqrt(self.I4) - 1)*np.exp(k4*(np.sqrt(self.I4) - 1)**2)/np.sqrt(self.I4)
        return None, None, None, dPsidI4

class volPenalty(InvariantHyperelastic):
    '''
    Volumetric penality
    Psi = kappa/2.*(J-1)**2
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(kappa=1)
        self.param_low_bd   = dict(kappa=1)
        self.param_up_bd    = dict(kappa=1)
       
    def _energy(self,kappa,**extra_args):
        return kappa/2.*(self.J-1)**2

    def partial_deriv(self,kappa,**extra_args):
        return None, None, kappa*(self.J-1), None

class polyI4(InvariantHyperelastic):
    '''
    Polynomial in I4 model
    Psi = d1*(I4-1)+d2*(I4-1)**2+d3*(I4-1)**3
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(d1=1.,d2=1.,d3=1.)
        self.param_low_bd   = dict(d1=0.0001,d2=0.,d3=0.)
        self.param_up_bd    = dict(d1=100.,d2=100.,d3=100.)

    def _energy(self,d1,d2,d3,**extra_args):
        return sum(d1*(self.I4-1)+d2*(self.I4-1)**2+d3*(self.I4-1)**3)

    def partial_deriv(self,d1,d2,d3,**extra_args):
        return None, None, None, d1+2*d2*(self.I4-1)+3*d3*(self.I4-1)**2


class splineI1I4(InvariantHyperelastic):
    '''
    Spline-based model in I1 and I4
    Psi is loaded from a spline
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict()
        self.param_low_bd   = dict()
        self.param_up_bd    = dict()
        import scipy

    def set(self,W):
        if not isinstance(mean_sp,scipy.interpolate.fitpack2.RectBivariateSpline):
            raise ValueError("W must be a RectBivariateSpline")
        x,y = W.get_knots()
        self.minx,self.maxx,self.miny,self.maxy = np.min(x), np.max(x), np.min(y), np.max(y)
        self._W = W

    def _energy(self,**extra_args):
        if self.I1<self.minx or self.I1>self.maxx or self.I4<self.miny or self.I4>self.maxy:
            warnings.warn("Outside the training range; be careful interpreting the results")
        return self._W(self.I1,self.I4)

    def partial_deriv(self,**extra_args):
        if self.I1<self.minx or self.I1>self.maxx or self.I4<self.miny or self.I4>self.maxy:
            warnings.warn("Outside the training range; be careful interpreting the results")
        return self._W(self.I1,self.I4,dx=1),None,None,self._W(self.I1,self.I4,dy=1)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    for mname in ['nh','yeoh','ls','mn','expI1','goh','Holzapfel','hgo','hy','polyI4']:
        mat = MatModel(mname)
        mm = mat.models
        print(mname)
        mm[0].fiber_dirs = [np.array([1,0,0])]
        mm[0].test(mat.parameters)

