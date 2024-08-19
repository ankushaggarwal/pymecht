import numpy as np
from math import cos,sin,exp,sqrt,pi,tan,log
import scipy.optimize as opt
import scipy
import warnings
from scipy.special import erf
from .ParamDict import *

class MatModel:
    '''
    The main material model class that allows adding different models together. 
    This will usually be the starting point in using pyMechT.

    The models can be provided as a list of model objects, or as a list of strings.

    The passed strings can have lower/upper case. The following models are available:
        
            * 'NH': Neo-Hookean model
            * 'MR': Mooney-Rivlin model
            * 'YEOH': Yeoh model
            * 'LS': Lee-Sacks model
            * 'MN': May-Newman model
            * 'GOH': Gasser-Ogden-Holzapfel model
            * 'HGO': Holzapfel-Gasser-Ogden model
            * 'expI1': A model with an exponential of I1
            * 'polyI4': A model with a polynomial of I4
            * 'HY': Humphrey-Yin model
            * 'Holzapfel': Holzapfel model
            * 'volPenalty': A penalty model for volumetric change
            * 'ArrudaBoyce': Arruda-Boyce model
            * 'Gent': Gent model
            * 'splineI1': A spline model of I1
            * 'splineI1I4': A spline model of I1 and I4
            * 'StructModel': A structural model with fiber distribution

    Multiple models can be added together to create a composite model, 
    with each model having its own fiber(s) and parameters.

    Example
    -------
        >>> from MatModel import *
        >>> mat1 = NH() #A neo-Hookean model
        >>> mat2 = GOH([np.array([1,0,0]),np.array([0.5,0.5,0])]) #A GOH model with two fiber families
        >>> mat_model = MatModel(mat1,mat2) #can provide as many models as needed as long as their parameters are uniquely named
        >>> mat_model = MatModel(mat1) + MatModel(mat2) #can also add them after creating the MatModel object
        >>> mat_model.models #provides a list of the models included
        >>> mat_model.parameters #provides a dictionary of parameters and their default values
        >>> mat_model = MatModel('GOH','NH') #can also provide a list of model names, however one has to assign fiber directions afterwards
        >>> mats = mat_model.models
        >>> mats[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]

    '''
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
        self._param_names, self._params = [], ParamDict()
        for i,m in enumerate(self._models):
            t1, t2, t3 = m.param_default, m.param_low_bd, m.param_up_bd
            self._param_names.append({k+'_'+str(i):k for k in t1})
            for k in t1:
                self._params[k+'_'+str(i)] = Param(t1[k],t2[k],t3[k],False)
            #self._params.update({k+'_'+str(i):t1[k] for k in t1})
        self._stressnames = [['cauchy'],['1pk','1stpk','firstpk'],['2pk','2ndpk','secondpk']]

    @property
    def parameters(self):
        '''
        Parameters of the model
        '''
        theta = ParamDict()
        theta.update(self._params)
        return theta

    @property
    def models(self):
        '''
        List of component models
        '''
        return self._models

    @parameters.setter
    def parameters(self,theta):
        if self._params.keys() != theta.keys():
            raise ValueError("Keys of the input parameters do not match the model parameter keys")
        else:
            self._params.update(theta)

    @models.setter
    def models(self,modelsList):
        raise ValueError("The component models should not be changed in this way")

    def energy(self,F=np.identity(3),theta=None):
        '''
        Returns the energy density of the material model
        
        Parameters
        ----------
        F: np.array([3,3]) 
            the deformation gradient,   default: identity matrix (no deformation)
            
        theta: ParamDict or dict 
            the parameters of the model, if None, then the default values are used
        
        Returns
        -------
        float
            the energy density
        
        '''
        if theta is None:
            theta=self._params
        if type(theta) is ParamDict:
            theta = theta._val()
        elif type(theta[list(theta.keys())[0]]) is Param:
            raise ValueError("Something changed the parameter dictionary that converted it from custom type to regular one")
        en = 0.
        for i,m in enumerate(self._models):
            thetai = {self._param_names[i][k]:theta[k] for k in self._param_names[i]}
            en += m.energy(F,thetai)
        return en

    def stress(self,F=np.identity(3),theta=None,stresstype='cauchy',incomp=False,Fdiag=False):
        '''
        Returns the stress tensor of the material model

        Parameters
        ----------
        F: np.array([3,3])
            the deformation gradient, default: identity matrix (no deformation)
        
        theta: ParamDict or dict
            the parameters of the model, if None, then the default values are used
        
        stresstype: str
            the type of stress tensor to return with the following options (case insensitive) 

                * 'cauchy': Cauchy stress,
                * '1pk' or '1stpk' or 'firstpk': 1st Piola-Kirchoff stress
                * '2pk' or '2ndpk' or 'secondpk': 2nd Piola-Kirchoff stress
                * default: Cauchy stress
            
        incomp: bool
            if True, then the material is assumed to be incompressible, default: False
            
        Fdiag: bool
            if True, then it is assumed that F is diagonal (for faster computation), default: False
        
        Returns
        -------
        np.array([3,3])
            the stress 3X3 tensor
        
        '''
        if theta is None:
            theta=self._params
        if type(theta) is ParamDict:
            theta = theta._val()
        elif type(theta[list(theta.keys())[0]]) is Param:
            raise ValueError("Something changed the parameter dictionary that converted it from custom type to regular one")
        stresstype = stresstype.replace(" ", "").lower()
        stype = None
        for i in range(len(self._stressnames)):
            if any(stresstype==x for x in self._stressnames[i]):
                stype = i
        if stype is None:
            raise ValueError('Unknown stress type, only the following allowed:', self._stressnames)

        #Calculate the second PK stress
        S = np.zeros([3,3])
        detF = 1.
        for i,m in enumerate(self._models):
            thetai = {self._param_names[i][k]:theta[k] for k in self._param_names[i]}
            S += m.secondPK(F,thetai)
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

    def energy_stress(self,F=np.identity(3),theta=None,stresstype='cauchy',incomp=False,Fdiag=False):
        '''
        Returns the energy density and stress tensor of the material model

        Parameters
        ----------
        F: np.array([3,3])
            the deformation gradient,   default: identity matrix (no deformation)
            
        theta: ParamDict or dict
            the parameters of the model, if None, then the default values are used
           
        stresstype: str
            the type of stress tensor to return with the following options (case insensitive) 

                * 'cauchy': Cauchy stress,
                * '1pk' or '1stpk' or 'firstpk': 1st Piola-Kirchoff stress
                * '2pk' or '2ndpk' or 'secondpk': 2nd Piola-Kirchoff stress
                * default: Cauchy stress
            
        incomp: bool
            if True, then the material is assumed to be incompressible, default: False
            
        Fdiag: bool
            if True, then it is assumed that F is diagonal (for faster computation), default: False
        
        Returns
        -------
        tuple of (float,np.array([3,3]))
            first element is the energy density
            second element is the stress 3X3 tensor

        '''
        if theta is None:
            theta=self._params
        if type(theta) is ParamDict:
            theta = theta._val()
        elif type(theta[list(theta.keys())[0]]) is Param:
            raise ValueError("Something changed the parameter dictionary that converted it from custom type to regular one")
        stresstype = stresstype.replace(" ", "").lower()
        stype = None
        for i in range(len(self._stressnames)):
            if any(stresstype==x for x in self._stressnames[i]):
                stype = i
        if stype is None:
            raise ValueError('Unknown stress type, only the following allowed:', self._stressnames)

        en = 0.
        #Calculate the second PK stress
        S = np.zeros([3,3])
        detF = 1.
        for i,m in enumerate(self._models):
            thetai = {self._param_names[i][k]:theta[k] for k in self._param_names[i]}
            e,s = m.energy_stress(F,thetai)
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

    def _test(self,theta=None):
        if theta is None:
            theta=self._params
        if type(theta) is ParamDict:
            theta = theta._val()
        elif theta is not None and type(theta[list(theta.keys())[0]]) is Param:
            raise ValueError("Something changed the parameter dictionary that converted it from custom type to regular one")
        result = []
        #Test each model if it is of InvariantHyperelastic class
        for i,m in enumerate(self._models):
            if isinstance(m,InvariantHyperelastic):
                thetai = {self._param_names[i][k]:theta[k] for k in self._param_names[i]}
                result.append(m.test(thetai))
        return all(result)

    def __str__(self):
        p = "s" if len(self._models)>1 else ""
        out = "Material model with "+str(len(self._models)) +" component"+ p +":\n"
        for i in range(len(self._models)):
            out += "Component"+str(i+1)+": "
            out += self._models[i].__class__.__name__
            if self._models[i].fiber_dirs is not None:
                out += " with fiber direction(s):" + str(self._models[i].fiber_dirs) 
            out += "\n"
        return out

    def __repr__(self) -> str:
        return self.__str__()
    
class InvariantHyperelastic:
    '''
    An abstract class from which all the invariant-based hyperelastic models should be derived.
    Currently, it allows for models that depend on I1, I2, J, and I4 with multiple fiber families
    It should not be used directly, but rather derived from. Use MatModel class to use the derived classes.
    '''
    def __init__(self):
        self.I1 = 3.
        self.I2 = 3.
        self.J = 1.
        self.I4 = None
        self.I4term = False
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
        elif self.I4term:
            raise ValueError(self.__class__.__name__+" model class uses I4 but no fiber directions have been defined. Did you forget to set the fiber directions?")
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
            #print(dPsidI1,dPsidI1FD,abs(dPsidI1-dPsidI1FD))
            assert abs(dPsidI1-dPsidI1FD)<1e-6
            self.I1 += delta
        if dPsidI2 is not None:
            self.I2 += delta
            eplus = self._energy(**theta)
            self.I2 -= 2*delta
            eminus = self._energy(**theta)
            dPsidI2FD = (eplus-eminus)/2/delta
            #print(dPsidI2,dPsidI2FD,abs(dPsidI2-dPsidI2FD))
            assert abs(dPsidI2-dPsidI2FD)<1e-6
            self.I2 += delta
        if dPsidJ is not None:
            self.J += delta
            eplus = self._energy(**theta)
            self.J -= 2*delta
            eminus = self._energy(**theta)
            dPsidJFD = (eplus-eminus)/2/delta
            #print(dPsidJ,dPsidJFD,abs(dPsidJ-dPsidJFD))
            assert abs(dPsidJ-dPsidJFD)<1e-6
            self.J += delta
        if dPsidI4 is not None:
            self.I4 += delta
            eplus = self._energy(**theta)
            self.I4 -= 2*delta
            eminus = self._energy(**theta)
            dPsidI4FD = (eplus-eminus)/2/delta
            #print(dPsidI4,dPsidI4FD,abs(dPsidI4-dPsidI4FD))
            assert abs(dPsidI4-dPsidI4FD)<1e-6
            self.I4 += delta
        return True

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
        '''
        Fiber direction vector(s)
        '''
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
    
    .. math::
        \\Psi = \\frac{\\mu}{2}(I_1-3)
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

class MR(InvariantHyperelastic):
    '''
    Mooney-Rivlin model
    
    .. math::
        \\Psi = c_1(I_1-3) + c_2(I_2-3)
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(c1=1.,c2=1.)
        self.param_low_bd   = dict(c1=0.0001,c2=0.)
        self.param_up_bd    = dict(c1=100.,c2=100.)

    def _energy(self,c1, c2, **extra_args):
        return c1*(self.I1-3) + c2*(self.I2-3) 

    def partial_deriv(self,c1, c2, **extra_args):
        return c1, c2, None, None

class YEOH(InvariantHyperelastic):
    '''
    Yeoh model

    .. math::
        \\Psi = \\sum_{i=1}^4 c_i(I_1-3)^i
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

    .. math::
        \\Psi = \\frac{k_1}{2(k_4k_2+(1-k_4)k_3)}(k_4\\exp(k_2(I_1-3)^2)+(1-k_4)\\exp(k_3(I_4-1)^2)-1)
    
    Derivation
    ----------
        The partial derivative expressions are obtained using the following code
        
        >>> from sympy import *
        >>> k1,k2,k3,k4,I1bar,I4bar = symbols('k1 k2 k3 k4 I1bar I4bar')
        >>> Psi = k1/2/(k4*k2+(1-k4)*k3)*(k4*exp(k2*(I1bar-3)**2) + (1-k4)*exp(k3*(I4bar-1)**2)-1)
        >>> diff(Psi,I1bar)
        >>> diff(Psi,I4bar)
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
        self.I4term = True

    def _energy(self,k1,k2,k3,k4,**extra_args):
        esum = k1/2/(k4*k2+(1-k4)*k3)*(k4*(exp(k2*(self.I1-3)**2)-1))
        for i4 in self.I4:
            esum += k1/2/(k4*k2+(1-k4)*k3)*((1-k4)*(exp(k3*(i4-1)**2)-1))
        return esum

    def partial_deriv(self,k1,k2,k3,k4,**extra_args):
        dPsidI1 = k1/(k4*k2+(1-k4)*k3)*k2*k4*(2*self.I1 - 6)*np.exp(k2*(self.I1 - 3)**2)/2.
        dPsidI4 = k1/(k4*k2+(1-k4)*k3)*k3*(2*self.I4 - 2)*(-k4 + 1)*np.exp(k3*(self.I4 - 1)**2)/2.
        return dPsidI1, None, None, dPsidI4

class MN(InvariantHyperelastic):
    '''
    MayNewman model

    .. math::
        \\Psi = \\sum_{i=1}^N \\frac{k_1}{k_2+k_3}(\\exp(k_2(I_1-3)^2+k_3(\\sqrt{I_{4i}}-1)^4)-1)

    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
        >>> from sympy import *
        >>> k1,k2,k3,I1bar,I4bar = symbols('k1 k2 k3 I1bar I4bar')
        >>> Psi = k1/(k2+k3)*(exp(k2*(I1bar-3)**2+k3*(sqrt(I4bar)-1)**4)-1)
        >>> diff(Psi,I1bar)
        >>> diff(Psi,I4bar)
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
                    print('Warning: MN model should be used with only one fiber.', \
                        'Other situations can give unexpected behavior and', \
                        'non-zero energy at identity deformation gradient')
            else:
                self.M = [M]
        self.normalize()
        self.I4term = True

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
    Gasser-Ogden-Holzapfel model (without the Neo-Hookean term)

    .. math::
        \\Psi = \\sum_{i=1}^N \\frac{k_1}{2k_2}(\\exp(k_2(k_3I_1+(1-3k_3)I_{4i}-1)^2)-1)

    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
            >>> from sympy import *
            >>> k1,k2,k3,I1bar,I4bar = symbols('k1 k2 k3 I1bar I4bar')
            >>> Psi = k1/2/k2*(exp(k2*(k3*I1bar+(1-3*k3)*I4bar-1)**2)-1)
            >>> diff(Psi,I1bar)
            >>> diff(Psi,I4bar)
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
        self.I4term = True
    
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
    Holzapfel (2005) model without the neo-Hookean term

    .. math::
        \\Psi = \\sum_{i=1}^N \\frac{k_1}{2k_2}(\\exp(k_2(k_3(I_1-3)^2+(1-k_3)(I_{4i}-1)^2)-1)
    
    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
        >>> from sympy import *
        >>> k1,k2,k3,I1bar,I4bar = symbols('k1 k2 k3 I1bar I4bar')
        >>> Psi = k1/2/k2*(exp(k2*(k3*(I1bar-3)**2+(1-k3)*(I4bar-1)**2))-1)
        >>> diff(Psi,I1bar)
        >>> diff(Psi,I4bar)
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
        self.I4term = True
    
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

    .. math::
        \\Psi = \\frac{k_1}{k_2}(\\exp(k_2(I_1-3))-1)
    
    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
        >>> from sympy import *
        >>> k1,k2,I1bar = symbols('k1 k2 I1bar')
        >>> Psi = k1/k2*(exp(k2*(I1bar-3))-1) 
        >>> diff(Psi,I1bar)
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

    .. math::
        \\Psi = \\sum_{i=1}^N \\frac{k_3}{2k_4}(\\exp(k_4(I_{4i}-1)^2)-1)
    
    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
        >>> from sympy import *
        >>> k1,k2,k3,k4,I1bar,I4bar = symbols('k1 k2 k3 k4 I1bar I4bar')
        >>> Psi = k3/2./k4*(exp(k4*(I4bar-1)**2)-1)
        >>> diff(Psi,I1bar)
        >>> diff(Psi,I4bar)
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
        self.I4term = True

    def _energy(self,k3,k4,**extra_args):
        return sum(k3/2./k4*(np.exp(k4*(self.I4-1)**2)-1))

    def partial_deriv(self,k3,k4,**extra_args):
        dPsidI4 = k3*(self.I4 - 1)*np.exp(k4*(self.I4 - 1)**2)
        return None, None, None, dPsidI4

class HY(InvariantHyperelastic):
    '''
    Humphrey-Yin model's fiber part

    .. math::
        \\Psi = \\sum_{i=1}^N \\frac{k_3}{k_4}(\\exp(k_4(\\sqrt{I_{4i}}-1)^2)-1)
    
    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
        >>> from sympy import *
        >>> k1,k2,k3,k4,I1bar,I4bar = symbols('k1 k2 k3 k4 I1bar I4bar')
        >>> Psi = k3/k4*(exp(k4*(sqrt(I4bar)-1)**2)-1)
        >>> diff(Psi,I1bar)
        >>> diff(Psi,I4bar)
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
        self.I4term = True

    def _energy(self,k3,k4,**extra_args):
        return sum(k3/k4*(np.exp(k4*(np.sqrt(self.I4)-1)**2)-1))

    def partial_deriv(self,k3,k4,**extra_args):
        dPsidI4 = k3*(np.sqrt(self.I4) - 1)*np.exp(k4*(np.sqrt(self.I4) - 1)**2)/np.sqrt(self.I4)
        return None, None, None, dPsidI4

class volPenalty(InvariantHyperelastic):
    '''
    Volumetric penality

    .. math::
        \\Psi = \\frac{\\kappa}{2}(J-1)^2
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

    .. math::
        \\Psi = \\sum_{i=1}^3 d_i(I_{4i}-1)^i

    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(d1=0.,d2=1.,d3=1.)
        self.param_low_bd   = dict(d1=0.0001,d2=0.,d3=0.)
        self.param_up_bd    = dict(d1=100.,d2=100.,d3=100.)
        self.normalize()
        self.I4term = True

    def _energy(self,d1,d2,d3,**extra_args):
        return sum(d1*(self.I4-1)+d2*(self.I4-1)**2+d3*(self.I4-1)**3)

    def partial_deriv(self,d1,d2,d3,**extra_args):
        return None, None, None, d1+2*d2*(self.I4-1)+3*d3*(self.I4-1)**2

class ArrudaBoyce(InvariantHyperelastic):
    '''
    Arruda-Boyce model

    .. math::
        \\Psi = \\sum_{i=1}^5 C\\frac{\\alpha_i}{N^i}(I_1^i-3^i)

    where :math:`\\alpha_i` are [1,1/20,11/1050,19/7000,519/673750]
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(C=1.,Ninv=0.8)
        self.param_low_bd   = dict(C=0.0001,Ninv=0.)
        self.param_up_bd    = dict(C=100.,Ninv=1.)
        self.I4term = False
        self.alpha = np.array([1./2.,1./20.,11./1050.,19./7000.,519./673750.])
        self.subtract = np.array([3.,3.**2,3.**3,3.**4,3.**5])

    def _energy(self,C,Ninv,**extra_args):
        Iterms = np.array([self.I1,self.I1**2,self.I1**3,self.I1**4,self.I1**5])-self.subtract
        Nterms = np.array([1.,Ninv,Ninv**2,Ninv**3,Ninv**4])
        return np.sum(C*Nterms*self.alpha*Iterms)

    def partial_deriv(self,C,Ninv,**extra_args):
        Iterms = np.array([1.,self.I1,self.I1**2,self.I1**3,self.I1**4])
        Nterms = np.array([1.,2*Ninv,3.*Ninv**2,4.*Ninv**3,5.*Ninv**4])
        return np.sum(C*Nterms*self.alpha*Iterms), None, None, None

class Gent(InvariantHyperelastic):
    '''
    Gent model

    .. math::
        \\Psi = -\\frac{\\mu J_m}{2}\\ln\\left(1-\\frac{I_1-3}{J_m}\\right)

    Derivation
    ----------
    The partial derivative expressions are obtained using the following code
        >>> from sympy import *
        >>> I1, mu, Jm = symbols('I1,mu,Jm')
        >>> Psi = -mu*Jm/2.*ln(1-(I1-3)/Jm)
        >>> diff(Psi,I1)
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(mu=1.,Jm=5)
        self.param_low_bd   = dict(mu=0.0001,Jm=0.)
        self.param_up_bd    = dict(mu=1000.,Jm=1000.)
        self.I4term = False

    def _energy(self,mu,Jm,**extra_args):
        return  -0.5*Jm*mu*np.log(1 - (self.I1 - 3)/Jm) 

    def partial_deriv(self,mu,Jm,**extra_args):
        return 0.5*mu/(1 - (self.I1 - 3)/Jm), None, None, None

class splineI1(InvariantHyperelastic):
    '''
    Spline-based model in I1 for data driven models (without an analytical expression)
    
    Psi is loaded from a scipy spline object
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(alpha=1)
        self.param_low_bd   = dict(alpha=-10)
        self.param_up_bd    = dict(alpha=10)
        self._warn = False
        self.normalize()

    def set(self,W,alpha=1):
        '''
        Set the spline function and the weight
        
        Parameters
        ----------

        W : scipy.interpolate._bsplines.BSpline
            The spline function
        
        alpha : float
            The weight of W in the energy function (i.e., :math:`\\Psi = \\alpha W(I1)`)
        
        Returns
        -------
        None

        '''
        if not isinstance(W,scipy.interpolate._bsplines.BSpline):
            raise ValueError("W must be a RectBivariateSpline")
        x = W.t
        self.minx,self.maxx = np.min(x), np.max(x)
        self._W = W
        self._dWdI1 = W.derivative()
        self._alpha = alpha

    def _energy(self,alpha=None,**extra_args):
        if alpha is None:
            alpha=self._alpha
        if self._warn and (self.I1<self.minx or self.I1>self.maxx):
            w = "Outside the training range; be careful interpreting the results "+str(self.I1)+"\n"+str(self.minx)+" "+str(self.maxx)
            warnings.warn(w)
        return alpha*np.sum(self._W(self.I1))

    def partial_deriv(self,alpha=None,**extra_args):
        if alpha is None:
            alpha=self._alpha
        if self._warn and (self.I1<self.minx or self.I1>self.maxx):
            w = "Outside the training range; be careful interpreting the results "+str(self.I1)+"\n"+str(self.minx)+" "+str(self.maxx)
            warnings.warn(w)
        a = self._dWdI1(self.I1)
        return alpha*np.sum(a),None,None,None

class splineI1I4(InvariantHyperelastic):
    '''
    Spline-based model in I1 and I4 for data driven models (without an analytical expression)
    
    Psi is loaded from a scipy spline object
    '''
    def __init__(self):
        super().__init__()
        self.param_default  = dict(alpha=1)
        self.param_low_bd   = dict(alpha=-10)
        self.param_up_bd    = dict(alpha=10)
        self._warn = False
        self.normalize()
        self.I4term = True

    def set(self,W,alpha=1):
        '''
        Set the spline function and the weight
        
        Parameters
        ----------

        W : scipy.interpolate.fitpack2.RectBivariateSpline
            The spline function
        
        alpha : float
            The weight of W in the energy function (i.e., :math:`\\Psi = \\alpha W(I1,I4)`)
        
        Returns
        -------
        None

        Notes
        -----
        The spline function should be defined on a rectangular grid
        '''
        if not isinstance(W,scipy.interpolate.fitpack2.RectBivariateSpline):
            raise ValueError("W must be a RectBivariateSpline")
        x,y = W.get_knots()
        self.minx,self.maxx,self.miny,self.maxy = np.min(x), np.max(x), np.min(y), np.max(y)
        self._W = W
        self._alpha = alpha

    def _energy(self,alpha=None,**extra_args):
        if alpha is None:
            alpha=self._alpha
        if self._warn and (self.I1<self.minx or self.I1>self.maxx or np.any(self.I4<self.miny) or np.any(self.I4>self.maxy)):
            w = "Outside the training range; be careful interpreting the results "+str(self.I1)+" "+str(self.I4)+"\n"+str(self.minx)+" "+str(self.maxx)+" "+str(self.miny)+" "+str(self.maxy)
            warnings.warn(w)
        return alpha*np.sum(self._W(self.I1,self.I4))

    def partial_deriv(self,alpha=None,**extra_args):
        if alpha is None:
            alpha=self._alpha
        if self._warn and (self.I1<self.minx or self.I1>self.maxx or np.any(self.I4<self.miny) or np.any(self.I4>self.maxy)):
            w = "Outside the training range; be careful interpreting the results "+str(self.I1)+" "+str(self.I4)+"\n"+str(self.minx)+" "+str(self.maxx)+" "+str(self.miny)+" "+str(self.maxy)
            warnings.warn(w)
        a,b = self._W(self.I1,self.I4,dx=1), self._W(self.I1,self.I4,dy=1)
        return alpha*np.sum(a),None,None,alpha*b.flatten()

class StructModel:
    '''
    A class for simplified structural model with integration over fiber directions

    .. math::
        \\Psi = \\int_{-\\pi/2}^{\\pi/2} \\Gamma(\\theta) \\psi(\\theta) d\\theta

    where :math:`\\Gamma(\\theta)` is the fiber distribution function and :math:`\\psi(\\theta)` is the fiber energy function.

    The fiber distribution function is assumed to be a truncated Gaussian distribution 

    .. math::
        \\Gamma(\\theta) = d_e\\frac{1}{P}\\exp\\left(-\\frac{(\\theta-\\theta_0)}{2\\sigma^2}\\right) + \\frac{1-d_e}{\\pi}

    where :math:`P` is the normalization factor and :math:`d_e` is the anisotropy fraction.

    The fiber energy function is assumed to be an exponential function

    .. math::
        \\psi(\\epsilon) = \\frac{A}{B}\\left(\\frac{\\exp(B \\epsilon)-1}{B}-\\epsilon\\right)

    '''
    def __init__(self):
        #define integration rule
        self.nseg = 1
        self.order_quad = 10
        self.nint = self.nseg*self.order_quad
        self._theta_i = np.zeros(self.nint)
        self._theta_weight_i = np.zeros(self.nint)
        #if self.M is None or len(self.M) != 1:
        #    raise ValueError(self.__class__.__name__+" class needs exactly one fiber family")
        #self.m = self.M[0]
        self.Gamma_iterate = np.zeros(self.nint)
        self.nvector_iterate = np.zeros((self.nint,3))
        self.updateIntegrationTheta(-pi/2.,pi/2.)
        self.prefactor1 = np.sqrt(2*np.pi)
        self.prefactor2 = np.pi/2./np.sqrt(2)
        self.nvector_iterate[:,0],self.nvector_iterate[:,1] = np.cos(self._theta_i), np.sin(self._theta_i)
        self.Q = np.eye(3)
        self.param_default  = dict(A=1.,B=1.,mean_theta=0.,sigma=0.1,aniso_fraction=0.9)
        self.param_low_bd   = dict(A=0.001,B=1.,mean_theta=0.,sigma=0.1,aniso_fraction=0.)
        self.param_up_bd    = dict(A=100.,B=100.,mean_theta=0.,sigma=0.5,aniso_fraction=1)
        self.J = 1.

    def updateIntegrationTheta(self,T1,T2):
        x,w = np.polynomial.legendre.leggauss(self.order_quad)
        DeltaTheta = (T2-T1)/self.nseg
        for i in range(self.nseg):
            t1,t2 = T1+i*DeltaTheta, T1+(i+1)*DeltaTheta
            jac = (t2-t1)/2.
            dsize = (t2+t1)/2.
            xi = jac*x+dsize
            wi = jac*w
            self._theta_i[i*self.order_quad:(i+1)*self.order_quad] = xi
            self._theta_weight_i[i*self.order_quad:(i+1)*self.order_quad] = wi

    def fiber_distribution(self,normalize,theta,sigma,aniso_fraction,**extra_args):
        return aniso_fraction*np.exp(-theta*theta/(2.*sigma*sigma))/normalize + (1.-aniso_fraction)/np.pi

    def updateGammaIterate(self,mean_theta,sigma,aniso_fraction,**extra_args):
        normalize = self.prefactor1*sigma*erf(self.prefactor2/sigma)
        theta = self._theta_i - mean_theta
        self.Gamma_iterate = aniso_fraction*np.exp(-theta*theta/(2.*sigma*sigma))/normalize + (1.-aniso_fraction)/np.pi

    def update(self,F):
        self.E = (F.T@F-np.eye(3))/2.
        self.J = np.linalg.det(F)
        return

    def updateQ(self,n1,n2):
        #n1 is the reference x axis, and n2 is another vector in the tissue's plane
        n3 = np.cross(n1,n2) #normal
        n2 = np.cross(n3,n1) 
        n1 = n1/np.linalg.norm(n1)
        n2 = n2/np.linalg.norm(n2)
        n3 = n3/np.linalg.norm(n3)
        self.Q = np.array([n1,n2,n3])
        return

    def energy(self,F,theta):
        self.update(F)
        self.updateGammaIterate(**theta)
        return self._energy(**theta)

    def secondPK(self,F,theta):
        self.update(F)
        self.updateGammaIterate(**theta)
        return self._stress(**theta)

    def energy_stress(self,F,theta):
        self.update(F)
        self.updateGammaIterate(**theta)
        return self._energy_stress(**theta)
    
    def _energy(self,A,B,mean_theta,sigma,aniso_fraction,**extra_args):
        vectors = self.nvector_iterate@self.Q
        stretches = np.einsum('ij,ji->i',vectors,(self.E@vectors.T))
        stretches[stretches<0]=0.
        energy = A*np.sum(self.Gamma_iterate*(np.exp(B*stretches)/B-1./B-stretches)*self._theta_weight_i)
        return energy

    def _stress(self,A,B,mean_theta,sigma,aniso_fraction,**extra_args):
        vectors = self.nvector_iterate@self.Q
        stretches = np.einsum('ij,ji->i',vectors,(self.E@vectors.T))
        stretches[stretches<0]=0.
        tensors = np.array([np.outer(n,n) for n in vectors])
        stress = A*np.sum(self.Gamma_iterate*(np.exp(B*stretches)-1)*self._theta_weight_i*tensors.T,axis=-1)
        return stress

    def _energy_stress(self,A,B,mean_theta,sigma,aniso_fraction,**extra_args):
        vectors = self.nvector_iterate@self.Q
        stretches = np.einsum('ij,ji->i',vectors,(self.E@vectors.T))
        stretches[stretches<0]=0.
        tensors = np.array([np.outer(n,n) for n in vectors])
        energy = A*np.sum(self.Gamma_iterate*(np.exp(B*stretches)/B-1./B-stretches)*self._theta_weight_i)
        stress = A*np.sum(self.Gamma_iterate*(np.exp(B*stretches)-1)*self._theta_weight_i*tensors.T,axis=-1)
        return energy,stress
