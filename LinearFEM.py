import numpy as np
from scipy.linalg import solveh_banded
from scipy.optimize import minimize
import warnings
from collections import OrderedDict

class LinearFEM1D:
    def __init__(self,node_locations,DOF='zeros',compute=None,stiffness='banded'):
        self.nNodes = len(node_locations)
        self.nElems = self.nNodes-1
        self.node_locations = node_locations.copy()
        self._compute = compute
        #for 1 point GQ (TODO add the option of having higher GQ)
        self.N,self.dN = np.array([0.5,0.5]),np.array([-1.,1.])
        self.outerNN = np.outer(self.N,self.N)
        self.outerNdN = np.outer(self.N,self.dN)
        self.outerdNN = np.outer(self.dN,self.N)
        self.outerdNdN = np.outer(self.dN,self.dN)

        if DOF=='equal':
            self.DOF = self.node_locations.copy()
        elif DOF=='zeros':
            self.DOF = np.zeros_like(self.node_locations)
        elif type(DOF) is np.ndarray:
            self.DOF = DOF
        elif callable(DOF):
            self.DOF = DOF(self.node_locations)
        elif type(DOF) is int or type(DOF) is float:
            self.DOF = np.ones(self.nNodes)*DOF
        else:
            raise ValueError(DOF," not allowed for DOF argument")
        self.elem_lengths = np.array([self.node_locations[i+1]-self.node_locations[i] for i in range(0,self.nElems)]) 
        self.elem_locations = np.array([(self.node_locations[i+1]+self.node_locations[i])/2. for i in range(0,self.nElems)])

        self.energy = 0.
        self.fglobal = np.zeros(self.nNodes)
        self.stiffness=stiffness
        if stiffness=='full':
            self.Kglobal = np.zeros([self.nNodes,self.nNodes])
        elif stiffness=='banded':
            self.Kglobal = np.zeros([2,self.nNodes])
        else:
            self.Kglobal = None

    @property
    def DOF(self):
        return self._DOF

    @DOF.setter
    def DOF(self,x):
        assert(len(x)==self.nNodes)
        # Ensure variable is defined
        try:
            #make it writeable
            self._DOF.flags.writeable = True
            #fill the values with that of x
            self._DOF[:] = x
        except AttributeError:
           self._DOF = x.copy()
        #make it non-writeable
        self._DOF.flags.writeable = False

    def addDOF(self,x,i=None):
        self._DOF.flags.writeable = True
        if i is None:
            assert(len(x)==self.nNodes)
            self._DOF += x
        else:
            assert(type(x) is float or type(x) is int)
            self._DOF[i] += x
        self._DOF.flags.writeable = False

    def derivs(self):
        #calculate the derivatives
        y = self.DOF
        dy = np.empty(self.nElems)
        for i in range(0,self.nElems):
            edof = y[i:i+2]
            dy[i] = np.sum(edof*self.dN)/self.elem_lengths[i]

        return dy

    @property
    def compute(self):
        return self._compute

    @compute.setter
    def compute(self,f):
        assert(callable(f))
        self._compute = f
        self.consistency_check_compute()
    
    def zero_out(self):
        self.energy = 0.
        self.fglobal.fill(0.)
        if self.Kglobal is not None:
            self.Kglobal.fill(0.)

    def consistency_check_compute(self):
        #TODO need to make sure this is OK
        Y = self.DOF[0]
        y = np.random.random()*Y
        X = 0.
        dy = np.random.random()
        eps = 1e-6
        self._compute(dof=y,deriv=dy,location=X,eid=0)
        Psi,dPsidy,dPsiddy = self._compute.energy(), self._compute.variation_dof(), self._compute.variation_deriv()
        if Psi is None:
            warnings.warn("The compute function does not provide energy. Cannot check consistency")
            return
        self._compute(dof=y+eps,deriv=dy,location=X,eid=0)
        Psi2 = self._compute.energy()
        self._compute(dof=y,deriv=dy+eps,location=X,eid=0)
        Psi3 = self._compute.energy()
        if np.abs((Psi2-Psi)/eps-dPsidy)>1e-3:
           message = f"Consistency of compute function failed {Psi}, {Psi2}, {np.abs((Psi2-Psi)/eps-dPsidy)}, {dPsidy}"
           warnings.warn(message)
        if np.abs((Psi3-Psi)/eps-dPsiddy)>1e-3:
           message = f"Consistency of compute function failed {Psi}, {Psi3}, {np.abs((Psi3-Psi)/eps-dPsiddy)}, {dPsiddy}"
           warnings.warn(message)

    def elem_cal(self,edof,le,X,eid,energy,force,stiffness):
        assert callable(self._compute),"Compute function must be set before any computation"
        #TODO need to improve this, this is the point where it is connected to the model
        
        y,X = np.sum(edof*self.N), np.sum(X*self.N)
        dy = np.sum(edof*self.dN)/le 
        
        #call the compute function
        self._compute(dof=y,deriv=dy,location=X,eid=eid)
        Psi,dPsidy,dPsiddy = self._compute.energy(), self._compute.variation_dof(), self._compute.variation_deriv()

        if energy:
            en = Psi*le
        else:
            en = None
        if force:
            f = dPsidy*self.N*le + dPsiddy*self.dN
        else:
            f = None
        
        if stiffness:#TODO better symbols below
            eps = 1e-6
            self._compute(dof=y+eps,deriv=dy,location=X,eid=eid)
            dPsidy1,dPsiddy1 = self._compute.variation_dof(), self._compute.variation_deriv()
            self._compute(dof=y,deriv=dy+eps,location=X,eid=eid)
            dPsidy2,dPsiddy2 = self._compute.variation_dof(), self._compute.variation_deriv()

            dy2,ddydy,dyddy,ddy2 = (dPsidy1-dPsidy)/eps, (dPsiddy1-dPsiddy)/eps, (dPsidy2-dPsidy)/eps, (dPsiddy2-dPsiddy)/eps
            K = le*dy2*self.outerNN + ddydy*self.outerNdN + dyddy*self.outerdNN + ddy2*self.outerdNdN/le
        else:
            K = None

        return en,f,K

    def node_cal(self,ndof,X,nodeid,energy,force,stiffness):
        #assume that nodal contributions do not depend on derivatives
        self._compute(dof=ndof,deriv=None,location=X,nodeid=nodeid)
        Psi,dPsidy = self._compute.energy(), self._compute.variation_dof()

        if energy:
            en = Psi
        else:
            en = None
        if force:
            f = dPsidy
        else:
            f = None

        if stiffness:
            eps = 1.e-6
            self._compute(dof=ndof+eps,deriv=None,location=X,nodeid=nodeid)
            dPsidy1 = self._compute.variation_dof()
            K = (dPsidy1-dPsidy)/eps
        else:
            K = None

        return en,f,K

    def assemble(self,energy=True,force=True,stiffness=True):
        self.zero_out()
        #Sum over all elements
        for i in range(0,self.nElems):
            try:
                e,f,K = self.elem_cal(self.DOF[i:i+2],self.elem_lengths[i],self.node_locations[i:i+2], i, energy, force, stiffness)
            except:
                raise ValueError("Element calculation failed")
            if np.isnan(f).any() or np.isnan(K).any():
                raise ValueError('NaN found in the force vector')
    
            if energy:
                self.energy += e

            if force:
                self.fglobal[i:i+2] += f
            
            if stiffness:
                if self.stiffness=='full':
                    self.Kglobal[i:i+2,i:i+2] += K
                elif self.stiffness=='banded':
                    self.Kglobal[1,i:i+2] += np.diag(K)
                    self.Kglobal[0,i+1] += (K[0,1]+K[1,0])/2.

        for i in [0,self.nNodes-1]:
            e,f,K = self.node_cal(self.DOF[i],self.node_locations[i],i,energy,force,stiffness)
            if energy:
                self.energy += e
            if force:
                self.fglobal[i] += f
            if stiffness:
                if self.stiffness=='full':
                    self.Kglobal[i,i] += K
                elif self.stiffness=='banded':
                    self.Kglobal[1,i] += K
        return

    def force_energy(self,BClogic=None):
        self.assemble(True,True,False)
        if BClogic is None:
            return self.energy, self.fglobal

        assert(len(BClogic)==self.nNodes)
        return self.energy, self.fglobal[np.logical_not(BClogic)]

    def force(self,BClogic=None):
        self.assemble(False,True,False)
        if BClogic is None:
            return self.fglobal

        assert(len(BClogic)==self.nNodes)
        return self.fglobal[np.logical_not(BClogic)]

    def force_stiffness(self,BClogic=None):
        self.assemble(False,True,True)
        if BClogic is None:
            return self.fglobal, self.Kglobal

        assert(len(BClogic)==self.nNodes)
        fglobal_reduced = self.fglobal[np.logical_not(BClogic)]

        if self.stiffness=='full':
            stiff_reduced = self.Kglobal[np.logical_not(BClogic), :][:,np.logical_not(BClogic)]
        elif self.stiffness=='banded':
            stiff_reduced = self.Kglobal[:,np.logical_not(BClogic)]
            stiff_reduced[0,0] = 0.

        return fglobal_reduced, stiff_reduced

    def newton_step(self,BClogic=None):
        f,K = self.force_stiffness(BClogic)
        if self.stiffness=='full':
            try:
                dx = -np.linalg.solve(K,f)
            except:
                print('Singular matrix\n',K)
        elif self.stiffness=='banded':
            try:
                dx = -solveh_banded(K,f)
            except:
                print('Singular matrix\n',K)
        print('Newton step:', dx)
        if BClogic is not None:
            dx_all = np.zeros(self.nNodes)
            dx_all[np.logical_not(BClogic)] = dx

        self.addDOF(dx_all)

    def global_consistency_check(self):
        dx = np.zeros(self.nNodes)
        eps = 1e-6
        self.assemble(False,True,False)
        ftrue = self.fglobal.copy()
        fnum = np.zeros_like(ftrue)
        for i in range(0,nNodes):
            dx[i]=eps
            self.DOF = self.DOF + dx
            self.assemble(True,False,False)
            en1=self.energy
            dx[i]=-eps
            self.DOF = self.DOF + dx
            self.assemble(True,False,False)
            en2=self.energy
            fnum[i]=(en1-en2)/(2.*eps)
            dx[i]=0.
        print("Force consistency: ",np.norm(ftrue-fnum),ftrue,fnum)

    #TODO's 
    #1. add non-zero essential BC - done at the global level, but with this the quadratic convergence only starts later in the iterations
    #2. add consistency checks at the element and global levels - DONE, need to check it for tube
    #3. add higher GQ orders
    #4. write theory used here to explain the symbols

class InternalVariables:
    def __init__(self,params,nElems):
        if type(params) is dict:
            self.param_dict = OrderedDict(params)
            self.list = False
            self.nvar = 0
            self.varkeys = []
        elif type(params) is list:
            self.param_dict = [OrderedDict(p) for p in params]
            self.list = True
            self.nvar = 0
            self.varkeys = [[] for p in params]
        self.nElems = nElems
        self.variables = np.zeros([0,self.nElems])

    def vary(self,keys=[]): #TODO make it work for layered models
        if self.list: 
            self.nvar = sum(len(el) for el in keys)
        else:
            self.nvar = len(keys)
        self.variables = np.zeros([self.nvar,self.nElems])
        self.varkeys = keys
        if self.list:
            assert len(keys)==len(self.param_dict),"For list of parameters, they keys should also be given as a list of list"
            i=0
            for j,keys in enumerate(self.varkeys):
                for key in keys:
                    self.variables[i,:] = self.param_dict[j][key]
                    i += 1
        else:
            for i,key in enumerate(keys):
                self.variables[i,:] = self.param_dict[key]
        
    def __call__(self,e=None):
        if len(self.varkeys)==0:
            return self.param_dict
        if e is None:
            raise ValueError("Element ID is required for interpolating internal variables")
        #indices = [e,e+1]
        #N = np.array([0.5,0.5])
        #assert(len(indices)==len(N))
        var = self.variables[:,e]#@N
        if not self.list:
            for i,key in enumerate(self.varkeys):
                self.param_dict[key] = var[i]
            return self.param_dict
        else:
            i = 0
            for listi,keysi in enumerate(self.varkeys):
                for key in keysi:
                    self.param_dict[listi][key] = var[i]
                    i += 1
            return self.param_dict

    def get(self,key,i=None):
        if self.list:
            assert(i is not None)
            assert(i>=0 and i<len(self.varkeys))
            if not(key in self.varkeys[i]):
                return np.ones(self.nElems)*self.param_dict[i][key]
            #find the location of key
            j=sum(len(k) for k in self.varkeys[:i])+self.varkeys[i].index(key)
            return self.variables[j,:]
        if not(key in self.varkeys):
            return np.ones(self.nElems)*self.param_dict[key]
        #find the location of key
        j=self.varkeys.index(key)
        return self.variables[j,:]
'''
#Example of how to use the InternalVariable class
from SampleExperiment import *
from MatModel import *
from LinearFEM import *
material = MatModel('goh','nh')
mm = material.models
mm[0].fiber_dirs = [np.array([0,cos(0.1),sin(0.1)]),np.array([0,cos(-0.1),sin(-0.1)])]
intima = UniformAxisymmetricTubeInflationExtension(material)
media = UniformAxisymmetricTubeInflationExtension(material)
artery = LayeredTube(intima,media)
combined_parameters = artery.parameters
theta=InternalVariables(combined_parameters,10)
theta.vary([['Ri'],['mu','Ri']])
theta.variables[0]=np.linspace(1,2,10)
theta.variables[1]=np.linspace(2,3,10)
theta.variables[2]=np.linspace(3,4,10)
theta.get('Ri',1)[:]=np.linspace(1,10,10)
theta.get('Ri',1)
theta(1)
'''
