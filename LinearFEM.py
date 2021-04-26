import numpy as np
from scipy.linalg import solveh_banded
from scipy.optimize import minimize

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
            assert(len(DOF)==self.nNodes)
            self.DOF = DOF
        elif callable(DOF):
            self.DOF = DOF(self.node_locations)
            assert(len(self.DOF)==self.nNodes)
        elif type(DOF) is int or type(DOF) is float:
            self.DOF = np.ones(self.nNodes)*DOF
        else:
            raise ValueError(DOF," not allowed for DOF argument")
        self.dof=self.DOF
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
        return self._DOF.copy()

    @DOF.setter
    def DOF(self,x):
        assert(len(x)==self.nNodes)
        self._DOF = x.copy()

    @property
    def dof(self):
        return self._dof.copy()

    @dof.setter
    def dof(self,x):
        assert(len(x)==self.nNodes)
        self._dof = x.copy()

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
       Y = self.dof[np.random.randint(0,self.nNodes)]
       y = np.random.random()*Y
       dY = 0.
       X = 0.
       dy = np.random.random()
       eps = 1e-6
       Psi,dPsidy,dPsiddy = self._compute(y=y,dy=dy,Y=Y,dY=dY,X=X)
       Psi2,dPsidy2,dPsiddy2 = self._compute(y=y,dy=dy,Y=Y,dY=dY,X=X)
       Psi3,dPsidy3,dPsiddy3 = self._compute(y=y,dy=dy,Y=Y,dY=dY,X=X)
       if np.abs((Psi2-Psi)/eps-dPsidy)>1e-3:
           warnings.warn("Consistency of compute function failed",Psi,Psi2)
       if np.abs((Psi3-Psi)/eps-dPsiddy)>1e-3:
           warnings.warn("Consistency of compute function failed",Psi,Psi3)

    def elem_cal(self,dof,DOF,le,X,energy,force,stiffness):
        #assert callable(self._compute),"Compute function must be set before any computation"
        #TODO need to improve this, this is the point where it is connected to the model
        
        y,Y,X = np.sum(dof*self.N), np.sum(DOF*self.N), np.sum(X*self.N)
        dy,dY = np.sum(dof*self.dN)/le, np.sum(DOF*self.dN)/le
        
        #call the compute function
        Psi,dPsidy,dPsiddy = self._compute(y=y,dy=dy,Y=Y,dY=dY,X=X)

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
            _,dPsidy1,dPsiddy1 = self._compute(y=y+eps,dy=dy,Y=Y,dY=dY,X=X)
            _,dPsidy2,dPsiddy2 = self._compute(y=y,dy=dy+eps,Y=Y,dY=dY,X=X)

            dy2,ddydy,dyddy,ddy2 = (dPsidy1-dPsidy)/eps, (dPsiddy1-dPsiddy)/eps, (dPsidy2-dPsidy)/eps, (dPsiddy2-dPsiddy)/eps
            K = le*dy2*self.outerNN + ddydy*self.outerNdN + dyddy*self.outerdNN + ddy2*self.outerdNdN/le
        else:
            K = None

        return en,f,K

    def assemble(self,energy=True,force=True,stiffness=True):
        self.zero_out()
        for i in range(0,self.nElems):
            try:
                e,f,K = self.elem_cal(self.dof[i:i+2],self.DOF[i:i+2],self.elem_lengths[i],self.node_locations[i:i+2], energy, force, stiffness)
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
        print(dx)
        if BClogic is not None:
            dx_all = np.zeros(self.nNodes)
            dx_all[np.logical_not(BClogic)] = dx

        self.dof = self.dof + dx_all

    #TODO's 
    #1. add non-zero essential BC
    #2. add consistency checks at the element and global levels
    #3. ability to set the _compute function 
