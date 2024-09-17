import pymecht as pmt
import pytest
import numpy as np

def test_arbitrary_MatModel_I1():
    model = pmt.ARB('mu/2.*(I1-3.)','mu=1.','mu=0.01','mu=10.')
    mat = pmt.MatModel(model)
    mat2 = pmt.MatModel('NH')
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
    
def test_arbitrary_MatModel_I2():
    model = pmt.ARB('c1*(I1-3.) + c2*(I2-3.)','c1=1., c2=1.','c1=0.0001, c2=0.','c1=100., c2=100.')
    mat = pmt.MatModel(model)
    mat2 = pmt.MatModel('MR')
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
    
def test_arbitrary_MatModel_J():
    model = pmt.ARB('kappa/2.*(J-1.)**2','kappa=1.','kappa=1.','kappa=1.')
    mat = pmt.MatModel(model)
    mat2 = pmt.MatModel('volPenalty')
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
    
def test_arbitrary_MatModel_I4():
    model = pmt.ARB('d1*(I4-1.)+d2*(I4-1.)**2+d3*(I4-1.)**3', 'd1=0., d2=1., d3=1.', 'd1=0.0001, d2=0., d3=0.', 'd1=100., d2=100., d3=100.')
    mat = pmt.MatModel(model)
    (mat.models)[0].fiber_dirs = np.array([0.5,0.5,0])
    mat2 = pmt.MatModel('polyI4')
    (mat2.models)[0].fiber_dirs = np.array([0.5,0.5,0])
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
    
def test_combined_arbitrary_MatModel():
    model_I1 = pmt.ARB('mu/2.*(I1-3.)','mu=1.','mu=0.01','mu=10.')
    model_I2 = pmt.ARB('c1*(I1-3.) + c2*(I2-3.)','c1=1., c2=1.','c1=0.0001, c2=0.','c1=100., c2=100.')
    model_J = pmt.ARB('kappa/2.*(J-1.)**2','kappa=1.','kappa=1.','kappa=1.')
    model_I4 = pmt.ARB('d1*(I4-1.)+d2*(I4-1.)**2+d3*(I4-1.)**3', 'd1=0., d2=1., d3=1.', 'd1=0.0001, d2=0., d3=0.', 'd1=100., d2=100., d3=100.')
    mat = pmt.MatModel(model_I1, model_I2, model_J, model_I4)
    (mat.models)[-1].fiber_dirs = np.array([0.5,0.5,0])
    mat2 = pmt.MatModel('NH', 'MR', 'volPenalty', 'polyI4')
    (mat2.models)[-1].fiber_dirs = np.array([0.5,0.5,0])
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))

# Define a random deformation gradient
random_F = np.array([[0.85862107, 0.91032637, 0.80119846], [0.16268142, 0.8596134, 0.17696991], [0.93450122, 0.30132757, 0.55042838]])

test_arbitrary_MatModel_I1()
test_arbitrary_MatModel_I2()
test_arbitrary_MatModel_J()
test_arbitrary_MatModel_I4()
test_combined_arbitrary_MatModel()