from Examples import *
import pytest

mat_model_list = ['nh','yeoh','ls','mn','expI1','goh','Holzapfel','hgo','hy','volPenalty','polyI4','ArrudaBoyce','Gent','splineI1I4','StructModel']
samples_list = [UniaxialExtension,PlanarBiaxialExtension,TubeInflation, LinearSpring]

def test_mat_creation():
    #Test creating all individual material models
    for mname in mat_model_list:
        output = MatModel(mname)
        mm = output.models
        mm[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]
        assert isinstance(output, MatModel)
        assert len(output.models) == 1
        assert len(output.models[0].fiber_dirs) == 2 

def test_mat_addition():
    #Two ways of adding material models
    mat1, mat2 = MatModel('nh'), MatModel('goh')
    mat = mat1 + mat2
    assert isinstance(mat, MatModel)
    assert len(mat.models) == 2
    assert len(mat.parameters) == len(mat1.parameters) + len(mat2.parameters)

    mat3 = MatModel('nh','goh')
    assert isinstance(mat3, MatModel)
    assert len(mat3.models) == 2
    assert len(mat3.parameters) == len(mat1.parameters) + len(mat2.parameters)

def test_mat_reference():
    #Test that reference stress and energy are correctly calculated
    for mname in mat_model_list:
        if mname in ['splineI1I4']: #TODO: splineI1I4 needs a spline setting
            continue
        model = MatModel(mname)
        model.models[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]
        e, S = model.energy_stress(np.eye(3),model.parameters,stresstype='cauchy',incomp=True)
        assert e == pytest.approx(0.0)
        assert S == pytest.approx(np.zeros((3,3)))

def test_mat_partial_derivs():
    #Tests that the partial derivatives are correctly implemented
    for mname in mat_model_list:
        if mname in ['splineI1I4']:
            continue
        model = MatModel(mname)
        model.models[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]
        assert model._test(model.parameters) #raises error since the parameter names are different

def test_samples():
    #Test that the samples are correctly generated
    mat = MatModel('nh')
    for s in samples_list:
        sample = s(mat)
        assert isinstance(sample, SampleExperiment)
        assert sample._mat_model == mat

def test_samples_reference():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    for s in samples_list:
        sample = s(material)
        assert sample.disp_controlled(sample._x0, sample.parameters) == pytest.approx(0.0)
        assert sample.force_controlled(np.zeros_like(sample._x0), sample.parameters) == pytest.approx(sample._x0)
        params = sample.parameters
        for k in params.keys():
            params[k] *= 2
        sample.parameters = params
        assert sample.parameters == params
    
def test_layered_samples():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    sample = LayeredUniaxial(UniaxialExtension(material, disp_measure='length', force_measure='force'),UniaxialExtension(material,disp_measure='length', force_measure='force'))
    assert isinstance(sample, LayeredSamples)
    assert len(sample._samples) == 2
    assert sample._samples[0]._mat_model == material
    assert sample._samples[1]._mat_model == material
    assert sample.disp_controlled(sample._samples[0]._x0, sample.parameters) == pytest.approx(0.0)
    assert sample.force_controlled(np.zeros_like(sample._samples[0]._x0), sample.parameters) == pytest.approx(sample._samples[0]._x0)

    sample = LayeredPlanarBiaxial(PlanarBiaxialExtension(material,disp_measure='length',force_measure='force'),PlanarBiaxialExtension(material,disp_measure='length',force_measure='force'))
    assert isinstance(sample, LayeredSamples)
    assert len(sample._samples) == 2
    assert sample._samples[0]._mat_model == material
    assert sample._samples[1]._mat_model == material
    assert sample.disp_controlled(sample._samples[0]._x0, sample.parameters) == pytest.approx(0.0)
    assert sample.force_controlled(np.zeros_like(sample._samples[0]._x0), sample.parameters) == pytest.approx(sample._samples[0]._x0, 1e-3)

    sample = LayeredTube(TubeInflation(material),TubeInflation(material))
    assert isinstance(sample, LayeredSamples)
    assert len(sample._samples) == 2
    assert sample._samples[0]._mat_model == material
    assert sample._samples[1]._mat_model == material
    assert sample.disp_controlled([sample._samples[0]._x0], sample.parameters) != pytest.approx(0.0)
    assert sample.force_controlled(np.zeros_like(sample._samples[0]._x0), sample.parameters) != pytest.approx(sample._samples[0]._x0)

def test_mcmc():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    sample = UniaxialExtension(material)
    def prob_func(params):
        return 1.0, 0.0
    mcmc = MCMC(prob_func, sample.parameters)
    assert isinstance(mcmc, MCMC)
    mcmc.std[:] = 1.
    mcmc.run(5)
    assert mcmc._samples is not None

def test_param_fitter():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    sample = UniaxialExtension(material)
    inp = np.linspace(0,1,10)+1
    def sim_func(params):
        return sample.disp_controlled(inp, params)
    
    param_fitter = ParamFitter(sim_func, np.linspace(0,1,len(inp)), sample.parameters)
    assert isinstance(param_fitter, ParamFitter)
    result = param_fitter.fit()
    assert result is not None

def test_random_params():
    material = MatModel('goh','nh')
    mm = material.models
    mm[0].fiber_dirs = [np.array([cos(0.),sin(0.),0])]
    sample = UniaxialExtension(material)
    random_params = RandomParameters(sample.parameters)
    assert isinstance(random_params, RandomParameters)
    param_samples = random_params.sample(10)
    assert type(param_samples) is list
    assert len(param_samples) == 10
    assert len(param_samples[0]) == len(sample.parameters)


def test_unixex():
    output = unixex()
    assert output[0] == pytest.approx([1.0, 1.1111111111111112, 1.2222222222222223, 1.3333333333333333, 1.4444444444444444, 1.5555555555555556, 1.6666666666666665, 1.7777777777777777, 1.8888888888888888, 2.0])
    assert output[1] == pytest.approx([0.0, 3.889237649923175, 24.210969202431325, 306.3625384491437, 12166.895534560083, 2130684.6891950294, 2322650211.527586, 22534357192278.957, 2.830189006942182e+18, 6.826789038537727e+24])
    assert output[2] == pytest.approx([0.0, 0.2222222222222222, 0.4444444444444444, 0.6666666666666666, 0.8888888888888888, 1.1111111111111112, 1.3333333333333333, 1.5555555555555554, 1.7777777777777777, 2.0])
    assert output[3] == pytest.approx([1.000000000000003, 1.0096555981502184, 1.0189205712648177, 1.0277536117002184, 1.036130527984696, 1.0440426440692123, 1.0514940560882118, 1.0584984779904174, 1.0650762207778208, 1.0712516151400189])

def test_biaxex():
    output = biaxex()
    assert output[0] == pytest.approx([1.0, 1.1111111111111112, 1.2222222222222223, 1.3333333333333333, 1.4444444444444444, 1.5555555555555556, 1.6666666666666665, 1.7777777777777777, 1.8888888888888888, 2.0])
    assert output[1] == pytest.approx([0.254101373152648, 0.4283530730656771, 52.36024847136841, 7.605019052701357, 178192.56555630412, 24004.871190623522, 1841622820725.367, 253780758993.39755, 2.079354372883468e+24, 2.869974506675743e+23])
    assert output[2] == pytest.approx([0.0, 0.2222222222222222, 0.4444444444444444, 0.6666666666666666, 0.8888888888888888, 1.1111111111111112, 1.3333333333333333, 1.5555555555555554, 1.7777777777777777, 2.0])
    assert output[3] == pytest.approx([0.9649576634391019, 1.0739485623031386, 1.0008208331098385, 1.1733660460332205, 1.0044509001685087, 1.2829357327009965, 1.006445307545602, 1.3843445169845912, 1.0073732969995206, 1.4760209149414139])

def test_validate_tube():
    output = validate_tube()
    assert output[0] == pytest.approx(output[1])
    assert output[2] == pytest.approx(0.6702771145876144)

# def test_artery0Dmodel():
    # assert artery0Dmodel() == None

def test_randomex():
    assert randomex() == None
