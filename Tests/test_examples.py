from Examples import *

def test_mat_creation():
    assert str(type(mat_creation())) == "<class 'pymecht.MatModel.MatModel'>"
    assert len(mat_creation().models) == 2
    assert mat_creation().param_names == [{'k1_0': 'k1', 'k2_0': 'k2', 'k3_0': 'k3'}, {'mu_1': 'mu'}]
    assert mat_creation().parameters == {'k1_0': 10.0, 'k2_0': 10.0, 'k3_0': 0.1, 'mu_1': 1.0}
    assert list(mat_creation().models[0].fiber_dirs[0]) == [1., 0., 0.]

def test_unixex():
    assert list(unixex()[0]) == [1.0, 1.1111111111111112, 1.2222222222222223, 1.3333333333333333, 1.4444444444444444, 1.5555555555555556, 1.6666666666666665, 1.7777777777777777, 1.8888888888888888, 2.0]
    assert list(unixex()[1]) == [0.0, 3.889237649923175, 24.210969202431325, 306.3625384491437, 12166.895534560083, 2130684.6891950294, 2322650211.527586, 22534357192278.957, 2.830189006942182e+18, 6.826789038537727e+24]
    assert list(unixex()[2]) == [0.0, 0.2222222222222222, 0.4444444444444444, 0.6666666666666666, 0.8888888888888888, 1.1111111111111112, 1.3333333333333333, 1.5555555555555554, 1.7777777777777777, 2.0]
    assert list(unixex()[3]) == [1.000000000000003, 1.0096555981502184, 1.0189205712648177, 1.0277536117002184, 1.036130527984696, 1.0440426440692123, 1.0514940560882118, 1.0584984779904174, 1.0650762207778208, 1.0712516151400189]

def test_biaxex():
    assert biaxex() == None

def test_validate_tube():
    assert validate_tube() == None

# def test_artery0Dmodel():
    # assert artery0Dmodel() == None

def test_randomex():
    assert randomex() == None