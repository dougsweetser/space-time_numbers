#!/usr/bin/env python
# coding: utf-8

# # Developing Quaternions for iPython
import math
from copy import deepcopy
import pytest
import sympy as sp
from QH import QH, QHStates

Q: QH = QH([1, -2, -3, -4], q_type="Q")
P: QH = QH([0, 4, -3, 0], q_type="P")
R: QH = QH([3, 0, 0, 0], q_type="R")
C: QH = QH([2, 4, 0, 0], q_type="C")
t, x, y, z = sp.symbols("t x y z")
q_sym: QH = QH([t, x, y, x * y * z])
q22: QH = QH([2, 2, 0, 0])
q44: QH = QH([4, 4, 0, 0])
q4321: QH = QH([4, 3, 2, 1])
q1324: QH = QH([1, 3, 2, 4])
q2244: QH = QH([2, 2, 4, 4])


def test_1000_qt():
    assert Q.t == 1


def test_1010_subs():
    q_z = q_sym.subs({t: 1, x: 2, y: 3, z: 4})
    print("t x y xyz sub 1 2 3 4: ", q_z)
    assert q_z.equals(QH([1, 2, 3, 24]))


def test_1020_scalar():
    q_z = Q.scalar()
    print("scalar(q): ", q_z)
    assert q_z.t == 1
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1030_vector():
    q_z = Q.vector()
    print("vector(q): ", q_z)
    assert q_z.t == 0
    assert q_z.x == -2
    assert q_z.y == -3
    assert q_z.z == -4


def test_1040_xyz():
    q_z = Q.xyz()
    print("q.xyz()): ", q_z)
    assert q_z[0] == -2
    assert q_z[1] == -3
    assert q_z[2] == -4


def test_1050_q_0():
    q_z = Q.q_0()
    print("q_0: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1060_q_1():
    q_z: QH = QH.q_1()
    print("q_1: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1070_q_i():
    q_z = Q.q_i()
    print("q_i: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 1
    assert q_z.y == 0
    assert q_z.z == 0


def test_1080_q_j():
    q_z = Q.q_j()
    print("q_j: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0
    assert q_z.y == 1
    assert q_z.z == 0


def test_1090_q_k():
    q_z = Q.q_k()
    print("q_k: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 1


def test_1100_q_random():
    q_z = QH().q_random()
    print("q_random():", q_z)
    assert -1 <= q_z.t <= 1
    assert -1 <= q_z.x <= 1
    assert -1 <= q_z.y <= 1
    assert -1 <= q_z.z <= 1


def test_1200_equals():
    assert Q.equals(Q)
    assert not Q.equals(P)


def test_1210_conj_0():
    q_z = Q.conj()
    print("q_conj 0: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == 4


def test_1220_conj_1():
    q_z = Q.conj(1)
    print("q_conj 1: ", q_z)
    assert q_z.t == -1
    assert q_z.x == -2
    assert q_z.y == 3
    assert q_z.z == 4


def test_1230_conj_2():
    q_z = Q.conj(2)
    print("q_conj 2: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 2
    assert q_z.y == -3
    assert q_z.z == 4


def test_1240_conj_q():
    q_z = Q.conj_q(Q)
    print("conj_q(conj_q): ", q_z)
    assert q_z.t == -1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == -4


def sign_1250_flips():
    q_z = Q.flip_signs()
    print("sign_flips: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == 4


def test_1260_vahlen_conj_minus():
    q_z = Q.vahlen_conj()
    print("q_vahlen_conj -: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == 4


def test_1270_vahlen_conj_star():
    q_z = Q.vahlen_conj("*")
    print("q_vahlen_conj *: ", q_z)
    assert q_z.t == 1
    assert q_z.x == -2
    assert q_z.y == -3
    assert q_z.z == 4


def test_1280_vahlen_conj_prime():
    q_z = Q.vahlen_conj("'")
    print("q_vahlen_conj ': ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == -4


def test_1290_square():
    q_z = Q.square()
    print("square: ", q_z)
    assert q_z.t == -28
    assert q_z.x == -4
    assert q_z.y == -6
    assert q_z.z == -8


def test_1300_norm_squared():
    q_z = Q.norm_squared()
    print("norm_squared: ", q_z)
    assert q_z.t == 30
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1310_norm_squared_of_vector():
    q_z = Q.norm_squared_of_vector()
    print("norm_squared_of_vector: ", q_z)
    assert q_z.t == 29
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1320_abs_of_q():
    q_z = P.abs_of_q()
    print("abs_of_q: ", q_z)
    assert q_z.t == 5
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1330_normalize():
    q_z = P.normalize()
    print("q_normalized: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0.8
    assert math.isclose(q_z.y, -0.6)
    assert q_z.z == 0


def test_1340_abs_of_vector():
    q_z = P.abs_of_vector()
    print("abs_of_vector: ", q_z)
    assert q_z.t == 5
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1350_add():
    q_z = Q.add(P)
    print("add: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == -6
    assert q_z.z == -4


def test_1360_dif():
    q_z = Q.dif(P)
    print("dif: ", q_z)
    assert q_z.t == 1
    assert q_z.x == -6
    assert q_z.y == 0
    assert q_z.z == -4


def test_1370_product():
    q_z = Q.product(P)
    print("product: ", q_z)
    assert q_z.t == -1
    assert q_z.x == -8
    assert q_z.y == -19
    assert q_z.z == 18


def test_1380_product_even():
    q_z = Q.product(P, kind="even")
    print("product, kind even: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 4
    assert q_z.y == -3
    assert q_z.z == 0


def test_1390_product_odd():
    q_z = Q.product(P, kind="odd")
    print("product, kind odd: ", q_z)
    assert q_z.t == 0
    assert q_z.x == -12
    assert q_z.y == -16
    assert q_z.z == 18


def test_1400_product_even_minus_odd():
    q_z = Q.product(P, kind="even_minus_odd")
    print("product, kind even_minus_odd: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 16
    assert q_z.y == 13
    assert q_z.z == -18


def test_1410_product_reverse():
    q1q2_rev = Q.product(P, reverse=True)
    q2q1 = P.product(Q)
    assert q1q2_rev.equals(q2q1)


def test_1430_inverse():
    q_z = P.inverse()
    print("inverse: ", q_z)
    assert q_z.t == 0
    assert q_z.x == -0.16
    assert q_z.y == 0.12
    assert q_z.z == 0


def test_1440_divide_by():
    q_z = Q.divide_by(Q)
    print("divide_by: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test_1450_triple_product():
    q_z = Q.triple_product(P, Q)
    print("triple product: ", q_z)
    assert q_z.t == -2
    assert q_z.x == 124
    assert q_z.y == -84
    assert q_z.z == 8


def test_1460_rotate():
    q_z = Q.rotate(QH([0, 1, 0, 0]))
    print("rotate: ", q_z)
    assert q_z.t == 1
    assert q_z.x == -2
    assert q_z.y == 3
    assert q_z.z == 4


def test_1470_rotation_and_or_boost():
    q1_sq = Q.square()
    beta: float = 0.003
    gamma = 1 / math.sqrt(1 - beta ** 2)
    h = QH([gamma, gamma * beta, 0, 0])
    q_z = Q.rotation_and_or_boost(h)
    q_z2 = q_z.square()
    print("q1_sq: ", q1_sq)
    print("boosted: ", q_z)
    print("boosted squared: ", q_z2)
    assert round(q_z2.t, 5) == round(q1_sq.t, 5)


def test_1471_Lorentz_next_rotation():
    with pytest.raises(ValueError):
        QH.Lorentz_next_rotation(Q, q4321)
    next_rotation = QH.Lorentz_next_rotation(Q, q1324)
    print("next_rotation: ", next_rotation)
    assert next_rotation.t == 0
    rot = q2244.rotation_and_or_boost(next_rotation)
    assert math.isclose(rot.t, 2)
    assert math.isclose(rot.square().t, q2244.square().t)
    next_rotation = QH.Lorentz_next_rotation(Q, Q)
    assert next_rotation.equals(Q.vector().normalize())


def test_1472_Lorentz_next_boost():
    with pytest.raises(ValueError):
        QH.Lorentz_next_boost(Q, q4321)
    next_boost = QH.Lorentz_next_boost(Q, q1324)
    print(f"next_boost: {next_boost}")
    assert next_boost.t != 0
    boost = q2244.rotation_and_or_boost(next_boost)
    assert math.isclose(boost.square().t, q2244.square().t)


def test_1480_Lorentz_by_rescaling():
    q2 = Q.square()
    rescale = q22.Lorentz_by_rescaling(op=q22.add, h=q22)
    print("rescale_q_22+q22: ", rescale)
    print(rescale.equals(q44))
    rescale = q22.Lorentz_by_rescaling(op=q22.dif, h=q22)
    print("rescale_q22-q22: ", rescale)
    rescale = Q.Lorentz_by_rescaling(op=Q.dif, h=Q)
    print("rescale Q-Q: ", rescale)
    print(rescale.equals(Q))
    rescale = Q.Lorentz_by_rescaling(op=Q.add, h=P)
    print("rescale_Q+P: ", rescale)
    r2 = rescale.square()
    assert math.isclose(r2.t, q2.t)
    rescale = q22.Lorentz_by_rescaling(op=q22.add, h=Q)
    print("rescale_q22+Q: ", rescale)
    print(rescale.equals(q22))


def test_1490_g_shift():
    q1_sq = Q.square()
    q_z = Q.g_shift(0.003)
    q_z2 = q_z.square()
    q_z_minimal = Q.g_shift(0.003, g_form="minimal")
    q_z2_minimal = q_z_minimal.square()
    print("q1_sq: ", q1_sq)
    print("g_shift: ", q_z)
    print("g squared: ", q_z2)
    assert q_z2.t != q1_sq.t
    assert q_z2.x == q1_sq.x
    assert q_z2.y == q1_sq.y
    assert q_z2.z == q1_sq.z
    assert q_z2_minimal.t != q1_sq.t
    assert q_z2_minimal.x == q1_sq.x
    assert q_z2_minimal.y == q1_sq.y
    assert q_z2_minimal.z == q1_sq.z


def test_1500_sin():
    assert QH([0, 0, 0, 0]).sin().equals(QH().q_0())
    assert Q.sin().equals(QH(
        [
            91.7837157840346691,
            -21.8864868530291758,
            -32.8297302795437673,
            -43.7729737060583517,
        ]
    )
    )
    assert P.sin().equals(QH([0, 59.3625684622310033, -44.5219263466732542, 0]))
    assert R.sin().equals(QH([0.1411200080598672, 0, 0, 0]))
    assert C.sin().equals(QH([24.8313058489463785, -11.3566127112181743, 0, 0]))


def test_1510_cos():
    assert QH([0, 0, 0, 0]).cos().equals(QH().q_1())
    assert Q.cos().equals(QH([
        58.9336461679439481,
        34.0861836904655959,
        51.1292755356983974,
        68.1723673809311919,
    ]))
    assert P.cos().equals(QH([74.2099485247878476, 0, 0, 0]))
    assert R.cos().equals(QH([-0.9899924966004454, 0, 0, 0]))
    assert C.cos().equals(QH([-11.3642347064010600, -24.8146514856341867, 0, 0]))


def test_1520_tan():
    assert QH([0, 0, 0, 0]).tan().equals(QH().q_0())
    assert Q.tan().equals(QH([0.0000382163172501,
                              -0.3713971716439372,
                              -0.5570957574659058,
                              -0.7427943432878743, ]))
    assert P.tan().equals(QH([0, 0.7999273634100760, -0.5999455225575570, 0]))
    assert R.tan().equals(QH([-0.1425465430742778, 0, 0, 0]))
    assert C.tan().equals(QH([-0.0005079806234700, 1.0004385132020521, 0, 0]))


def test_1530_sinh():
    assert QH([0, 0, 0, 0]).sinh().equals(QH().q_0())
    assert Q.sinh().equals(
        QH(
            [
                0.7323376060463428,
                0.4482074499805421,
                0.6723111749708131,
                0.8964148999610841,
            ]
        )
    )
    assert P.sinh().equals(QH([0, -0.7671394197305108, 0.5753545647978831, 0]))
    assert R.sinh().equals(QH([10.0178749274099026, 0, 0, 0]))
    assert C.sinh().equals(QH([-2.3706741693520015, -2.8472390868488278, 0, 0]))


def test_1540_cosh():
    assert QH([0, 0, 0, 0]).cosh().equals(QH().q_1())
    assert Q.cosh().equals(QH(
        [
            0.9615851176369565,
            0.3413521745610167,
            0.5120282618415251,
            0.6827043491220334,
        ]
    )
    )
    assert P.cosh().equals(QH([0.2836621854632263, 0, 0, 0]))
    assert R.cosh().equals(QH([10.0676619957777653, 0, 0, 0]))
    assert C.cosh().equals(QH([-2.4591352139173837, -2.7448170067921538, 0, 0]))


def test_1550_tanh():
    assert QH([0, 0, 0, 0]).tanh().equals(QH().q_0())
    assert Q.tanh().equals(
        QH(
            [
                1.0248695360556623,
                0.1022956817887642,
                0.1534435226831462,
                0.2045913635775283,
            ]
        )
    )
    assert P.tanh().equals(QH([0, -2.7044120049972684, 2.0283090037479505, 0]))
    assert R.tanh().equals(QH([0.9950547536867305, 0, 0, 0]))
    assert C.tanh().equals(QH([1.0046823121902353, 0.0364233692474038, 0, 0]))


def test_1560_exp():
    assert QH([0, 0, 0, 0]).exp().equals(QH().q_1())
    assert Q.exp().equals(QH(
        [
            1.6939227236832994,
            0.7895596245415588,
            1.1843394368123383,
            1.5791192490831176,
        ]
    )
    )
    assert P.exp().equals(QH([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0]))
    assert R.exp().equals(QH([20.0855369231876679, 0, 0, 0]))
    assert C.exp().equals(QH([-4.8298093832693851, -5.5920560936409816, 0, 0]))


def test_1570_ln():
    assert Q.ln().exp().equals(Q)
    assert Q.ln().equals(QH(
        [
            1.7005986908310777,
            -0.5151902926640850,
            -0.7727854389961275,
            -1.0303805853281700,
        ]
    )
    )
    assert P.ln().equals(QH([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0]))
    assert R.ln().equals(QH([1.0986122886681098, 0, 0, 0]))
    assert C.ln().equals(QH([1.4978661367769954, 1.1071487177940904, 0, 0]))


def test_1580_q_2_q():
    assert Q.q_2_q(P).equals(QH(
        [
            -0.0197219653530713,
            -0.2613955437374326,
            0.6496281248064009,
            -0.3265786562423951,
        ]
    )
    )


Q12: QH = QH([1, 2, 0, 0])
Q1123: QH = QH([1, 1, 2, 3])
Q11p: QH = QH([1, 1, 0, 0], representation="polar")
Q12p: QH = QH([1, 2, 0, 0], representation="polar")
Q12np: QH = QH([1, -2, 0, 0], representation="polar")
Q21p: QH = QH([2, 1, 0, 0], representation="polar")
Q23p: QH = QH([2, 3, 0, 0], representation="polar")
Q13p: QH = QH([1, 3, 0, 0], representation="polar")
Q5p: QH = QH([5, 0, 0, 0], representation="polar")


def test_txyz_2_representation():
    qr = QH(Q12.txyz_2_representation(""))
    assert qr.equals(Q12)
    qr = QH(Q12.txyz_2_representation("polar"))
    assert qr.equals(QH([2.23606797749979, 1.10714871779409, 0, 0]))
    qr = QH(Q1123.txyz_2_representation("spherical"))
    assert qr.equals(QH([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409]))


def test_representation_2_txyz():
    qr = QH(Q12.representation_2_txyz(""))
    assert qr.equals(Q12)
    qr = QH(Q12.representation_2_txyz("polar"))
    assert qr.equals(QH([-0.4161468365471424, 0.9092974268256817, 0, 0]))
    qr = QH(Q1123.representation_2_txyz("spherical"))
    assert qr.equals(
        QH(
            [
                1.0,
                -0.9001976297355174,
                0.12832006020245673,
                -0.4161468365471424,
            ]
        )
    )


def test_polar_products():
    qr = Q11p.product(Q12p)
    print("polar 1 1 0 0 * 1 2 0 0: ", qr)
    assert qr.equals(Q13p)
    qr = Q12p.product(Q21p)
    print("polar 1 2 0 0 * 2 1 0 0: ", qr)
    assert qr.equals(Q23p)


def test_polar_conj():
    qr = Q12p.conj()
    print("polar conj of 1 2 0 0: ", qr)
    assert qr.equals(Q12np)


q_0: QH = QH().q_0()
q_1 = QH().q_1()
q_i = QH().q_i()
q_n1 = QH([-1, 0, 0, 0])
q_2 = QH([2, 0, 0, 0])
q_n2 = QH([-2, 0, 0, 0])
q_3 = QH([3, 0, 0, 0])
q_n3 = QH([-3, 0, 0, 0])
q_4 = QH([4, 0, 0, 0])
q_5 = QH([5, 0, 0, 0])
q_6 = QH([6, 0, 0, 0])
q_10 = QH([10, 0, 0, 0])
q_n5 = QH([-5, 0, 0, 0])
q_7 = QH([7, 0, 0, 0])
q_8 = QH([8, 0, 0, 0])
q_9 = QH([9, 0, 0, 0])
q_n11 = QH([-11, 0, 0, 0])
q_21 = QH([21, 0, 0, 0])
q_n34 = QH([-34, 0, 0, 0])
v3: QHStates = QHStates([q_3])
v1123: QHStates = QHStates([q_1, q_1, q_2, q_3])
v3n1n21: QHStates = QHStates([q_3, q_n1, q_n2, q_1])
v9: QHStates = QHStates([q_1, q_1, q_2, q_3, q_1, q_1, q_2, q_3, q_2])
v9i: QHStates = QHStates(
    [
        QH([0, 1, 0, 0]),
        QH([0, 2, 0, 0]),
        QH([0, 3, 0, 0]),
        QH([0, 4, 0, 0]),
        QH([0, 5, 0, 0]),
        QH([0, 6, 0, 0]),
        QH([0, 7, 0, 0]),
        QH([0, 8, 0, 0]),
        QH([0, 9, 0, 0]),
    ]
)
vv9 = v9.add(v9i)
q_1d0 = QH([1.0, 0, 0, 0])
q12: QHStates = QHStates([q_1d0, q_1d0])
q14: QHStates = QHStates([q_1d0, q_1d0, q_1d0, q_1d0])
q19: QHStates = QHStates([q_1d0, q_0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0])
qn627 = QH([-6, 27, 0, 0])
v33 = QHStates([q_7, q_0, q_n3, q_2, q_3, q_4, q_1, q_n1, q_n2])
v33inv: QHStates = QHStates([q_n2, q_3, q_9, q_8, q_n11, q_n34, q_n5, q_7, q_21])
q_i3: QHStates = QHStates([q_1, q_1, q_1])
q_i2d: QHStates = QHStates([q_1, q_0, q_0, q_1])
q_i3_bra = QHStates([q_1, q_1, q_1], "bra")
q_6_op: QHStates = QHStates([q_1, q_0, q_0, q_1, q_i, q_i], "op")
q_6_op_32: QHStates = QHStates([q_1, q_0, q_0, q_1, q_i, q_i], "op", rows=3, columns=2)
q_i2d_op: QHStates = QHStates([q_1, q_0, q_0, q_1], "op")
q_i4 = QH([0, 4, 0, 0])
q_0_q_1: QHStates = QHStates([q_0, q_1])
q_1_q_0: QHStates = QHStates([q_1, q_0])
q_1_q_i: QHStates = QHStates([q_1, q_i])
q_0_q_i: QHStates = QHStates([q_0, q_i])
A: QHStates = QHStates([QH([4, 0, 0, 0]), QH([0, 1, 0, 0])], "bra")
B: QHStates = QHStates([QH([0, 0, 1, 0]), QH([0, 0, 0, 2]), QH([0, 3, 0, 0])])
Op: QHStates = QHStates(
    [
        QH([3, 0, 0, 0]),
        QH([0, 1, 0, 0]),
        QH([0, 0, 2, 0]),
        QH([0, 0, 0, 3]),
        QH([2, 0, 0, 0]),
        QH([0, 4, 0, 0]),
    ],
    "op",
    rows=2,
    columns=3,
)
Op4i = QHStates([q_i4, q_0, q_0, q_i4, q_2, q_3], "op", rows=2, columns=3)
Op_scalar = QHStates([q_i4], "scalar")
q_1234 = QHStates(
    [QH([1, 1, 0, 0]), QH([2, 1, 0, 0]), QH([3, 1, 0, 0]), QH([4, 1, 0, 0])]
)
sigma_y = QHStates(
    [QH([1, 0, 0, 0]), QH([0, -1, 0, 0]), QH([0, 1, 0, 0]), QH([-1, 0, 0, 0])]
)
qn = QHStates([QH([3, 0, 0, 4])])
# TODO test exception like so: q_bad = QHStates([q_1], rows=2, columns=3)

b = QHStates([q_1, q_2, q_3], qs_type="bra")
k = QHStates([q_4, q_5, q_6], qs_type="ket")
o = QHStates([q_10], qs_type="op")

Q = QH([1, -2, -3, -4], q_type="Q")
Q_states = QHStates([Q])
P = QH([0, 4, -3, 0], q_type="P")
P_states = QHStates([P])
R = QH([3, 0, 0, 0], q_type="R")
C = QH([2, 4, 0, 0], q_type="C")
t, x, y, z = sp.symbols("t x y z")
q_sym = QH([t, x, y, x * y * z])
qs_22 = QHStates([QH([2, 2, 0, 0])])
q44 = QHStates([QH([4, 4, 0, 0])])

q1234 = QH([1, 2, 3, 4])
q4321 = QH([4, 3, 2, 1])
q2222 = QH([2, 2, 2, 2])
qsmall = QH([0.04, 0.2, 0.1, -0.3])
q2_states = QHStates([q1234, qsmall], "ket")
qs_1234 = QHStates([q1324, q1234])
qs_1324 = QHStates([q1324, q1324])

def test_1000_init():
    assert q_0_q_1.dim == 2


def test_1010_set_qs_type():
    bk = b.set_qs_type("ket")
    assert bk.rows == 3
    assert bk.columns == 1
    assert bk.qs_type == "ket"


def test_1020_set_rows_and_columns():
    assert q_i3.rows == 3
    assert q_i3.columns == 1
    assert q_i3_bra.rows == 1
    assert q_i3_bra.columns == 3
    assert q_i2d_op.rows == 2
    assert q_i2d_op.columns == 2
    assert q_6_op_32.rows == 3
    assert q_6_op_32.columns == 2


def test_1030_equals():
    assert A.equals(A)
    assert not A.equals(B)


def test_1031_subs():
    t, x, y, z = sp.symbols("t x y z")
    q_sym = QHStates([QH([t, x, y, x * y * z])])

    q_z = q_sym.subs({t: 1, x: 2, y: 3, z: 4})
    print("t x y xyz sub 1 2 3 4: ", q_z)
    assert q_z.equals(QHStates([QH([1, 2, 3, 24])]))


def test_1032_scalar():
    qs = q_1_q_i.scalar()
    print("scalar(q_1_q_i)", qs)
    assert qs.equals(q_1_q_0)


def test_1033_vector():
    qv = q_1_q_i.vector()
    print("vector(q_1_q_i)", qv)
    assert qv.equals(q_0_q_i)


def test_1034_xyz():
    qxyz = q_1_q_i.xyz()
    print("q_1_q_i.xyz()", qxyz)
    assert qxyz[0][0] == 0
    assert qxyz[1][0] == 1


def test_1035_q_0():
    q0 = QHStates().q_0(3)
    print("q0(3): ", q0)
    assert q0.dim == 3


def test_1036_q_1():
    q1 = QHStates().q_1(2.0, 3)
    print("q1(3): ", q1)
    assert q1.dim == 3
    assert q1.qs[0].t == 2.0


def test_1037_q_i():
    qi = QHStates().q_i(2.0, 3)
    print("qi(3): ", qi)
    assert qi.dim == 3
    assert qi.qs[0].x == 2.0


def test_1038_q_j():
    qj = QHStates().q_j(2.0, 3)
    print("qj(3): ", qj)
    assert qj.dim == 3
    assert qj.qs[0].y == 2.0


def test_1039_q_k():
    qk = QHStates().q_k(2.0, 3)
    print("qk(3): ", qk)
    assert qk.dim == 3
    assert qk.qs[0].z == 2.0


def test_1039_q_random():
    qr = QHStates().q_random(-2, 2, dim=3)
    print("qk(3): ", qr)
    assert qr.dim == 3
    assert qr.qs[0].z != qr.qs[0].t


def test_1040_conj():
    qc = q_1_q_i.conj()
    qc1 = q_1_q_i.conj(1)
    print("q_1_q_i*: ", qc)
    print("q_1_qc*1: ", qc1)
    assert qc.qs[1].x == -1
    assert qc1.qs[1].x == 1


def test_1042_conj_q():
    qc = q_1_q_i.conj_q(q_1)
    qc1 = q_1_q_i.conj_q(q_1)
    print("q_1_q_i conj_q: ", qc)
    print("q_1_qc*1 conj_q: ", qc1)
    assert qc.qs[1].x == -1
    assert qc1.qs[1].x == -1


def test_1050_flip_signs():
    qf = q_1_q_i.flip_signs()
    print("-q_1_q_i: ", qf)
    assert qf.qs[1].x == -1


def test_1060_inverse():
    inv_v1123 = v1123.inverse()
    print("inv_v1123 operator", inv_v1123)
    vvinv = inv_v1123.product(v1123)
    vvinv.print_state("vinvD x v")
    assert vvinv.equals(q14)

    inv_v33 = v33.inverse()
    print("inv_v33 operator", inv_v33)
    vv33 = inv_v33.product(v33)
    vv33.print_state("inv_v33D x v33")
    assert vv33.equals(q19)

    Ainv = A.inverse()
    print("A ket inverse, ", Ainv)
    AAinv = A.product(Ainv)
    AAinv.print_state("A x AinvD")
    assert AAinv.equals(q12)


def test_1070_normalize():
    qn_test = qn.normalize()
    print("Op normalized: ", qn_test)
    assert math.isclose(qn_test.qs[0].t, 0.6)
    assert qn_test.qs[0].z == 0.8


def test_1080_determinant():
    det_v3 = v3.determinant()
    print("det v3:", det_v3)
    assert det_v3.equals(q_3)
    det_v1123 = v1123.determinant()
    print("det v1123", det_v1123)
    assert det_v1123.equals(q_1)
    det_v9 = v9.determinant()
    print("det_v9", det_v9)
    assert det_v9.equals(q_9)
    det_vv9 = vv9.determinant()
    print("det_vv9", det_vv9)
    assert det_vv9.equals(qn627)


def test_1090_summation():
    q_01_sum = q_0_q_1.summation()
    print("sum: ", q_01_sum)
    assert type(q_01_sum) is QH
    assert q_01_sum.t == 1


def test_1100_add():
    q_0110_add = q_0_q_1.add(q_1_q_0)
    print("add 01 10: ", q_0110_add)
    assert q_0110_add.qs[0].t == 1
    assert q_0110_add.qs[1].t == 1


def test_1110_dif():
    q_0110_dif = q_0_q_1.dif(q_1_q_0)
    print("dif 01 10: ", q_0110_dif)
    assert q_0110_dif.qs[0].t == -1
    assert q_0110_dif.qs[1].t == 1


def test_1120_diagonal():
    Op4iDiag2 = Op_scalar.diagonal(2)
    print("Op4i on a diagonal 2x2", Op4iDiag2)
    assert Op4iDiag2.qs[0].equals(q_i4)
    assert Op4iDiag2.qs[1].equals(QH().q_0())


def test_1125_trace():
    trace = v1123.op(2, 2).trace()
    print("trace: ", trace)
    assert trace.equals(QHStates([q_4]))


def test_1130_identity():
    I2 = QHStates().identity(2, operator=True)
    print("Operator Idenity, diagonal 2x2", I2)
    assert I2.qs[0].equals(QH().q_1())
    assert I2.qs[1].equals(QH().q_0())
    I2 = QHStates().identity(2)
    print("Idenity on 2 state ket", I2)
    assert I2.qs[0].equals(QH().q_1())
    assert I2.qs[1].equals(QH().q_1())


def test_1140_product():
    assert b.product(o).equals(QHStates([QH([10, 0, 0, 0]), QH([20, 0, 0, 0]), QH([30, 0, 0, 0])]))
    assert b.product(k).equals(QHStates([QH([32, 0, 0, 0])]))
    assert b.product(o).product(k).equals(QHStates([QH([320, 0, 0, 0])]))
    assert b.product(b).equals(QHStates([QH([1, 0, 0, 0]), QH([4, 0, 0, 0]), QH([9, 0, 0, 0])]))
    assert o.product(k).equals(QHStates([QH([40, 0, 0, 0]), QH([50, 0, 0, 0]), QH([60, 0, 0, 0])]))
    assert o.product(o).equals(QHStates([QH([100, 0, 0, 0])]))
    assert k.product(k).equals(QHStates([QH([16, 0, 0, 0]), QH([25, 0, 0, 0]), QH([36, 0, 0, 0])]))
    assert k.product(b).equals(QHStates(
        [
            QH([4, 0, 0, 0]),
            QH([5, 0, 0, 0]),
            QH([6, 0, 0, 0]),
            QH([8, 0, 0, 0]),
            QH([10, 0, 0, 0]),
            QH([12, 0, 0, 0]),
            QH([12, 0, 0, 0]),
            QH([15, 0, 0, 0]),
            QH([18, 0, 0, 0]),
        ]
    )
    )


def test_1150_product_AA():
    Aket = deepcopy(A).ket()
    AA = A.product(Aket)
    print("<A|A>: ", AA)
    assert AA.equals(QHStates([QH([17, 0, 0, 0])]))


def test_1170_product_AOp():
    AOp: QHStates = A.product(Op)
    print("A Op: ", AOp)
    assert AOp.qs[0].equals(QH([11, 0, 0, 0]))
    assert AOp.qs[1].equals(QH([0, 0, 5, 0]))
    assert AOp.qs[2].equals(QH([4, 0, 0, 0]))


def test_1190_product_AOp4i():
    AOp4i: QHStates = A.product(Op4i)
    print("A Op4i: ", AOp4i)
    assert AOp4i.qs[0].equals(QH([0, 16, 0, 0]))
    assert AOp4i.qs[1].equals(QH([-4, 0, 0, 0]))


def test_1210_product_OpB():
    OpB: QHStates = Op.product(B)
    print("Op B: ", OpB)
    assert OpB.qs[0].equals(QH([0, 10, 3, 0]))
    assert OpB.qs[1].equals(QH([-18, 0, 0, 1]))


def test_1230_product_AOpB():
    AOpB = A.product(Op).product(B)
    print("A Op B: ", AOpB)
    assert AOpB.equals(QHStates([QH([0, 22, 11, 0])]))


def test_1250_product_AOp4i():
    AOp4i = A.product(Op4i)
    print("A Op4i: ", AOp4i)
    assert AOp4i.qs[0].equals(QH([0, 16, 0, 0]))
    assert AOp4i.qs[1].equals(QH([-4, 0, 0, 0]))


def test_1270_product_Op4iB():
    Op4iB = Op4i.product(B)
    print("Op4i B: ", Op4iB)
    assert Op4iB.qs[0].equals(QH([0, 6, 0, 4]))
    assert Op4iB.qs[1].equals(QH([0, 9, -8, 0]))


def test_1290_product_AOp4iB():
    AOp4iB = A.product(Op4i).product(B)
    print("A* Op4i B: ", AOp4iB)
    assert AOp4iB.equals(QHStates([QH([-9, 24, 0, 8])]))


def test_1430_inverse():
    q_z = P_states.inverse()
    print("inverse: ", q_z)
    assert q_z.equals(QHStates([QH([0, -0.16, 0.12, 0])]))


def test_1301_divide_by():
    q_z = Q_states.divide_by(Q_states)
    print("divide_by: ", q_z)
    assert q_z.equals(QHStates([q_1]))


def test_1302_triple_product():
    q_z = Q_states.triple_product(P_states, Q_states)
    print("triple product: ", q_z)
    assert q_z.equals(QHStates([QH([-2, 124, -84, 8])]))


def test_1303_rotate():
    q_z = Q_states.rotate(QHStates([q_i]))
    print("rotate: ", q_z)
    assert q_z.equals(QHStates([QH([1, -2, 3, 4])]))


def test_1304_rotation_and_or_boost():
    q1_sq = Q_states.square()
    beta = 0.003
    gamma = 1 / math.sqrt(1 - beta ** 2)
    h = QHStates([QH([gamma, gamma * beta, 0, 0])])
    q_z = Q_states.rotation_and_or_boost(h)
    q_z2 = q_z.square()
    print("q1_sq: ", q1_sq)
    print("boosted: ", q_z)
    print("boosted squared: ", q_z2)
    assert round(q_z2.qs[0].t, 5) == round(q1_sq.qs[0].t, 5)


def test_1305_Lorentz_next_rotation():
    with pytest.raises(ValueError):
        QHStates.Lorentz_next_rotation(qs_1234, q2_states)
    next_rot: QHStates = QHStates.Lorentz_next_rotation(qs_1234, qs_1324)
    print("next_rotation: ", next_rot)
    assert math.isclose(next_rot.qs[0].t, 0)
    assert math.isclose(next_rot.qs[1].t, 0)
    assert math.isclose(next_rot.norm_squared().qs[0].t, 2)
    assert not next_rot.qs[0].equals(next_rot.qs[1])


def test_1305_Lorentz_next_boost():
    with pytest.raises(ValueError):
        QHStates.Lorentz_next_boost(qs_1234, q2_states)
    next_boost: QHStates = QHStates.Lorentz_next_boost(qs_1234, qs_1324)
    print("next_boost: ", next_boost)
    assert next_boost.qs[0].t != 0
    assert next_boost.qs[1].t != 0
    assert next_boost.norm_squared().qs[0].t != 1
    assert not next_boost.qs[0].equals(next_boost.qs[1])
    boosted_square = q2_states.rotation_and_or_boost(next_boost).square()
    q2_states_square = q2_states.square()
    assert math.isclose(q2_states_square.qs[0].t, boosted_square.qs[0].t)


def test_1306_g_shift():
    qs1_sq = Q_states.square()
    qs_z = Q_states.g_shift(0.003)
    qs_z2 = qs_z.square()
    qs_z_minimal = Q_states.g_shift(0.003, g_form="minimal")
    qs_z2_minimal = qs_z_minimal.square()
    print("q1_sq: ", qs1_sq)
    print("g_shift: ", qs_z)
    print("g squared: ", qs_z2)
    assert qs_z2.qs[0].t != qs1_sq.qs[0].t
    assert qs_z2.qs[0].x == qs1_sq.qs[0].x
    assert qs_z2.qs[0].y == qs1_sq.qs[0].y
    assert qs_z2.qs[0].z == qs1_sq.qs[0].z
    assert qs_z2_minimal.qs[0].t != qs1_sq.qs[0].t
    assert qs_z2_minimal.qs[0].x == qs1_sq.qs[0].x
    assert qs_z2_minimal.qs[0].y == qs1_sq.qs[0].y
    assert qs_z2_minimal.qs[0].z == qs1_sq.qs[0].z


def test_1305_bracket():
    bracket1234 = QHStates().bracket(
        q_1234, QHStates().identity(4, operator=True), q_1234
    )
    print("bracket <1234|I|1234>: ", bracket1234)
    assert bracket1234.equals(QHStates([QH([34, 0, 0, 0])]))


def test_1310_op_q():
    opn = Op.op_q(q=q_i)
    print("op_q: ", opn)
    assert opn.qs[0].x == 3


def test_1312_square():
    ns = q_1_q_i.square()
    ns.print_state("q_1_q_i square")
    assert ns.equals(QHStates([q_1, q_n1]))


def test_1315_norm_squared():
    ns = q_1_q_i.norm_squared()
    ns.print_state("q_1_q_i norm squared")
    assert ns.equals(QHStates([QH([2, 0, 0, 0])]))


def test_1318_norm_squared_of_vector():
    ns = q_1_q_i.norm_squared_of_vector()
    ns.print_state("q_1_q_i norm squared of vector")
    assert ns.equals(QHStates([q_1]))


def test_1320_transpose():
    opt = q_1234.transpose()
    print("op1234 transposed: ", opt)
    assert opt.qs[0].t == 1
    assert opt.qs[1].t == 3
    assert opt.qs[2].t == 2
    assert opt.qs[3].t == 4
    optt = q_1234.transpose().transpose()
    assert optt.equals(q_1234)


def test_1330_Hermitian_conj():
    q_hc = q_1234.Hermitian_conj()
    print("op1234 Hermtian_conj: ", q_hc)
    assert q_hc.qs[0].t == 1
    assert q_hc.qs[1].t == 3
    assert q_hc.qs[2].t == 2
    assert q_hc.qs[3].t == 4
    assert q_hc.qs[0].x == -1
    assert q_hc.qs[1].x == -1
    assert q_hc.qs[2].x == -1
    assert q_hc.qs[3].x == -1


def test_1340_is_Hermitian():
    assert sigma_y.is_Hermitian()
    assert not q_1234.is_Hermitian()


def test_1350_is_square():
    assert not Op.is_square()
    assert Op_scalar.is_square()
