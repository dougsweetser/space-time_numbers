#!/usr/bin/env python
# coding: utf-8

# # Q8 Designed for nparray

# This notebook is designed to provide tools for doing basic math with a new type of number I call **space-time numbers**. They are quite similar to quaternions which are commonly defined over 4 real numbers (but a pair of complex numbers can also do the job). Instead, 8 numbers are used: one for the past, the future, left, right, up, down, near and far.
# 
# To be honest, this is a pain. Why bother? The motivation for this project is theoretically rooted in physics. 

# In[1]:


import math
import numpy as np
import random
import sympy as sp
import os
import unittest
from copy import deepcopy

from os.path import basename
from glob import glob


# Define the stretch factor $\gamma$ and the $\gamma \beta$ used in special relativity.

# In[2]:


def sr_gamma(beta_x=0, beta_y=0, beta_z=0):
    """The gamma used in special relativity using 3 velocites, some may be zero."""

    return 1 / (1 - beta_x ** 2 - beta_y ** 2 - beta_z ** 2) ** (1/2)

def sr_gamma_betas(beta_x=0, beta_y=0, beta_z=0):
    """gamma and the three gamma * betas used in special relativity."""

    g = sr_gamma(beta_x, beta_y, beta_z)
    
    return [g, g * beta_x, g * beta_y, g * beta_z]


# In[3]:


class Q8a(np.ndarray):
    """Quaternions on a quaternion manifold or space-time numbers."""

    def __new__(subtype, values=None, qtype="Q", representation="", 
                shape=8, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        
        obj = super(Q8a, subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)
        #obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)
        obj.values = values
        obj.qtype = qtype
        obj.representation = representation
        
        if values is not None:        
            if len(values) == 4:
                for i in range(0, 4):
                    obj[i * 2], obj[i * 2 + 1] = obj.__handle_negatives(values[i], 0)
            
            elif len(values) == 8:
                for i in range(0,8,2):
                    obj[i], obj[i+1] = obj.__handle_negatives(values[i], values[i+1])
                    
            else:
                print("Oops, need to be fed 4 or 8 numbers.")
        else:
            # Puts in zeroes by default.
            pass
        
        obj[obj < 10e-100] = 0
        
        return obj
    
    def __handle_negatives(self, a, b):
        """Figure out which value is negative"""
        
        if a >= 0 and b >= 0:
            return [a, b]
        
        else:
            c = a + b
            
            if c < 0:
                return [0, -1 * c]
            else:
                return [c, 0]
    
    def __array_finalize__(self, obj):
        
        if obj is None: return
        
        self.values = getattr(obj, 'values', None)
        self.qtype = getattr(obj, 'qtype', 'Q')
        self.representation = getattr(obj, 'representation', "") 
        
    def __str__(self, quiet=False):
        """Customize the output."""
        
        qtype = self.qtype
        
        if quiet:
            qtype = ""
        
        if self.representation == "":
            string = "(({tp}, {tn}), ({xp}, {xn}), ({yp}, {yn}), ({zp}, {zn})) {qt}".format(
                                                             tp=self[0], tn=self[1], 
                                                             xp=self[2], xn=self[3], 
                                                             yp=self[4], yn=self[5], 
                                                             zp=self[6], zn=self[7],
                                                             qt=qtype)
    
        return string
            
    def print_state(self, label, spacer=False, quiet=False):
        """Utility for printing a quaternion."""

        print(label)
        
        print(self.__str__(quiet))
        
        if spacer:
            print("")

    def check_representations(self, q1):
        """If they are the same, report true. If not, kick out an exception. Don't add apples to oranges."""

        if self.representation == q1.representation:
            return True
        
        else:
            raise Exception("Oops, 2 quaternions have different representations: {}, {}".format(self.representation, q1.representation))
            return False
            
    ### Static ways to create Q8a's.
    
    @staticmethod
    def q_0(qtype="0", representation=""):
        """Return a zero quaternion."""
        
        q0 = Q8a(values=np.array([0, 0, 0, 0]), qtype=qtype, representation=representation)
        
        return q0
      
    @staticmethod
    def q_1(n=1, qtype="1", representation=""):
        """Return a multiplicative identity quaternion. Set n=x to get other values."""
        
        q1 = Q8a(values=np.array([n, 0, 0, 0]), qtype=qtype, representation=representation)
        
        return q1
    
    @staticmethod
    def q_i(n=1, qtype="i", representation=""):
        """Return i."""
        
        qi = Q8a(values=np.array([0, n, 0, 0]), qtype=qtype, representation=representation)
        
        return qi
    
    @staticmethod
    def q_j(n=1, qtype="j", representation=""):
        """Return j."""
        
        qj = Q8a(values=np.array([0, 0, n, 0]), qtype=qtype, representation=representation)
        
        return qj
    
    @staticmethod
    def q_k(n=1, qtype="k", representation=""):
        """Return k."""
        
        qk = Q8a(values=np.array([0, 0, 0, n]), qtype=qtype, representation=representation)
        
        return qk

    @staticmethod
    def q_random(distribution="uniform", qtype="?", representation=""):
        """Return a random-valued quaternion. Can add more distributions if needed.
           exponential, laplace, logistic, lognormal, normal, poisson, uniform (default)."""

        distributions = {}
        distributions["exponential"] = np.random.exponential
        distributions["laplace"] = np.random.laplace
        distributions["logistic"] = np.random.logistic
        distributions["lognormal"] = np.random.lognormal
        distributions["normal"] = np.random.normal
        distributions["poisson"] = np.random.poisson
        distributions["uniform"] = np.random.uniform
        
        numbers = distributions[distribution](size=8).tolist()
        qr = Q8a(numbers, qtype=qtype)
        qr.representation = representation
        
        return qr
    
    def reduce(self, qtype="-reduce"):
        """Put all Doubletas into the reduced form so one of each pair is zero."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        tp, tn, xp, xn, yp, yn, zp, zn = self.tolist()
        
        t, x, y, z = tp - tn, xp - xn, yp - yn, zp - zn
        
        if t < 0:
            t0, t1 = 0, -1 * t
        else:
            t0, t1 = t, 0
            
        if x < 0:
            x0, x1 = 0, -1 * x 
        else:
            x0, x1 = x, 0
            
        if y < 0:
            y0, y1 = 0, -1 * y
        else:
            y0, y1 = y, 0
            
        if z < 0:
            z0, z1 = 0, -1 * z
        else:
            z0, z1 = z, 0
                
        q_red = Q8a([t0, t1, x0, x1, y0, y1, z0, z1], qtype=self.qtype)
        
        q_red.qtype = end_qtype
        q_red.representation = self.representation
        
        return q_red
    
    def equals(self, q2):
        """Tests if two quaternions are equal."""
        
        self_red = self.reduce()
        q2_red = q2.reduce()
        result = True
        
        for i in range(8):
            if not math.isclose(self_red[i], q2_red[i]):
                result = False
        
        return result
    
    ### Parts.
    
    def scalar(self, qtype="scalar"):
        """Returns the scalar part of a quaternion."""
        
        end_qtype = "scalar({})".format(self.qtype)
        
        s = Q8a([self[0], self[1], 0, 0, 0, 0, 0, 0], qtype=end_qtype, representation=self.representation)
        return s
    
    def vector(self, qtype="v"):
        """Returns the vector part of a quaternion."""
        
        end_qtype = "vector({})".format(self.qtype)
        
        v = Q8a([0, 0, self[2], self[3], self[4], self[5], self[6], self[7]], qtype=end_qtype, representation=self.representation)
        return v
    
    def t(self):
        """Returns a real-value as an np.array."""
        
        return np.array([self[0] - self[1]])
    
    def xyz(self):
        """Returns a real-valued 3-vector as an np.array."""
        
        return np.array([self[2] - self[3], self[4] - self[5], self[6] - self[7]])
    
    def txyz(self):
        """Returns a real-valued 4-vector as an np.array."""
        
        return np.array([self[0] - self[1], self[2] - self[3], self[4] - self[5], self[6] - self[7]])
    
    
    ### Conjugation.
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""
        
        tp, tn, xp, xn, yp, yn, zp, zn = self.tolist()
        
        # Flip all but t.                          
        if conj_type == 0:
            conj_q = Q8a([tp, tn, xn, xp, yn, yp, zn, zp])
        
        # Flip all but x.
        if conj_type == 1:
            conj_q = Q8a([tn, tp, xp, xn, yn, yp, zn, zp])
            qtype += "1"

        # Flip all but y.                                 
        if conj_type == 2:
            conj_q = Q8a([tn, tp, xn, xp, yp, yn, zn, zp])
            qtype += "2"
            
        # conj_q.qtype = self.qtype + qtype
        # conj_q.representation = self.representation
        
        return conj_q

    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""

        tp, tn, xp, xn, yp, yn, zp, zn = self.tolist()
        
        conj_q = Q8a()

        if conj_type == "-":
            
            conj_q = Q8a([tp, tn, xn, xp, yn, yp, zn, zp])
            qtype += "-"

        # Flip the sign of x and y.
        if conj_type == "'":
            
            conj_q = Q8a([tp, tn, xn, xp, yn, yp, zp, zn])
            qtype += "'"
            
        # Flip the sign of only z.
        if conj_type == "*":
            
            
            conj_q = Q8a([tp, tn, xp, xn, yp, yn, zn, zp])
            qtype += "*"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
        
        return conj_q

    def conj_q(self, q1):
        """Given a quaternion with 0's or 1's, will do the standard conjugate, first conjugate
           second conjugate, sign flip, or all combinations of the above."""
        
        _conj = deepcopy(self)
    
        if q1[0] or q1[1]:
            _conj = _conj.conj(conj_type=0)
            
        if q1[2] or q1[3]:
            _conj = _conj.conj(conj_type=1)    
        
        if q1[4] or q1[5]:
            _conj = _conj.conj(conj_type=2)    
        
        if q1[6] or q1[7]:
            _conj = _conj.flip_signs()
    
        return _conj
    
    def flip_signs(self, conj_type=0, qtype="-"):
        """Flip all the signs, just like multipying by -1."""

        end_qtype = "-{}".format(self.qtype)
        
        tp, tn, xp, xn, yp, yn, zp, zn = self.tolist()
        
        flip_q = Q8a([tn, tp, xn, xp, yn, yp, zn, zp], qtype=end_qtype)
        flip_q.qtype = end_qtype
        flip_q.representation = self.representation
        
        return flip_q
    
    ### Adding.
    
    def add(self, q1, qtype="+"):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        add_q = Q8a()
        
        for i in range(0, 8):
            add_q[i] = self[i] + q1[i]
                    
        add_q.qtype = "{f}+{s}".format(f=self.qtype, s=q1.qtype)
        add_q.representation = self.representation    
        
        return add_q    

    def dif(self, q1, qtype="-"):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        dif_q = Q8a()

        for i in range(0, 8, 2):
            dif_q[i]   = self[i] + q1[i+1]
            dif_q[i+1] = self[i+1] + q1[i]
        
        dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=q1.qtype)
        dif_q.representation = self.representation
        
        return dif_q
    
    ### Multiplication.
    
    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        products = {'tt0': self[0] * q1[0] + self[1] * q1[1],
                    'tt1': self[0] * q1[1] + self[1] * q1[0],
                    
                    'xx+yy+zz0': self[2] * q1[2] + self[3] * q1[3] + self[4] * q1[4] + self[5] * q1[5] + self[6] * q1[6] + self[7] * q1[7], 
                    'xx+yy+zz1': self[2] * q1[3] + self[3] * q1[2] + self[4] * q1[5] + self[5] * q1[4] + self[6] * q1[7] + self[7] * q1[6], 
                    
                    'tx+xt0': self[0] * q1[2] + self[1] * q1[3] + self[2] * q1[0] + self[3] * q1[1],
                    'tx+xt1': self[0] * q1[3] + self[1] * q1[2] + self[3] * q1[0] + self[2] * q1[1],
                    
                    'ty+yt0': self[0] * q1[4] + self[1] * q1[5] + self[4] * q1[0] + self[5] * q1[1],
                    'ty+yt1': self[0] * q1[5] + self[1] * q1[4] + self[5] * q1[0] + self[4] * q1[1],
                    
                    'tz+zt0': self[0] * q1[6] + self[1] * q1[7] + self[6] * q1[0] + self[7] * q1[1],
                    'tz+zt1': self[0] * q1[7] + self[1] * q1[6] + self[7] * q1[0] + self[6] * q1[1]
                    }
        
        return products
    
    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        yz0 = self[4] * q1[6] + self[5] * q1[7]
        yz1 = self[4] * q1[7] + self[5] * q1[6]
        zy0 = self[6] * q1[4] + self[7] * q1[5]
        zy1 = self[6] * q1[5] + self[7] * q1[4]

        zx0 = self[6] * q1[2] + self[7] * q1[3]
        zx1 = self[6] * q1[3] + self[7] * q1[2]
        xz0 = self[2] * q1[6] + self[3] * q1[7]
        xz1 = self[2] * q1[7] + self[3] * q1[6]

        xy0 = self[2] * q1[4] + self[3] * q1[5]
        xy1 = self[2] * q1[5] + self[3] * q1[4]
        yx0 = self[4] * q1[2] + self[5] * q1[3]
        yx1 = self[4] * q1[3] + self[5] * q1[2]
                                   
        products = {'yz-zy0': yz0 + zy1,
                    'yz-zy1': yz1 + zy0,
                    
                    'zx-xz0': zx0 + xz1,
                    'zx-xz1': zx1 + xz0,
                    
                    'xy-yx0': xy0 + yx1,
                    'xy-yx1': xy1 + yx0,
                   
                    'zy-yz0': yz1 + zy0,
                    'zy-yz1': yz0 + zy1,
                    
                    'xz-zx0': zx1 + xz0,
                    'xz-zx1': zx0 + xz1,
                    
                    'yx-xy0': xy1 + yx0,
                    'yx-xy1': xy0 + yx1
                   }
        
        return products
    
    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))
        
        return products
    
    def square(self, qtype="^2"):
        """Square a quaternion."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        qxq = self._commuting_products(self)
        
        sq_q = Q8a(qtype=self.qtype)        
        sq_q[0] = qxq['tt0'] + (qxq['xx+yy+zz1'])
        sq_q[1] = qxq['tt1'] + (qxq['xx+yy+zz0'])
        sq_q[2] = qxq['tx+xt0']
        sq_q[3] = qxq['tx+xt1']
        sq_q[4] = qxq['ty+yt0']
        sq_q[5] = qxq['ty+yt1']
        sq_q[6] = qxq['tz+zt0']
        sq_q[7] = qxq['tz+zt1']
        
        sq_q.qtype = end_qtype
        sq_q.representation = self.representation
        
        return sq_q
    
    def norm_squared(self, qtype="|| ||^2"):
        """The norm_squared of a quaternion."""
        
        end_qtype = "||{}||^2".format(self.qtype)
        
        qxq = self._commuting_products(self)
        
        n_q = Q8a().q_0()
        n_q[0] = qxq['tt0'] + qxq['xx+yy+zz0']
        n_q[1] = qxq['tt1'] + qxq['xx+yy+zz1']
        n_q = n_q.reduce()
        
        n_q.qtype = end_qtype
        n_q.representation = self.representation
        
        return n_q
    
    def norm_squared_of_vector(self, qtype="V(|| ||)^2"):
        """The norm_squared of the vector of a quaternion."""
        
        end_qtype = "V||({})||^2".format(self.qtype)
        
        qxq = self._commuting_products(self)
        
        nv_q = Q8a().q_0()
        nv_q[0] = qxq['xx+yy+zz0']
        nv_q[1] = qxq['xx+yy+zz1']
        result = nv_q.reduce()
        result.qtype = end_qtype
        result.representation = self.representation

        return result
        
    def abs_of_q(self, qtype="| |"):
        """The absolute value, the square root of the norm_squared."""

        end_qtype = "|{}|".format(self.qtype)
        
        abq = self.norm_squared()
        sqrt_t0 = abq[0] ** (1/2)
        abq[0] = sqrt_t0
        abq.qtype = end_qtype
        abq.representation = self.representation
        
        return abq

    def abs_of_vector(self, qtype="|V()|)"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        end_qtype = "|V({})|".format(self.qtype, qtype)
        
        av = self.norm_squared_of_vector()
        sqrt_t = av[0] ** (1/2)
        av[0] = sqrt_t
        av.qtype = end_qtype
        av.representation = self.representation
        
        return av
    
    def normalize(self, n=1, qtype="U"):
        """Normalize a quaternion"""
        
        end_qtype = "{}U".format(self.qtype)
        
        abs_q_inv = self.abs_of_q().inverse()
        n_q = self.product(abs_q_inv).product(Q8a([n, 0, 0, 0]))
        n_q.qtype = end_qtype
        n_q.representation=self.representation
        
        return n_q
    
    def product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product given 2 quaternions."""

        self.check_representations(q1)
        
        commuting = self._commuting_products(q1)
        q_even = Q8a().q_0()
        q_even[0] = commuting['tt0'] + commuting['xx+yy+zz1']
        q_even[1] = commuting['tt1'] + commuting['xx+yy+zz0']
        q_even[2] = commuting['tx+xt0']
        q_even[3] = commuting['tx+xt1']
        q_even[4] = commuting['ty+yt0']
        q_even[5] = commuting['ty+yt1']
        q_even[6] = commuting['tz+zt0']
        q_even[7] = commuting['tz+zt1']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = Q8a().q_0()
        
        if reverse:
            q_odd[2] = anti_commuting['zy-yz0']
            q_odd[3] = anti_commuting['zy-yz1']
            q_odd[4] = anti_commuting['xz-zx0']
            q_odd[5] = anti_commuting['xz-zx1']
            q_odd[6] = anti_commuting['yx-xy0']
            q_odd[7] = anti_commuting['yx-xy1']
            
        else:
            q_odd[2] = anti_commuting['yz-zy0']
            q_odd[3] = anti_commuting['yz-zy1']
            q_odd[4] = anti_commuting['zx-xz0']
            q_odd[5] = anti_commuting['zx-xz1']
            q_odd[6] = anti_commuting['xy-yx0']
            q_odd[7] = anti_commuting['xy-yx1']
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        else:
            raise Exception("Three 'kind' values are known: '', 'even', and 'odd'")
            
        if reverse:
            times_symbol = times_symbol.replace('x', 'xR')    
            
        result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
        result.representation = self.representation
        
        return result

    def Euclidean_product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        self.check_representations(q1)
        
        pq = Q8a().q_0()
        pq = self.conj().product(q1, kind, reverse, qtype)
        pq.representation = self.representation
        
        return pq

    def inverse(self, qtype="^-1", additive=False):
        """Inverse a quaternion."""
        
        if additive:
            end_qtype = "-{}".format(self.qtype)
            q_inv = self.flip_signs()
            q_inv.qtype = end_qtype
            
        else:
            end_qtype = "{}{}".format(self.qtype, qtype)
        
            q_conj = self.conj()
            q_norm_squared = self.norm_squared().reduce()
        
            if q_norm_squared[0] == 0:
                return self.q_0()
        
            q_norm_squared_inv = Q8a([1.0 / q_norm_squared[0], 0, 0, 0, 0, 0, 0, 0])

            q_inv = q_conj.product(q_norm_squared_inv)
        
        q_inv.qtype = end_qtype
        q_inv.representation = self.representation
        
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""

        self.check_representations(q1)
        
        q_inv = q1.inverse()
        q_div = self.product(q_inv) 
        q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
        q_div.representation = self.representation    
        
        return q_div
    
    ### Triple products.
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        self.check_representations(q1)
        self.check_representations(q2)
        
        triple = self.product(q1).product(q2)
        
        return triple
    
    # Quaternion rotation involves a triple product:  u R 1/u
    def rotate(self, u):
        """Do a rotation using a triple product."""
    
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)
        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        q_rot.representation = self.representation
        
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    # b -> b' = h b h* + 1/2 ((hhb)* -(h*h*b)*)
    # where h is of the form (cosh(a), sinh(a)) OR (0, a, b, c)
    def boost(self, h, qtype="boost"):
        """A boost along the x, y, and/or z axis."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        boost = h
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Q8a([0.5, 0, 0, 0, 0, 0, 0, 0]))
        triple_123 = triple_1.add(half_23)
        
        triple_123.qtype = end_qtype
        triple_123.representation = self.representation
        
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
            if qtype == "g_shift":
                qtype = "g_exp"
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
            if qtype == "g_shift":
                qtype = "g_minimal"
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self
        exp_g = sp.exp(dimensionless_g)
        
        tp, tn = self[0] / exp_g, self[1] / exp_g
        xp, xn = self[2] * exp_g, self[3] * exp_g
        yp, yn = self[4] * exp_g, self[5] * exp_g
        zp, zn = self[6] * exp_g, self[7] * exp_g
        
        g_q = Q8a([tp, tn, xp, xn, yp, yn, zp, zn], qtype=self.qtype)
        
        g_q.qtype = end_qtype
        g_q.representation = self.representation
        
        return g_q
    
    def exp(self, qtype="exp"):
        """Take the exponential of a quaternion."""
        # exp(q) = (exp(t) cos(|R|), exp(t) sin(|R|) R/|R|)
        
        end_qtype = "exp({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]
        
        if red_t < 0:
            et = math.exp(-1 * red_t)
            
            if (abs_v[0] == 0):
                return Q8a([et, 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
            cosR = math.cos(abs_v[0])
            sinR = math.sin(abs_v[0])
    
        else:
            et = math.exp(red_t)
            
            if (abs_v[0] == 0):
                return Q8a([et, 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
            cosR = math.cos(abs_v[0])
            sinR = math.sin(abs_v[0])
            
        k = et * sinR / abs_v[0]
                       
        expq_dt = et * cosR
        expq_dx = k * (self[2] - self[3])
        expq_dy = k * (self[4] - self[5])
        expq_dz = k * (self[6] - self[7])
        expq = Q8a([expq_dt, expq_dx, expq_dy, expq_dz], qtype=end_qtype, representation=self.representation)
                       
        return expq
    
    def ln(self, qtype="ln"):
        """Take the natural log of a quaternion."""
        # ln(q) = (0.5 ln t^2 + R.R, atan2(|R|, t) R/|R|)
        
        end_qtype = "ln({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]
        
        if red_t < 0:
            if (abs_v[0] == 0):
                # I don't understant this, but mathematica does the same thing, but it looks wrong to me.
                return(Q8a([math.log(-self.dt.n), math.pi, 0, 0], qtype=end_qtype))   
            
            t_value = 0.5 * math.log(red_t * red_t + abs_v[0] * abs_v[0])
            k = math.atan2(abs_v[0], red_t) / abs_v[0]
        
        else:
            if (abs_v[0] == 0):
                return(Q8a([math.log(red_t), 0, 0, 0], qtype=end_qtype, representation=self.representation))
                
            t_value = 0.5 * math.log(red_t * red_t + abs_v[0] * abs_v[0])
            k = math.atan2(abs_v[0], red_t) / abs_v[0]
            
        lnq_dt = t_value
        lnq_dx = k * (self[2] - self[3])
        lnq_dy = k * (self[4] - self[5])
        lnq_dz = k * (self[6] - self[7])
        lnq = Q8a([lnq_dt, lnq_dx, lnq_dy, lnq_dz], qtype=end_qtype, representation=self.representation)
                       
        return lnq
    
    def q_2_q(self, q1, qtype="P"):
        """Take the natural log of a quaternion, q^p = exp(ln(q) * p)."""
        
        self.check_representations(q1)
        
        end_qtype = "{st}^P".format(st=self.qtype)
        lnq = self.ln()
        print("lnq: ", lnq)
        lnxp = lnq.product(q1).reduce()
        print("lnq x p reduced:", lnxp)
        q2q = lnxp.exp()
        print("q_2_q: ", q2q)
        q2q.qtype = end_qtype
        q2q.representation = self.representation
        
        return q2q
    
    def sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""
        
        end_qtype = "sin({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]

        if abs_v[0] == 0:    
            return Q8a([math.sin(red_t), 0, 0, 0], qtype=end_qtype, representation=self.representation)
        
        sint = math.sin(red_t)
        cost = math.cos(red_t)    
            
        sinhR = math.sinh(abs_v[0])
        coshR = math.cosh(abs_v[0])
        
        k = cost * sinhR / abs_v[0]
            
        q_out_dt = sint * coshR
        q_out_dx = k * (self[2] - self[3])
        q_out_dy = k * (self[4] - self[5])
        q_out_dz = k * (self[6] - self[7])
        q_out = Q8a([q_out_dt, q_out_dx, q_out_dy, q_out_dz], qtype=end_qtype, representation=self.representation)
        
        return q_out
     
    def cos(self, qtype="cos"):
        """Take the cosine of a quaternion, (cos(t) cosh(|R|), - sin(t) sinh(|R|) R/|R|)"""

        end_qtype = "cos({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]
       
        if abs_v[0] == 0:    
            return Q8a([math.cos(red_t), 0, 0, 0], qtype=end_qtype)
        
        sint = math.sin(red_t)
        cost = math.cos(red_t) 
            
        sinhR = math.sinh(abs_v[0])
        coshR = math.cosh(abs_v[0])
        
        k = -1 * sint * sinhR / abs_v[0]
            
        q_out_dt = cost * coshR
        q_out_dx = k * (self[2] - self[3])
        q_out_dy = k * (self[4] - self[5])
        q_out_dz = k * (self[6] - self[7])
        q_out = Q8a([q_out_dt, q_out_dx, q_out_dy, q_out_dz], qtype=end_qtype, representation=self.representation)

        return q_out

    def tan(self, qtype="sin"):
        """Take the tan of a quaternion, sin/cos"""

        end_qtype = "tan({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()        
        red_t = self[0] - self[1]
        
        if abs_v[0] == 0:    
            return Q8a([math.tan(red_t), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
        sinq = self.sin()
        cosq = self.cos()
            
        q_out = sinq.divide_by(cosq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation

        return q_out
    
    def sinh(self, qtype="sinh"):
        """Take the sinh of a quaternion, (sinh(t) cos(|R|), cosh(t) sin(|R|) R/|R|)"""
        
        end_qtype = "sinh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]
        
        if abs_v[0] == 0:    
            return Q8a([math.sinh(red_t), 0, 0, 0], qtype=end_qtype, representation=self.representation)
        
        sinht = math.sinh(red_t)
        cosht = math.cosh(red_t)
            
        sinR = math.sin(abs_v[0])
        cosR = math.cos(abs_v[0])
        
        k = cosht * sinR / abs_v[0]
            
        q_out_dt = sinht * cosR
        q_out_dx = k * (self[2] - self[3])
        q_out_dy = k * (self[4] - self[5])
        q_out_dz = k * (self[6] - self[7])
        q_out = Q8a([q_out_dt, q_out_dx, q_out_dy, q_out_dz], qtype=end_qtype, representation=self.representation)

        return q_out
     
    def cosh(self, qtype="sin"):
        """Take the cosh of a quaternion, (cosh(t) cos(|R|), sinh(t) sin(|R|) R/|R|)"""

        end_qtype = "cosh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]
        
        if abs_v[0] == 0:    
            return Q8a([math.cosh(red_t), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
        sinht = math.sinh(red_t)
        cosht = math.cosh(red_t)
             
        sinR = math.sin(abs_v[0])
        cosR = math.cos(abs_v[0])
        
        k = sinht * sinR / abs_v[0]
            
        q_out_dt = cosht * cosR
        q_out_dx = k * (self[2] - self[3])
        q_out_dy = k * (self[4] - self[5])
        q_out_dz = k * (self[6] - self[7])
        q_out = Q8a([q_out_dt, q_out_dx, q_out_dy, q_out_dz], qtype=end_qtype, representation=self.representation)

        return q_out
    
    def tanh(self, qtype="sin"):
        """Take the tanh of a quaternion, sin/cos"""
        
        end_qtype = "tanh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self[0] - self[1]
        
        if abs_v[0] == 0:
            return Q8a([math.tanh(red_t), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
        sinhq = self.sinh()
        coshq = self.cosh()
            
        q_out = sinhq.divide_by(coshq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out

    def trunc(self):
        """Truncates values."""
        
        for i in range(8):
            self[i] = math.trunc(self[i])
        
        return self
    
    def ops(self, q2=None, q3=None, op="add", dim=10):
        """Apply an operator to all terms in a quaternion series max=n times."""
    
        results = []
        last_q = self
        
        for i in range(dim):
        
            if op in Q8a.unary_op:
                new_q = Q8a.unary_op[op](last_q)
                last_q = new_q
                
            elif op in Q8a.unary_with_option_op:
                if op == "conj_1":
                    new_q = Q8a.unary_with_option_op[op](last_q, conj_type=1)
                elif op == "conj_2":
                    new_q = Q8a.unary_with_option_op[op](last_q, conj_type=2)
                elif op == "conj_prime":
                    new_q = Q8a.unary_with_option_op[op](last_q, conj_type="'")
                elif op == "conj_star":
                    new_q = Q8a.unary_with_option_op[op](last_q, conj_type="*")
                
                last_q = new_q
        
            elif op in Q8a.binary_op:
                new_q = Q8a.binary_op[op](last_q, q2)
                last_q = new_q

            elif op in Q8a.trinary_op:
                new_q = Q8a.trinary_op[op](last_q, q2, q3)
                last_q = new_q

            else:
                print("Oops, don't know that operator.")
                return
        
            yield new_q
            
    # All the operators.
    unary_op = {}
    unary_op["reduce"] = reduce
    unary_op["scalar"] = scalar
    unary_op["vector"] = vector
    unary_op["t"] = t
    unary_op["xyz"] = xyz
    unary_op["txyz"] = txyz
    unary_op["conj"] = conj
    unary_op["vahlen_conj"] = vahlen_conj
    unary_op["flip_signs"] = flip_signs
    unary_op["square"] = square
    unary_op["norm_squared"] = norm_squared
    unary_op["norm_squared_of_vector"] = norm_squared_of_vector
    unary_op["abs_of_q"] = abs_of_q
    unary_op["abs_of_vector"] = abs_of_vector
    unary_op["normalize"] = normalize
    unary_op["inverse"] = inverse
    unary_op["exp"] = exp
    unary_op["ln"] = ln
    unary_op["sin"] = sin
    unary_op["cos"] = cos
    unary_op["tan"] = tan
    unary_op["sinh"] = sinh
    unary_op["cosh"] = cosh
    unary_op["tanh"] = tanh

    unary_with_option_op = {}
    unary_with_option_op["conj_1"] = conj
    unary_with_option_op["conj_2"] = conj
    unary_with_option_op["vahlen_conj_prime"] = vahlen_conj
    unary_with_option_op["vahlen_conj_star"] = vahlen_conj

    binary_op = {}
    binary_op["equals"] = equals
    binary_op["add"] = add
    binary_op["dif"] = dif
    binary_op["product"] = product
    binary_op["Euclidean_product"] = Euclidean_product
    binary_op["divide_by"] = divide_by
    binary_op["q_2_q"] = q_2_q

    trinary_op = {}
    trinary_op["triple_product"] = triple_product


# In[4]:


class TestQ8a(unittest.TestCase):

    """Class to make sure all the functions work as expected."""
    q_0 = Q8a.q_0()
    q_1 = Q8a.q_1()
    q1 = Q8a([1, 0, 0, 2, 0, 3, 0, 4])
    q2 = Q8a([0, 0, 4, 0, 0, 3, 0, 0])
    q_big = Q8a([1, 2, 3, 4, 5, 6, 7, 8])
    Q = Q8a([1, -2, -3, -4], qtype="Q")
    P = Q8a([0, 4, -3, 0], qtype="P")
    R = Q8a([3, 0, 0, 0], qtype="R")
    C = Q8a([2, 4, 0, 0], qtype="C")
    verbose = True
    
    def test_qt(self):
        self.assertTrue(self.q1[0] == 1)
    
    def test_scalar(self):
        q_z = self.q1.scalar()
        print("scalar(q): ", q_z)
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[6] == 0)
        
    def test_vector(self):
        q_z = self.q1.vector()
        print("vector(q): ", q_z)
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[3] == 2)
        self.assertTrue(q_z[5] == 3)
        self.assertTrue(q_z[7] == 4)
        
    def test_t(self):
        q_z = self.q1.t()
        print("q.t()): ", q_z)
        self.assertTrue(q_z[0] == 1)
    
    def test_xyz(self):
        q_z = self.q1.xyz()
        print("q.xyz()): ", q_z)
        self.assertTrue(q_z[0] == -2)
        self.assertTrue(q_z[1] == -3)
        self.assertTrue(q_z[2] == -4)
    
    def test_txyz(self):
        q_z = self.q1.txyz()
        print("q.txyz()): ", q_z)
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[1] == -2)
        self.assertTrue(q_z[2] == -3)
        self.assertTrue(q_z[3] == -4)
        
    def test_q_zero(self):
        q_z = self.q1.q_0()
        print("q0: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[6] == 0)
        
    def test_q_1(self):
        q_z = self.q1.q_1()
        q_zn = self.q1.q_1(-1)
        print("q_1: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_zn[1] == 1)
        
    def test_q_i(self):
        q_z = self.q1.q_i()
        q_zn = self.q1.q_i(-1)
        print("q_i: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[2] == 1)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_zn[3] == 1)
        
    def test_q_j(self):
        q_z = self.q1.q_j()
        q_zn = self.q1.q_j(-1)
        print("q_j: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[4] == 1)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_zn[5] == 1)
        
    def test_q_k(self):
        q_z = self.q1.q_k()
        q_zn = self.q1.q_k(-1)
        print("q_k: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[6] == 1)
        self.assertTrue(q_zn[7] == 1)
                
    def test_q_random(self):
        q_z = self.q1.q_random()
        print("q_random():", q_z)
        for i in range(8):
            self.assertTrue(q_z[i] >= 0 and q_z[i] <= 1)
            
    def test_conj_0(self):
        q_z = self.q1.conj()
        print("conj 0: {}".format(q_z))
        self.assertTrue(1)
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[2] == 2)
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[6] == 4)
            
    def test_equals(self):
        self.assertTrue(self.q1.equals(self.q1))
        self.assertFalse(self.q1.equals(self.q2))
                 
    def test_conj_1(self):
        q_z = self.q1.conj(1)
        print("conj 1: {}".format(q_z))
        self.assertTrue(q_z[1] == 1)
        self.assertTrue(q_z[3] == 2)
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[6] == 4)
                 
    def test_conj_2(self):
        q_z = self.q1.conj(2)
        print("conj 2: {}".format(q_z))
        self.assertTrue(q_z[1] == 1)
        self.assertTrue(q_z[2] == 2)
        self.assertTrue(q_z[5] == 3)
        self.assertTrue(q_z[6] == 4)
        
    def test_vahlen_conj_0(self):
        q1 = Q8a(values=[1, 0, 0, 2, 0, 3, 0, 4])
        q_z = q1.vahlen_conj()
        print("vahlen conj -: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[2] == 2)
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[6] == 4)
                 
    def test_vahlen_conj_1(self):
        q_z = self.q1.vahlen_conj("'")
        print("vahlen conj ': {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[2] == 2)
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[7] == 4)
                 
    def test_vahlen_conj_2(self):
        q_z = self.q1.vahlen_conj('*')
        print("vahlen conj *: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[3] == 2)
        self.assertTrue(q_z[5] == 3)
        self.assertTrue(q_z[6] == 4)
        
    def test_conj_q(self):
        q_z = self.q1.conj_q(self.q1)
        print("conj_q(conj_q): ", q_z)
        self.assertTrue(q_z[1] == 1)
        self.assertTrue(q_z[2] == 2)
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[7] == 4)
        
    def test_reduce(self):
        q_z = self.q_big.reduce()
        print("q_big reduced: {}".format(q_z))
        for i in range(0, 8, 2):
            self.assertTrue(q_z[i] == 0)
            self.assertTrue(q_z[i + 1] == 1)
            
    def test_flip_signs(self):
        q_z = self.q_big.flip_signs()
        print("q_big sign_flip: {}".format(q_z))
        for i in range(0, 8, 2):
            self.assertTrue(q_z[i] == self.q_big[i+1])
            self.assertTrue(q_z[i+1] == self.q_big[i])
            
    def test_add(self):
        q_z = self.q1.add(self.q2)
        print("add: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 4)
        self.assertTrue(q_z[3] == 2)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 6)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 4)
        
    def test_add_reduce(self):
        q_z_red = self.q1.add(self.q2).reduce()
        print("add reduce: {}".format(q_z_red))
        self.assertTrue(q_z_red[0] == 1)
        self.assertTrue(q_z_red[1] == 0)
        self.assertTrue(q_z_red[2] == 2)
        self.assertTrue(q_z_red[3] == 0)
        self.assertTrue(q_z_red[4] == 0)
        self.assertTrue(q_z_red[5] == 6)
        self.assertTrue(q_z_red[6] == 0)
        self.assertTrue(q_z_red[7] == 4)
        
    def test_dif(self):
        q_z = self.q1.dif(self.q2)
        print("dif: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 6) 
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[5] == 3)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 4) 
        
    def test_square(self):
        q_sq = self.q1.square()
        q_sq_red = q_sq.reduce()
        print("square: {}".format(q_sq))
        print("square reduced: {}".format(q_sq_red))
        self.assertTrue(q_sq[0] == 1)
        self.assertTrue(q_sq[1] == 29)
        self.assertTrue(q_sq[3] == 4)
        self.assertTrue(q_sq[5] == 6)
        self.assertTrue(q_sq[7] == 8)
        self.assertTrue(q_sq_red[0] == 0)
        self.assertTrue(q_sq_red[1] == 28)
        
    def test_norm_squared(self):
        q_z = self.q1.norm_squared()
        print("norm_squared: {}".format(q_z))
        self.assertTrue(q_z[0] == 30)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 0)
        
    def test_norm_squared_of_vector(self):
        q_z = self.q1.norm_squared_of_vector()
        print("norm_squared_of_vector: {}".format(q_z))
        self.assertTrue(q_z[0] == 29)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 0)
        
    def test_abs_of_q(self):
        q_z = self.q2.abs_of_q()
        print("abs_of_q: {}".format(q_z))
        self.assertTrue(q_z[0] == 5)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[7] == 0)
        
    def test_abs_of_vector(self):
        q_z = self.q2.abs_of_vector()
        print("abs_of_vector: {}".format(q_z))
        self.assertTrue(q_z[0] == 5)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[7] == 0)
        
    def test_normalize(self):
        q_z = self.q2.normalize()
        print("q_normalized: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0.8)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertAlmostEqual(q_z[5], 0.6)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 0)
    
    def test_product(self):
        q_z = self.q1.product(self.q2).reduce()
        print("product: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[1] == 1)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 8)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 19)
        self.assertTrue(q_z[6] == 18)
        self.assertTrue(q_z[7] == 0)
        
    def test_product_even(self):
        q_z = self.q1.product(self.q2, kind="even").reduce()
        print("product, kind even: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[1] == 1)
        self.assertTrue(q_z[2] == 4)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 3)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 0)
        
    def test_product_odd(self):
        q_z = self.q1.product(self.q2, kind="odd").reduce()
        print("product, kind odd: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 12)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 16)
        self.assertTrue(q_z[6] == 18)
        self.assertTrue(q_z[7] == 0)
        
    def test_product_reverse(self):
        q1q2_rev = self.q1.product(self.q2, reverse=True)
        q2q1 = self.q2.product(self.q1)
        self.assertTrue(q1q2_rev.equals(q2q1))
        
    def test_Euclidean_product(self):
        q_z = self.q1.Euclidean_product(self.q2).reduce()
        print("Euclidean product: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 16)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[4] == 13)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 18)
        
    def test_inverse(self):
        q_z = self.q2.inverse().reduce()
        print("inverse: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 0.16)
        self.assertTrue(q_z[4] == 0.12)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[6] == 0)
        self.assertTrue(q_z[7] == 0)

    def test_triple_product(self):
        q_z = self.q1.triple_product(self.q2, self.q1).reduce()
        print("triple: {}".format(q_z))
        self.assertTrue(q_z[0] == 0)
        self.assertTrue(q_z[1] == 2)
        self.assertTrue(q_z[2] == 124)
        self.assertTrue(q_z[3] == 0)
        self.assertTrue(q_z[4] == 0)
        self.assertTrue(q_z[5] == 84)
        self.assertTrue(q_z[6] == 8)
        self.assertTrue(q_z[7] == 0)

    def test_rotate(self):
        q_z = self.q1.rotate(Q8a([0, 1, 0, 0])).reduce()
        print("rotate: {}".format(q_z))
        self.assertTrue(q_z[0] == 1)
        self.assertTrue(q_z[1] == 0)
        self.assertTrue(q_z[2] == 0)
        self.assertTrue(q_z[3] == 2)
        self.assertTrue(q_z[4] == 3)
        self.assertTrue(q_z[5] == 0)
        self.assertTrue(q_z[6] == 4)
        self.assertTrue(q_z[7] == 0)
        
    def test_boost(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.boost(Q8a(sr_gamma_betas(0.003)))
        q_z2 = q_z.square().reduce()
        print("q1_sq: {}".format(q1_sq))
        print("boosted: {}".format(q_z))
        print("b squared: {}".format(q_z2))
        self.assertTrue(round(q_z2[1], 12) == round(q1_sq[1], 12))
        
    def test_g_shift(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.g_shift(0.003)
        q_z2 = q_z.square().reduce()
        print("q1_sq: {}".format(q1_sq))
        print("g_shift: {}".format(q_z))
        print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2[1] != q1_sq[1])
        self.assertTrue(q_z2[2] == q1_sq[2])
        self.assertTrue(q_z2[3] == q1_sq[3])
        self.assertTrue(q_z2[4] == q1_sq[4])
        self.assertTrue(q_z2[5] == q1_sq[5])
        self.assertTrue(q_z2[6] == q1_sq[6])
        self.assertTrue(q_z2[7] == q1_sq[7])
        
    def test_exp(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).exp().equals(Q8a().q_1()))
        Q_z = self.Q.exp()
        P_z = self.P.exp()
        R_z = self.R.exp()
        C_z = self.C.exp()
        print("exp(Q): ", Q_z)
        print("exp(P): ", P_z)
        print("exp(R): ", R_z)
        print("exp(C): ", C_z)
        self.assertTrue(Q_z.equals(Q8a([1.6939227236832994, 0.7895596245415588, 1.1843394368123383, 1.5791192490831176])))
        self.assertTrue(P_z.equals(Q8a([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(R_z.equals(Q8a([20.0855369231876679, 0, 0, 0])))
        self.assertTrue(C_z.equals(Q8a([-4.8298093832693851, -5.5920560936409816, 0, 0])))
    
    def test_ln(self):
        Q_z = self.Q.ln()
        P_z = self.P.ln()
        R_z = self.R.ln()
        C_z = self.C.ln()
        print("ln(Q): ", Q_z)
        print("ln(P): ", P_z)
        print("ln(R): ", R_z)
        print("ln(C): ", C_z)
        self.assertTrue(Q_z.exp().equals(self.Q))
        self.assertTrue(Q_z.equals(Q8a([1.7005986908310777, -0.5151902926640850, -0.7727854389961275, -1.0303805853281700])))
        self.assertTrue(P_z.equals(Q8a([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0])))
        self.assertTrue(R_z.equals(Q8a([1.0986122886681098, 0, 0, 0])))
        self.assertTrue(C_z.equals(Q8a([1.4978661367769954, 1.1071487177940904, 0, 0])))    
    
    @unittest.skip("Will have to investigate this one more.")
    def test_q_2_p(self):
        Q2P = self.Q.q_2_q(self.P)
        print("Q^P: ", Q2P)
        self.assertTrue(Q2P.equals(Q8a([-0.0197219653530713, -0.2613955437374326, 0.6496281248064009, -0.3265786562423951])))

    def test_sin(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).sin().reduce().equals(Q8a().q_0()))
        self.assertTrue(self.Q.sin().reduce().equals(Q8a([91.7837157840346691, -21.8864868530291758, -32.8297302795437673, -43.7729737060583517])))
        self.assertTrue(self.P.sin().reduce().equals(Q8a([0,  59.3625684622310033, -44.5219263466732542, 0])))
        self.assertTrue(self.R.sin().reduce().equals(Q8a([0.1411200080598672, 0, 0, 0])))
        self.assertTrue(self.C.sin().reduce().equals(Q8a([24.8313058489463785, -11.3566127112181743, 0, 0])))
     
    def test_cos(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).cos().equals(Q8a().q_1()))
        self.assertTrue(self.Q.cos().equals(Q8a([58.9336461679439481, 34.0861836904655959, 51.1292755356983974, 68.1723673809311919])))
        self.assertTrue(self.P.cos().equals(Q8a([74.2099485247878476, 0, 0, 0])))
        self.assertTrue(self.R.cos().equals(Q8a([-0.9899924966004454, 0, 0, 0])))
        self.assertTrue(self.C.cos().equals(Q8a([-11.3642347064010600, -24.8146514856341867, 0, 0])))
                            
    def test_tan(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).tan().equals(Q8a().q_0()))
        self.assertTrue(self.Q.tan().equals(Q8a([0.0000382163172501, -0.3713971716439372, -0.5570957574659058, -0.7427943432878743])))
        self.assertTrue(self.P.tan().equals(Q8a([0, 0.7999273634100760, -0.5999455225575570, 0])))
        self.assertTrue(self.R.tan().equals(Q8a([-0.1425465430742778, 0, 0, 0])))
        self.assertTrue(self.C.tan().equals(Q8a([-0.0005079806234700, 1.0004385132020521, 0, 0])))
        
    def test_sinh(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).sinh().equals(Q8a().q_0()))
        self.assertTrue(self.Q.sinh().equals(Q8a([0.7323376060463428, 0.4482074499805421, 0.6723111749708131, 0.8964148999610841])))
        self.assertTrue(self.P.sinh().equals(Q8a([0, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.sinh().equals(Q8a([10.0178749274099026, 0, 0, 0])))
        self.assertTrue(self.C.sinh().equals(Q8a([-2.3706741693520015, -2.8472390868488278, 0, 0])))    
        
    def test_cosh(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).cosh().equals(Q8a().q_1()))
        self.assertTrue(self.Q.cosh().equals(Q8a([0.9615851176369565, 0.3413521745610167, 0.5120282618415251, 0.6827043491220334])))
        self.assertTrue(self.P.cosh().equals(Q8a([0.2836621854632263, 0, 0, 0])))
        self.assertTrue(self.R.cosh().equals(Q8a([10.0676619957777653, 0, 0, 0])))
        self.assertTrue(self.C.cosh().equals(Q8a([-2.4591352139173837, -2.7448170067921538, 0, 0])))    
        
    def test_tanh(self):
        self.assertTrue(Q8a([0, 0, 0, 0]).tanh().equals(Q8a().q_0()))
        self.assertTrue(self.Q.tanh().equals(Q8a([1.0248695360556623, 0.1022956817887642, 0.1534435226831462, 0.2045913635775283])))
        self.assertTrue(self.P.tanh().equals(Q8a([0, -2.7044120049972684, 2.0283090037479505, 0])))
        self.assertTrue(self.R.tanh().equals(Q8a([0.9950547536867305, 0, 0, 0])))
        self.assertTrue(self.C.tanh().equals(Q8a([1.0046823121902353, 0.0364233692474038, 0, 0])))    
    
    def test_ops(self):
        qs = []
        for q in self.q_0.ops(q2=self.q_1, op="dif", dim=3):
            qs.append(q)
        print("ops dif -1, dim=3", qs[-1])
        self.assertTrue(qs[-1].equals(Q8a().q_1(-3)))

suite = unittest.TestLoader().loadTestsFromModule(TestQ8a())
unittest.TextTestRunner().run(suite);


# In[5]:


# class Q8aStates(Q8a):
class Q8aStates(object):
    """A class made up of many quaternions."""
    
    QS_TYPES = ["scalar", "bra", "ket", "op", "operator"]
    
    def __init__(self, qs=None, qs_type="ket", rows=0, columns=0):
        
        self.qs = qs
        self.array = np.array(qs)
        self.qs_type = qs_type
        self.rows = rows
        self.columns = columns
        
        if qs_type not in self.QS_TYPES:
            print("Oops, only know of these quaternion series types: {}".format(self.QS_TYPES))
            return None
        
        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = int(len(qs)), int(len(qs)), int(len(qs))
    
        self.set_qs_type(qs_type, rows, columns, copy=False)
 
        if self.dim > 0:
            self.array = self.array.reshape(self.rows, self.columns, 8)
    
    def set_qs_type(self, qs_type="", rows=0, columns=0, copy=True):
        """Set the qs_type to something sensible."""
    
        # Checks.
        if (rows) and (columns) and rows * columns != self.dim:
            print("Oops, check those values again for rows:{} columns:{} dim:{}".format(
                rows, columns, self.dim))
            self.qs, self.rows, self.columns = None, 0, 0
            return None
        
        new_q = self
        
        if copy:
            new_q = deepcopy(self)
        
        # Assign values if need be.
        if new_q.qs_type != qs_type:
            new_q.rows = 0
        
        if qs_type == "ket" and not new_q.rows:
            new_q.rows = new_q.dim
            new_q.columns = 1
            
        elif qs_type == "bra" and not new_q.rows:
            new_q.rows = 1
            new_q.columns = new_q.dim

        elif qs_type in ["op", "operator"] and not new_q.rows:
            # Square series
            root_dim = math.sqrt(new_q.dim)
            
            if root_dim.is_integer():
                new_q.rows = int(root_dim)
                new_q.columns = int(root_dim)
                qs_type = "op"
        
        elif rows * columns == new_q.dim and not new_q.qs_type:
            if new_q.dim == 1:
                qs_type = "scalar"
            elif new_q.rows == 1:
                qs_type = "bra"
            elif new_q.columns == 1:
                qs_type = "ket"
            else:
                qs_type = "op"
            
        if not qs_type:
            print("Oops, please set rows and columns for this quaternion series operator. Thanks.")
            return None
        
        if new_q.dim == 1:
            qs_type = "scalar"
            
        new_q.qs_type = qs_type
        
        #if self.dim > 0:
        #    self.array.reshape(self.rows, self.columns, 8)
        
        return new_q
        
    def bra(self):
        """Quickly set the qs_type to bra by calling set_qs_type()."""
        
        return self.set_qs_type("bra")
    
    def ket(self):
        """Quickly set the qs_type to ket by calling set_qs_type()."""
        
        return self.set_qs_type("ket")
    
    def op(self, rows=0, columns=0):
        """Quickly set the qs_type to op by calling set_qs_type()."""
        
        return self.set_qs_type("op", rows=rows, columns=columns)
    
    def __str__(self, quiet=False):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q.__str__(quiet))
        
        return states.rstrip()
    
    def print_state(self, label, spacer=True, quiet=False, sum=False):
        """Utility for printing states as a quaternion series."""

        print(label)
        
        for n, q in enumerate(self.qs):
            print("n={}: {}".format(n + 1, q.__str__(quiet)))
            
        if sum:
            print("sum= {ss}".format(ss=self.summation()))
            
        print("{t}: {r}/{c}".format(t=self.qs_type, r=self.rows, c=self.columns))
        
        if spacer:
            print("")

    def equals(self, q1):
        """Test if two states are equal."""
   
        if self.dim != q1.dim:
            return False
        
        result = True
    
        for selfq, q1q in zip(self.qs, q1.qs):
            if not selfq.equals(q1q):
                result = False
                
        return result

    def scalar(self, qtype="scalar"):
        """Returns the scalar part of a quaternion."""
    
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.scalar())
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def vector(self, qtype="v"):
        """Returns the vector part of a quaternion."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.vector())
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
      
    def t(self):
        """Returns a real-valued t as an np.array."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.t())
            
        return new_states
    
    def xyz(self):
        """Returns a real-valued 3-vector as an np.array."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.xyz())
            
        return new_states
    
    def txyz(self):
        """Returns a real-valued 4-vector as an np.array."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.txyz())
            
        return new_states
    
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def conj_q(self, q1):
        """Takes multiple conjugates of states, depending on true/false value of q1 parameter."""
        
        new_states = []
        
        for ket in self.qs:
            new_states.append(ket.conj_q(q1))
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def simple_q(self):
        """Simplify the states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.simple_q())
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def flip_signs(self):
        """Flip signs of all states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.flip_signs())
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def inverse(self, additive=False):
        """Inverseing bras and kets calls inverse() once for each.
        Inverseing operators is more tricky as one needs a diagonal identity matrix."""
    
        if self.qs_type in ["op", "operator"]:
        
            if additive:
                q_flip = self.inverse(additive=True)
                q_inv = q_flip.diagonal(self.dim)
                
            else:
                if self.dim == 1:
                    q_inv = Q8aStates(self.qs[0].inverse())
        
                elif self.qs_type in ["bra", "ket"]:
                    new_qs = []
                    
                    for q in self.qs:
                        new_qs.append(q.inverse())
                    
                    q_inv = Q8aStates(new_qs, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
        
                elif self.dim == 4:
                    det = self.determinant()
                    detinv = det.inverse()

                    q0 = self.qs[3].product(detinv)
                    q1 = self.qs[1].flip_signs().product(detinv)
                    q2 = self.qs[2].flip_signs().product(detinv)
                    q3 = self.qs[0].product(detinv)

                    q_inv =Q8aStates([q0, q1, q2, q3], qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
                elif self.dim == 9:
                    det = self.determinant()
                    detinv = det.inverse()
        
                    q0 = self.qs[4].product(self.qs[8]).dif(self.qs[5].product(self.qs[7])).product(detinv)
                    q1 = self.qs[7].product(self.qs[2]).dif(self.qs[8].product(self.qs[1])).product(detinv)
                    q2 = self.qs[1].product(self.qs[5]).dif(self.qs[2].product(self.qs[4])).product(detinv)
                    q3 = self.qs[6].product(self.qs[5]).dif(self.qs[8].product(self.qs[3])).product(detinv)
                    q4 = self.qs[0].product(self.qs[8]).dif(self.qs[2].product(self.qs[6])).product(detinv)
                    q5 = self.qs[3].product(self.qs[2]).dif(self.qs[5].product(self.qs[0])).product(detinv)
                    q6 = self.qs[3].product(self.qs[7]).dif(self.qs[4].product(self.qs[6])).product(detinv)
                    q7 = self.qs[6].product(self.qs[1]).dif(self.qs[7].product(self.qs[0])).product(detinv)
                    q8 = self.qs[0].product(self.qs[4]).dif(self.qs[1].product(self.qs[3])).product(detinv)
        
                    q_inv =Q8aStates([q0, q1, q2, q3, q4, q5, q6, q7, q8], qs_type=self.qs_type, rows=self.rows, columns=self.columns)
        
                else:
                    print("Oops, don't know how to inverse.")
                    q_inv =Q8aStates([Q8a().q_0()])
        
        else:                
            new_states = []
        
            for bra in self.qs:
                new_states.append(bra.inverse(additive=additive))
        
            q_inv =Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
        return q_inv
    
    def norm(self):
        """Norm of states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.norm())
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def normalize(self, n=1, states=None):
        """Normalize all states."""
        
        new_states = []
        
        zero_norm_count = 0
        
        for bra in self.qs:
            if bra.norm_squared()[0] == 0:
                zero_norm_count += 1
                new_states.append(Q8a().q_0())
            else:
                new_states.append(bra.normalize(n))
        
        new_states_normalized = []
        
        non_zero_states = self.dim - zero_norm_count
        
        for new_state in new_states:
            new_states_normalized.append(new_state.product(Q8a([math.sqrt(1/non_zero_states), 0, 0, 0])))
            
        return Q8aStates(new_states_normalized, qs_type=self.qs_type, rows=self.rows, columns=self.columns)

    def orthonormalize(self):
        """Given a quaternion series, resturn a normalized orthoganl basis."""
    
        last_q = self.qs.pop(0).normalize(math.sqrt(1/self.dim))
        orthonormal_qs = [last_q]
    
        for q in self.qs:
            qp = q.Euclidean_product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1/self.dim))
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q
        
        return Q8aStates(orthonormal_qs, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def determinant(self):
        """Calculate the determinant of a 'square' quaternion series."""
    
        if self.dim == 1:
            q_det = self.qs[0]
        
        elif self.dim == 4:
            ad =self.qs[0].product(self.qs[3])
            bc = self.qs[1].product(self.qs[2])
            q_det = ad.dif(bc)  
        
        elif self.dim == 9:
            aei = self.qs[0].product(self.qs[4].product(self.qs[8]))
            bfg = self.qs[3].product(self.qs[7].product(self.qs[2]))
            cdh = self.qs[6].product(self.qs[1].product(self.qs[5]))
            ceg = self.qs[6].product(self.qs[4].product(self.qs[2]))
            bdi = self.qs[3].product(self.qs[1].product(self.qs[8]))
            afh = self.qs[0].product(self.qs[7].product(self.qs[5]))
        
            sum_pos = aei.add(bfg.add(cdh))
            sum_neg = ceg.add(bdi.add(afh))
        
            q_det = sum_pos.dif(sum_neg)
        
        else:
            print("Oops, don't know how to calculate the determinant of this one.")
            return None
        
        return q_det
    
    def add(self, ket):
        """Add two states."""
        
        if ((self.rows != ket.rows) or (self.columns != ket.columns)):
            print("Oops, can only add if rows and columns are the same.")
            print("rows are: {}/{}, columns are: {}/{}".format(self.rows, ket.rows,
                                                               self.columns, ket.columns))
            return None
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)

    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = None
    
        for q in self.qs:
            if result is None:
                result = q
            else:
                result = result.add(q)
            
        return result    
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns))  
    
    def reduce(self):
        """Reduce the doublet values so one is zero."""
        
        new_states = []
        
        for ket in self.qs:
            new_states.append(ket.reduce())
            
        return(Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns))  
            
    def diagonal(self, dim):
        """Make a state dim*dim with q or qs along the 'diagonal'. Always returns an operator."""
        
        diagonal = []
        
        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        elif self.qs is None:
            print("Oops, the qs here is None.")
            return None
        else:
            print("Oops, need the length to be equal to the dimensions.")
            return None
        
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))
                else:
                    diagonal.append(Q8a().q_0())
        
        return Q8aStates(diagonal, qs_type="op", rows=dim, columns=dim)
        
    @staticmethod    
    def identity(dim, additive=False, non_zeroes=None, qs_type="ket"):
        """Identity operator for states or operators which are diagonal."""
    
        if qs_type == "ket":
            rows, columns = dim, 1
            
        elif qs_type == "bra":
            rows, columns = 1, dim
            
        else:
            rows, columns = dim, 1
    
        if additive:
            id_q = [Q8a().q_0() for i in range(dim)]
           
        elif non_zeroes is not None:
            
            id_q = []
            
            if len(non_zeroes) != dim:
                print("Oops, len(non_zeroes)={nz}, should be: {d}".format(nz=len(non_zeroes), d=dim))
                return Q8aStates([Q8a().q_0()], qs_type=qs_type, rows=rows, columns=columns)
            
            else:
                for non_zero in non_zeroes:
                    if non_zero:
                        id_q.append(Q8a().q_1())
                    else:
                        id_q.append(Q8a().q_0())
            
        else:
            id_q = [Q8a().q_1() for i in range(dim)]
                    
        if qs_type in ["op", "operator", "scalar"]:
            q_1 = Q8aStates(id_q, qs_type=qs_type, rows=rows, columns=columns)
            ident = Q8aStates.diagonal(q_1, dim)    
    
        else:
            ident = Q8aStates(id_q, qs_type=qs_type, rows=rows, columns=columns)
            
        return ident
    
    def product(self, q1, kind="", reverse=False):
        """Forms the quaternion product for each state."""
        
        self_copy = deepcopy(self)
        q1_copy = deepcopy(q1)
        
        # Diagonalize if need be.
        if ((self.rows == q1.rows) and (self.columns == q1.columns)) or             ("scalar" in [self.qs_type, q1.qs_type]):
                
            if self.columns == 1:
                qs_right = q1_copy
                qs_left = self_copy.diagonal(qs_right.rows)
      
            elif q1.rows == 1:
                qs_left = self_copy
                qs_right = q1_copy.diagonal(qs_left.columns)

            else:
                qs_left = self_copy
                qs_right = q1_copy
        
        # Typical matrix multiplication criteria.
        elif self.columns == q1.rows:
            qs_left = self_copy
            qs_right = q1_copy
        
        else:
            print("Oops, cannot multiply series with row/column dimensions of {}/{} to {}/{}".format(
                self.rows, self.columns, q1.rows, q1.columns))            
            return None 
        
        outer_row_max = qs_left.rows
        outer_column_max = qs_right.columns
        shared_inner_max = qs_left.columns
        projector_flag = (shared_inner_max == 1) and (outer_row_max > 1) and (outer_column_max > 1)
        
        result = [[Q8a().q_0(qtype='') for i in range(outer_column_max)] for j in range(outer_row_max)]
        
        for outer_row in range(outer_row_max):
            for outer_column in range(outer_column_max):
                for shared_inner in range(shared_inner_max):
                    
                    # For projection operators.
                    left_index = outer_row
                    right_index = outer_column
                    
                    if outer_row_max >= 1 and shared_inner_max > 1:
                        left_index = outer_row + shared_inner * outer_row_max
                        
                    if outer_column_max >= 1 and shared_inner_max > 1:
                        right_index = shared_inner + outer_column * shared_inner_max
                            
                    result[outer_row][outer_column] = result[outer_row][outer_column].add(
                        qs_left.qs[left_index].product(
                            qs_right.qs[right_index], kind=kind, reverse=reverse))
        
        # Flatten the list.
        new_qs = [item for sublist in result for item in sublist]
        new_states = Q8aStates(new_qs, rows=outer_row_max, columns=outer_column_max)

        if projector_flag:
            return new_states.transpose()
        
        else:
            return new_states
    
    def Euclidean_product(self, q1, kind="", reverse=False):
        """Forms the Euclidean product, what is used in QM all the time."""
                    
        return self.conj().product(q1, kind, reverse)
    
    def op_n(self, n, first=True, kind="", reverse=False):
        """Mulitply an operator times a number, in that order. Set first=false for n * Op"""
    
        new_states = []
    
        for op in self.qs:
        
            if first:
                new_states.append(op.product(n, kind, reverse))
                              
            else:
                new_states.append(n.product(op, kind, reverse))
    
        return Q8aStates(new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns)
    
    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning a scalar series."""
        
        return self.set_qs_type("bra").Euclidean_product(self.set_qs_type("ket"))
    
    def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m
               
        if n is None:
            n = int(self.dim / m)
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = self.qs[mi * n + ni]
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        # Switch rows and columns.
        return Q8aStates(qs_t, rows=self.columns, columns=self.rows)
        
    def Hermitian_conj(self, m=None, n=None, conj_type=0):
        """Returns the Hermitian conjugate."""
        
        return self.transpose(m, n).conj(conj_type)
    
    def dagger(self, m=None, n=None, conj_type=0):
        """Just calls Hermitian_conj()"""
        
        return self.Hermitian_conj(m, n, conj_type)
        
    def is_square(self):
        """Tests if a quaternion series is square, meaning the dimenion is n^2."""
                
        return math.sqrt(self.dim).is_integer()

    def is_Hermitian(self):
        """Tests if a series is Hermitian."""
        
        hc = self.Hermitian_conj()
        
        return self.equals(hc)
    
    @staticmethod
    def sigma(kind, theta=None, phi=None):
        """Returns a sigma when given a type like, x, y, z, xy, xz, yz, xyz, with optional angles theta and phi."""
        
        q0, q1, qi =Q8a().q_0(),Q8a().q_1(),Q8a().q_i()
        
        # Should work if given angles or not.
        if theta is None:
            sin_theta = 1
            cos_theta = 1
        else:
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
        if phi is None:
            sin_phi = 1
            cos_phi = 1
        else:
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
        x_factor = q1.product(Q8a([sin_theta * cos_phi, 0, 0, 0]))
        y_factor = qi.product(Q8a([sin_theta * sin_phi, 0, 0, 0]))
        z_factor = q1.product(Q8a([cos_theta, 0, 0, 0]))

        sigmas = {}
        sigma['x'] = Q8aStates([q0, x_factor, x_factor, q0], "op")
        sigma['y'] = Q8aStates([q0, y_factor, y_factor.flip_signs(), q0], "op") 
        sigma['z'] = Q8aStates([z_factor, q0, q0, z_factor.flip_signs()], "op")
  
        sigmas['xy'] = sigma['x'].add(sigma['y'])
        sigmas['xz'] = sigma['x'].add(sigma['z'])
        sigmas['yz'] = sigma['y'].add(sigma['z'])
        sigmas['xyz'] = sigma['x'].add(sigma['y']).add(sigma['z'])

        if kind not in sigma:
            print("Oops, I only know about x, y, z, and their combinations.")
            return None
        
        return signma[kind].normalize()
    
    @staticmethod
    def ops(q1, q2=None, q3=None, op="add", qs_type="ket", dim=10):
        
        new_states = []
        
        for new_q in q1.ops(q2, q3, op, dim):
            new_states.append(new_q)
            
        return Q8aStates(new_states, qs_type=qs_type, rows=dim, columns=1)
    
    def min(self, axis=0):
        """Return min values for all 8 positions as an ndarray.
           Pass axis=None to get the smallest value of all."""
        
        return np.min(self.array.reshape(self.dim, 8), axis=axis)
    
    def max(self, axis=0):
        """Return max values for all 8 positions as an ndarray.
           Pass axis=None to get the biggest value of all."""
        
        return np.max(self.array.reshape(self.dim, 8), axis=axis)
    
    def to_array(self):
        """Given a Q8aState, returns a np.ndarray."""
        
        return np.array(self.qs)


# In[6]:


class TestQ8aStates(unittest.TestCase):
    """Test states."""
    
    q_0 = Q8a().q_0()
    q_1 = Q8a().q_1()
    q_i = Q8a().q_i()
    q_n1 = Q8a([-1,0,0,0])
    q_2 = Q8a([2,0,0,0])
    q_n2 = Q8a([-2,0,0,0])
    q_3 = Q8a([3,0,0,0])
    q_n3 = Q8a([-3,0,0,0])
    q_4 = Q8a([4,0,0,0])
    q_5 = Q8a([5,0,0,0])
    q_6 = Q8a([6,0,0,0])
    q_10 = Q8a([10,0,0,0])
    q_n5 = Q8a([-5,0,0,0])
    q_7 = Q8a([7,0,0,0])
    q_8 = Q8a([8,0,0,0])
    q_9 = Q8a([9,0,0,0])
    q_n11 = Q8a([-11,0,0,0])
    q_21 = Q8a([21,0,0,0])
    q_n34 = Q8a([-34,0,0,0])
    v3 = Q8aStates([q_3])
    v1123 = Q8aStates([q_1, q_1, q_2, q_3])
    v3n1n21 = Q8aStates([q_3,q_n1,q_n2,q_1])
    q_1d0 = Q8a([1.0, 0, 0, 0])
    q12 = Q8aStates([q_1d0, q_1d0])
    q14 = Q8aStates([q_1d0, q_1d0, q_1d0, q_1d0])
    q19 = Q8aStates([q_1d0, q_0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0])
    v9 = Q8aStates([q_1, q_1, q_2, q_3, q_1, q_1, q_2, q_3, q_2])
    v9i = Q8aStates([Q8a([0,1,0,0]), Q8a([0,2,0,0]), Q8a([0,3,0,0]), Q8a([0,4,0,0]), Q8a([0,5,0,0]), Q8a([0,6,0,0]), Q8a([0,7,0,0]), Q8a([0,8,0,0]), Q8a([0,9,0,0])])
    vv9 = v9.add(v9i)
    qn627 = Q8a([-6,27,0,0])
    v33 = Q8aStates([q_7, q_0, q_n3, q_2, q_3, q_4, q_1, q_n1, q_n2])
    v33inv = Q8aStates([q_n2, q_3, q_9, q_8, q_n11, q_n34, q_n5, q_7, q_21])
    q_i3 = Q8aStates([q_1, q_1, q_1])
    q_i2d = Q8aStates([q_1, q_0, q_0, q_1])
    q_i3_bra = Q8aStates([q_1, q_1, q_1], "bra")
    # q_6_op = Q8aStates([q_1, q_0, q_0, q_1, q_i, q_i], "op")    
    q_6_op_32 = Q8aStates([q_1, q_0, q_0, q_1, q_i, q_i], "op", rows=3, columns=2)
    q_i2d_op = Q8aStates([q_1, q_0, q_0, q_1], "op")
    q_i4 = Q8a([0,4,0,0])
    q_0_q_1 = Q8aStates([q_0, q_1])
    q_1_q_0 = Q8aStates([q_1, q_0])
    q_1_q_i = Q8aStates([q_1, q_i])
    A = Q8aStates([Q8a([4,0,0,0]), Q8a([0,1,0,0])], "bra")
    B = Q8aStates([Q8a([0,0,1,0]), Q8a([0,0,0,2]), Q8a([0,3,0,0])])
    Op = Q8aStates([Q8a([3,0,0,0]), Q8a([0,1,0,0]), Q8a([0,0,2,0]), Q8a([0,0,0,3]), Q8a([2,0,0,0]), Q8a([0,4,0,0])], "op", rows=2, columns=3)
    Op4i = Q8aStates([q_i4, q_0, q_0, q_i4, q_2, q_3], "op", rows=2, columns=3) 
    Op_scalar = Q8aStates([q_i4], "scalar", rows=1, columns=1)
    q_1234 = Q8aStates([Q8a([1, 1, 0, 0]), Q8a([2, 1, 0, 0]), Q8a([3, 1, 0, 0]), Q8a([4, 1, 0, 0])])
    sigma_y = Q8aStates([Q8a([1, 0, 0, 0]), Q8a([0, -1, 0, 0]), Q8a([0, 1, 0, 0]), Q8a([-1, 0, 0, 0])])
    qn = Q8aStates([Q8a([3,0,0,4])])
    # q_bad = Q8aStates([q_1], rows=2, columns=3)
    
    b = Q8aStates([q_1, q_2, q_3], qs_type="bra")
    k = Q8aStates([q_4, q_5, q_6], qs_type="ket")
    o = Q8aStates([q_10], qs_type="op")
        
    def test_1000_init(self):
        self.assertTrue(self.q_0_q_1.dim == 2)
        
    def test_1020_set_rows_and_columns(self):
        self.assertTrue(self.q_i3.rows == 3)
        self.assertTrue(self.q_i3.columns == 1)
        self.assertTrue(self.q_i3_bra.rows == 1)
        self.assertTrue(self.q_i3_bra.columns == 3)
        self.assertTrue(self.q_i2d_op.rows == 2)
        self.assertTrue(self.q_i2d_op.columns == 2)
        self.assertTrue(self.q_6_op_32.rows == 3)
        self.assertTrue(self.q_6_op_32.columns == 2)
        
    def test_1030_equals(self):
        self.assertTrue(self.A.equals(self.A))
        self.assertFalse(self.A.equals(self.B))
        
    def test_1031_scalar(self):
        q_z = self.A.scalar()
        print("scalar(q): ", q_z)
        self.assertTrue(q_z.qs[0][0] == 4)
        self.assertTrue(q_z.qs[0][2] == 0)
        self.assertTrue(q_z.qs[0][4] == 0)
        self.assertTrue(q_z.qs[0][6] == 0)
        
    def test_1032_vector(self):
        q_z = self.B.vector()
        print("vector(q): ", q_z)
        self.assertTrue(q_z.qs[0][0] == 0)
        self.assertTrue(q_z.qs[0][2] == 0)
        self.assertTrue(q_z.qs[0][4] == 1)
        self.assertTrue(q_z.qs[0][6] == 0)
        
    def test_1033_t(self):
        q_z = self.A.t()
        print("q.t()): ", q_z)
        self.assertTrue(q_z[0] == 4)
    
    def test_1034_xyz(self):
        q_z = self.B.xyz()
        print("q.xyz()): ", q_z)
        self.assertTrue(q_z[0][0] == 0)
        self.assertTrue(q_z[0][1] == 1)
        self.assertTrue(q_z[0][2] == 0)
        
    def test_1035_txyz(self):
        q_z = self.B.txyz()
        print("q.txyz()): ", q_z)
        self.assertTrue(q_z[0][0] == 0)
        self.assertTrue(q_z[0][1] == 0)
        self.assertTrue(q_z[0][2] == 1)
        self.assertTrue(q_z[0][3] == 0)
        
    def test_1040_conj(self):
        qc = self.q_1_q_i.conj()
        qc1 = self.q_1_q_i.conj(1)
        print("q_1_q_i*: ", qc)
        print("q_1_qc*1: ", qc1)
        self.assertTrue(qc.qs[1][3] == 1)
        self.assertTrue(qc1.qs[1][2] == 1)
    
    def test_1042_conj_q(self):
        qc = self.q_1_q_i.conj_q(self.q_1)
        qc1 = self.q_1_q_i.conj_q(self.q_1)
        print("q_1_q_i* conj_q: ", qc)
        print("q_1_qc*1 conj_q: ", qc1)
        self.assertTrue(qc.qs[1][3] == 1)
        self.assertTrue(qc1.qs[1][3] == 1)
        
    def test_1050_flip_signs(self):
        qf = self.q_1_q_i.flip_signs()
        print("-q_1_q_i: ", qf)
        self.assertTrue(qf.qs[1][3] == 1)
        
    def test_1060_inverse(self):
        inv_v1123 = self.v1123.inverse()
        print("inv_v1123 operator", inv_v1123)
        vvinv = inv_v1123.product(self.v1123)
        vvinv.print_state("vinvD x v")
        self.assertTrue(vvinv.equals(self.q14))

        inv_v33 = self.v33.inverse()
        print("inv_v33 operator", inv_v33)
        vv33 = inv_v33.product(self.v33)
        vv33.print_state("inv_v33D x v33")
        self.assertTrue(vv33.equals(self.q19))
        
        Ainv = self.A.inverse()
        print("A ket inverse, ", Ainv)
        AAinv = self.A.product(Ainv)
        AAinv.print_state("A x AinvD")
        self.assertTrue(AAinv.equals(self.q12))
        
    def test_1070_normalize(self):
        qn = self.qn.normalize()
        print("Op normalized: ", qn)
        self.assertAlmostEqual(qn.qs[0][0], 0.6)
        self.assertTrue(qn.qs[0][6] == 0.8)
    
    def test_1080_determinant(self):
        det_v3 = self.v3.determinant()
        print("det v3:", det_v3)
        self.assertTrue(det_v3.equals(self.q_3))
        det_v1123 = self.v1123.determinant()
        print("det v1123", det_v1123)
        self.assertTrue(det_v1123.equals(self.q_1))
        det_v9 = self.v9.determinant()
        print("det_v9", det_v9)
        self.assertTrue(det_v9.equals(self.q_9))
        det_vv9 = self.vv9.determinant()
        print("det_vv9", det_vv9)
        self.assertTrue(det_vv9.equals(self.qn627))
        
    def test_1090_summation(self):
        q_01_sum = self.q_0_q_1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is Q8a)
        self.assertTrue(q_01_sum[0]== 1)
        
    def test_1100_add(self):
        q_0110_add = self.q_0_q_1.add(self.q_1_q_0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0][0]== 1)
        self.assertTrue(q_0110_add.qs[1][0]== 1)
        
    def test_1110_dif(self):
        q_0110_dif = self.q_0_q_1.dif(self.q_1_q_0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0][1]== 1)
        self.assertTrue(q_0110_dif.qs[1][0]== 1)
        
    def test_1120_diagonal(self):
        Op4iDiag2 = self.Op_scalar.diagonal(2)
        print("Op4i on a diagonal 2x2", Op4iDiag2)
        self.assertTrue(Op4iDiag2.qs[0].equals(self.q_i4))
        self.assertTrue(Op4iDiag2.qs[1].equals(Q8a().q_0()))
        
    def test_1130_identity(self):
        I2 = Q8aStates().identity(2, qs_type="operator")
        print("Operator Idenity, diagonal 2x2", I2)    
        self.assertTrue(I2.qs[0].equals(Q8a().q_1()))
        self.assertTrue(I2.qs[1].equals(Q8a().q_0()))
        I2 = Q8aStates().identity(2)
        print("Idenity on 2 state ket", I2)
        self.assertTrue(I2.qs[0].equals(Q8a().q_1()))
        self.assertTrue(I2.qs[1].equals(Q8a().q_1()))        

    def test_1140_product(self):
        self.assertTrue(self.b.product(self.o).equals(Q8aStates([Q8a([10,0,0,0]),Q8a([20,0,0,0]),Q8a([30,0,0,0])])))
        self.assertTrue(self.b.product(self.k).equals(Q8aStates([Q8a([32,0,0,0])])))
        self.assertTrue(self.b.product(self.o).product(self.k).equals(Q8aStates([Q8a([320,0,0,0])])))
        self.assertTrue(self.b.product(self.b).equals(Q8aStates([Q8a([1,0,0,0]),Q8a([4,0,0,0]),Q8a([9,0,0,0])])))
        self.assertTrue(self.o.product(self.k).equals(Q8aStates([Q8a([40,0,0,0]),Q8a([50,0,0,0]),Q8a([60,0,0,0])])))
        self.assertTrue(self.o.product(self.o).equals(Q8aStates([Q8a([100,0,0,0])])))
        self.assertTrue(self.k.product(self.k).equals(Q8aStates([Q8a([16,0,0,0]),Q8a([25,0,0,0]),Q8a([36,0,0,0])])))
        self.assertTrue(self.k.product(self.b).equals(Q8aStates([Q8a([4,0,0,0]),Q8a([5,0,0,0]),Q8a([6,0,0,0]),
                                                                      Q8a([8,0,0,0]),Q8a([10,0,0,0]),Q8a([12,0,0,0]),
                                                                      Q8a([12,0,0,0]),Q8a([15,0,0,0]),Q8a([18,0,0,0])])))
    
    def test_1150_product_AA(self):
        AA = self.A.product(self.A.ket())
        print("AA: ", AA)
        self.assertTrue(AA.equals(Q8aStates([Q8a([15, 0, 0, 0])])))
                  
    def test_1160_Euclidean_product_AA(self):
        AA = self.A.Euclidean_product(self.A.ket())
        print("A* A", AA)
        self.assertTrue(AA.equals(Q8aStates([Q8a([17, 0, 0, 0])])))

    def test_1170_product_AOp(self):
        AOp = self.A.product(self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8a([11, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8a([0, 0, 5, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8a([4, 0, 0, 0])))
                      
    def test_1180_Euclidean_product_AOp(self):
        AOp = self.A.Euclidean_product(self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8a([13, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8a([0, 0, 11, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8a([12, 0, 0, 0])))
        
    def test_1190_product_AOp4i(self):
        AOp4i = self.A.product(self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8a([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8a([-4, 0, 0, 0])))
                        
    def test_1200_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product(self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8a([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8a([4, 0, 0, 0])))

    def test_1210_product_OpB(self):
        OpB = self.Op.product(self.B)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8a([0, 10, 3, 0])))
        self.assertTrue(OpB.qs[1].equals(Q8a([-18, 0, 0, 1])))
                        
    def test_1220_Euclidean_product_OpB(self):
        OpB = self.Op.Euclidean_product(self.B)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8a([0, 2, 3, 0])))
        self.assertTrue(OpB.qs[1].equals(Q8a([18, 0, 0, -1])))

    def test_1230_product_AOpB(self):
        AOpB = self.A.product(self.Op).product(self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.equals(Q8aStates([Q8a([0, 22, 11, 0])])))
                        
    def test_1240_Euclidean_product_AOpB(self):
        AOpB = self.A.Euclidean_product(self.Op).product(self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.equals(Q8aStates([Q8a([0, 58, 13, 0])])))
        
    def test_1250_product_AOp4i(self):
        AOp4i = self.A.product(self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8a([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8a([-4, 0, 0, 0])))
                        
    def test_1260_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product(self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8a([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8a([4, 0, 0, 0])))

    def test_1270_product_Op4iB(self):
        Op4iB = self.Op4i.product(self.B)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(Q8a([0, 6, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(Q8a([0, 9, -8, 0])))
                        
    def test_1280_Euclidean_product_Op4iB(self):
        Op4iB = self.Op4i.Euclidean_product(self.B)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(Q8a([0, 6, 0, -4])))
        self.assertTrue(Op4iB.qs[1].equals(Q8a([0, 9, 8, 0])))

    def test_1290_product_AOp4iB(self):
        AOp4iB = self.A.product(self.Op4i).product(self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.equals(Q8aStates([Q8a([-9, 24, 0, 8])])))
                        
    def test_1300_Euclidean_product_AOp4iB(self):
        AOp4iB = self.A.Euclidean_product(self.Op4i).product(self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.equals(Q8aStates([Q8a([9, 24, 0, 24])])))

    def test_1310_op_n(self):
        opn = self.Op.op_n(n=self.q_i)
        print("op_n: ", opn)
        self.assertTrue(opn.qs[0][2] == 3)
        
    def test_1315_norm_squared(self):
        ns = self.q_1_q_i.norm_squared()
        ns.print_state("q_1_q_i norm squared")
        self.assertTrue(ns.equals(Q8aStates([Q8a([2,0,0,0])])))
        
    def test_1320_transpose(self):
        opt = self.q_1234.transpose()
        print("op1234 transposed: ", opt)
        self.assertTrue(opt.qs[0][0]== 1)
        self.assertTrue(opt.qs[1][0]== 3)
        self.assertTrue(opt.qs[2][0]== 2)
        self.assertTrue(opt.qs[3][0]== 4)
        optt = self.q_1234.transpose().transpose()
        self.assertTrue(optt.equals(self.q_1234))
        
    def test_1330_Hermitian_conj(self):
        q_hc = self.q_1234.Hermitian_conj()
        print("op1234 Hermtian_conj: ", q_hc)
        self.assertTrue(q_hc.qs[0][0]== 1)
        self.assertTrue(q_hc.qs[1][0]== 3)
        self.assertTrue(q_hc.qs[2][0]== 2)
        self.assertTrue(q_hc.qs[3][0]== 4)
        self.assertTrue(q_hc.qs[0][3] == 1)
        self.assertTrue(q_hc.qs[1][3] == 1)
        self.assertTrue(q_hc.qs[2][3] == 1)
        self.assertTrue(q_hc.qs[3][3] == 1)
        
    def test_1340_is_Hermitian(self):
        self.assertTrue(self.sigma_y.is_Hermitian())
        self.assertFalse(self.q_1234.is_Hermitian())
        
    def test_1350_is_square(self):
        self.assertFalse(self.Op.is_square())
        self.assertTrue(self.Op_scalar.is_square())    
    
    def test_1360_ops(self):
        q3 = Q8aStates().ops(q1=self.q_1, q2=self.q_1, op="dif", dim=3)
        q3.print_state("ops dif -1, dim=3")
        self.assertTrue(q3.qs[2][1] == 3.0)
        
    def test_1370_to_array(self):
        qa = self.q_1234.to_array()
        print("q_1234 to np.ndarray: \n", qa)
        self.assertTrue(type(qa).__name__ == "ndarray")
        
    def test_1380_min(self):
        qa = self.q_1234.min()
        print("q_1234.min(): ", qa)
        self.assertTrue(qa[0] == 1)
        qa = self.q_1234.min(axis=None)
        self.assertTrue(qa == 0)
        
    def test_1390_max(self):
        qa = self.q_1234.max()
        print("q_1234.max(): ", qa)
        self.assertTrue(qa[0] == 4)
        qa = self.q_1234.max(axis=None)
        self.assertTrue(qa == 4)
        
suite = unittest.TestLoader().loadTestsFromModule(TestQ8aStates())
unittest.TextTestRunner().run(suite);


# In[7]:


q1 = Q8a([1,2,3, 4])
q2 = Q8a([.1, -.2, -.3, .1])

for q in q1.ops(q2, dim=4):
    print(q)


# In[8]:


get_ipython().system('jupyter nbconvert --to python Q8nparray.ipynb')


# In[ ]:




