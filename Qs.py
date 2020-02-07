#!/usr/bin/env python
# coding: utf-8
"""
Developing Quaternions for iPython

Define a class Qs to manipulate quaternions as Hamilton would have done it so many years ago.
The "q_type" is a little bit of text to leave a trail of breadcrumbs about how a particular quaternion was generated.

The class QsStates is a semi-group with inverses, that has a row * column = dimensions as seen in
quantum mechanics.

The function calls for Qs and QsStates are meant to be very similar.
"""

from __future__ import annotations
import math
from copy import deepcopy

import numpy as np
import sympy as sp
from typing import Dict, List
from IPython.display import display
from bunch import Bunch


# noinspection PyTypeChecker
class Qs(object):
    """
    Quaternions as Hamilton would have defined them, on the manifold R^4.
    Add the usual operations should be here: add, dif, product, trig functions.
    """

    QS_TYPES = ["scalar_q", "bra", "ket", "op", "operator"]

    def __init__(self, values: object = None, q_type: object = "Q", representation: str = "", qs=None,
                 qs_type: str = "ket", rows: int = 0, columns: int = 0) -> Qs:

        self.qs = qs
        self.qs_type = qs_type
        self.rows = rows
        self.columns = columns

        if values is None:
            self.t, self.x, self.y, self.z = 0, 0, 0, 0
        elif len(values) == 4:
            self.t, self.x, self.y, self.z = values[0], values[1], values[2], values[3]

        elif len(values) == 8:
            self.t, self.x = values[0] - values[1], values[2] - values[3]
            self.y, self.z = values[4] - values[5], values[6] - values[7]

        else:
            raise ValueError(f"The program accepts lists/arrays of 4 or 8 dimensions, not {len(values)}")

        self.representation = representation

        # "Under the hood", all quaternions are manipulated in a Cartesian coordinate system.
        if representation != "":
            self.t, self.x, self.y, self.z = self.representation_2_txyz(representation)

        self.q_type = q_type

        if qs_type not in self.QS_TYPES:
            raise ValueError(f"Oops, only know of these quaternion series types: {self.QS_TYPES}")

        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = int(len(qs)), int(len(qs)), int(len(qs))

        self.set_qs_type(qs_type, rows, columns, copy=False)

    def set_qs_type(self: QsStates, qs_type: str = "", rows: int = 0, columns: int = 0, copy: bool = True) -> Qs:
        """
        Set the qs_type to something sensible.

        Args:
            qs_type: str:    can be scalar_q, ket, bra, op or operator
            rows: int        number of rows
            columns:         number of columns
            copy:

        Returns: QsStates

        """

        # Checks.
        if rows and columns and rows * columns != self.dim:
            raise ValueError(
                f"Oops, check those values again for rows:{rows} columns:{columns} dim:{self.dim}"
            )

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
                qs_type = "scalar_q"
            elif new_q.rows == 1:
                qs_type = "bra"
            elif new_q.columns == 1:
                qs_type = "ket"
            else:
                qs_type = "op"

        if not qs_type:
            raise Exception(
                "Oops, please set rows and columns for this quaternion series operator. Thanks."
            )

        if new_q.dim == 1:
            qs_type = "scalar_q"

        new_q.qs_type = qs_type

        return new_q

    def bra(self: QsStates) -> QsStates:
        """
        Quickly set the qs_type to bra by calling set_qs_type() with rows=1, columns=dim and taking a conjugate.

        Returns: QsStates

        """

        if self.qs_type == "bra":
            return self

        bra = deepcopy(self).conj()
        bra.rows = 1
        bra.columns = self.dim

        bra.qs_type = "bra" if self.dim > 1 else "scalar_q"

        return bra

    def ket(self: QsStates) -> QsStates:
        """
        Quickly set the qs_type to ket by calling set_qs_type() with rows=dim, columns=1 and taking a conjugate.

        Returns: QsStates

        """

        if self.qs_type == "ket":
            return self

        ket = deepcopy(self).conj()
        ket.rows = self.dim
        ket.columns = 1

        ket.qs_type = "ket" if self.dim > 1 else "scalar_q"

        return ket

    def op(self: QsStates, rows: int, columns: int) -> QsStates:
        """
        Quickly set the qs_type to op by calling set_qs_type().

        Args:
            rows: int:
            columns: int:

        Returns: QsStates

        """

        if rows * columns != self.dim:
            raise Exception(
                f"Oops, rows * columns != dim: {rows} * {columns}, {self.dimensions}"
            )

        op_q = deepcopy(self)

        op_q.rows = rows
        op_q.columns = columns

        if self.dim > 1:
            op_q.qs_type = "op"

        return op_q

    def __str__(self: QsStates, quiet: bool = False) -> str:
        """
        Print out all the states.

        Args:
            quiet: bool   Suppress printing the qtype.

        Returns: str

        """

        states = ""

        for n, q in enumerate(self.qs, start=1):
            states = states + f"n={n}: {q.__str__(quiet)}\n"

        return states.rstrip()

    def print_state(self: QsStates, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
        """
        Utility for printing states as a quaternion series.

        Returns: None

        """

        print(label)

        # Warn if empty.
        if self.qs is None or len(self.qs) == 0:
            raise ValueError("Oops, no quaternions in the series.")

        for n, q in enumerate(self.qs):
            print(f"n={n + 1}: {q.__str__(quiet)}")

        print(f"{self.qs_type}: {self.rows}/{self.columns}")

        if spacer:
            print("")
    def __str__(self, quiet: bool = False) -> str:
        """
        Customizes the output of a quaternion
        as a tuple given a particular representation.
        Since all quaternions 'look the same',
        the q_type after the tuple tries to summarize
        how this quaternions came into being.
        Quiet turns off printing the q_type.

        Args:
            quiet: bool

        Return: str
        """

        q_type = self.q_type

        if quiet:
            q_type = ""

        string = ""

        if self.representation == "":
            string = f"({self.t}, {self.x}, {self.y}, {self.z}) {q_type}"

        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            string = f"({rep[0]} A, {rep[1]} 𝜈x, {rep[2]} 𝜈y, {rep[3]} 𝜈z) {q_type}"

        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = f"({rep[0]} t, {rep[1]} R, {rep[2]} θ, {rep[3]} φ) {q_type}"

        return string

    def print_state(self: Qs, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
        """
        Utility to print a quaternion with a label.

        Args:
            label: str     User chosen
            spacer: bool   Adds a line return
            quiet: bool    Does not print q_type

        Return: None

        """

        print(label)

        print(self.__str__(quiet))

        if spacer:
            print("")

    def is_symbolic(self: Qs) -> bool:
        """
        Figures out if a quaternion has any symbolic terms.

        Return: bool

        """

        symbolic = False

        if (
                hasattr(self.t, "free_symbols")
                or hasattr(self.x, "free_symbols")
                or hasattr(self.y, "free_symbols")
                or hasattr(self.z, "free_symbols")
        ):
            symbolic = True

        return symbolic

    def txyz_2_representation(self: Qs, representation: str = "") -> List:
        """
        Given a quaternion in Cartesian coordinates
        returns one in another representation.
        Only 'polar' and 'spherical' are done so far.

        Args:
            representation: bool

        Return: Qs

        """

        symbolic = self.is_symbolic()
        rep = ""

        if representation == "":
            rep = [self.t, self.x, self.y, self.z]

        elif representation == "polar":
            amplitude = (self.t ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2) ** (
                    1 / 2
            )

            abs_v = abs_of_vector(self).t

            if symbolic:
                theta = sp.atan2(abs_v, self.t)
            else:
                theta = math.atan2(abs_v, self.t)

            if abs_v == 0:
                theta_x, theta_y, theta_z = 0, 0, 0

            else:
                theta_x = theta * self.x / abs_v
                theta_y = theta * self.y / abs_v
                theta_z = theta * self.z / abs_v

            rep = [amplitude, theta_x, theta_y, theta_z]

        elif representation == "spherical":

            spherical_t = self.t

            spherical_r = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** (1 / 2)

            if spherical_r == 0:
                theta = 0
            else:
                if symbolic:
                    theta = sp.acos(self.z / spherical_r)

                else:
                    theta = math.acos(self.z / spherical_r)

            if symbolic:
                phi = sp.atan2(self.y, self.x)
            else:
                phi = math.atan2(self.y, self.x)

            rep = [spherical_t, spherical_r, theta, phi]

        else:
            raise ValueError(f"Oops, don't know representation: representation")

        return rep

    def representation_2_txyz(self: Qs, representation: str = "") -> List:
        """
        Converts something in a representation such as
        polar, spherical
        and returns a Cartesian representation.

        Args:
            representation: str   can be polar or spherical

        Return: Qs

        """

        symbolic = False

        if (
                hasattr(self.t, "free_symbols")
                or hasattr(self.x, "free_symbols")
                or hasattr(self.y, "free_symbols")
                or hasattr(self.z, "free_symbols")
        ):
            symbolic = True

        if representation == "":
            box_t, box_x, box_y, box_z = self.t, self.x, self.y, self.z

        elif representation == "polar":
            amplitude, theta_x, theta_y, theta_z = self.t, self.x, self.y, self.z

            theta = (theta_x ** 2 + theta_y ** 2 + theta_z ** 2) ** (1 / 2)

            if theta == 0:
                box_t = self.t
                box_x, box_y, box_z = 0, 0, 0

            else:
                if symbolic:
                    box_t = amplitude * sp.cos(theta)
                    box_x = self.x / theta * amplitude * sp.sin(theta)
                    box_y = self.y / theta * amplitude * sp.sin(theta)
                    box_z = self.z / theta * amplitude * sp.sin(theta)
                else:
                    box_t = amplitude * math.cos(theta)
                    box_x = self.x / theta * amplitude * math.sin(theta)
                    box_y = self.y / theta * amplitude * math.sin(theta)
                    box_z = self.z / theta * amplitude * math.sin(theta)

        elif representation == "spherical":
            box_t, R, theta, phi = self.t, self.x, self.y, self.z

            if symbolic:
                box_x = R * sp.sin(theta) * sp.cos(phi)
                box_y = R * sp.sin(theta) * sp.sin(phi)
                box_z = R * sp.cos(theta)
            else:
                box_x = R * math.sin(theta) * math.cos(phi)
                box_y = R * math.sin(theta) * math.sin(phi)
                box_z = R * math.cos(theta)

        elif representation == "hyperbolic":
            u, v, theta, phi = self.t, self.x, self.y, self.z

            if symbolic:
                box_t = v * sp.exp(u)
                box_x = v * sp.exp(-u)
                box_y = v * sp.sin(theta) * sp.sin(phi)
                box_z = v * sp.cos(theta)

            else:
                box_t = v * math.exp(u)
                box_x = v * math.exp(-u)
                box_y = v * math.sin(theta) * sp.sin(phi)
                box_z = v * math.cos(theta)

        else:
            raise ValueError(f"Oops, don't know representation: representation")

        return [box_t, box_x, box_y, box_z]

    def check_representations(self: Qs, q_2: Qs) -> bool:
        """
        Checks if self and q_2 have the same representation.

        Args:
            q_2: Qs

        Returns: bool

        """

        if self.representation == q_2.representation:
            return True

        else:
            raise Exception(f"Oops, 2 have different representations: {self.representation} {q_2.representation}")

    def display_q(self: Qs, label: str = "") -> Qs:
        """
        Prints LaTeX-like output, one line for each of th 4 terms.

        Args:
            label: str  an additional bit of text.

        Returns: Qs

        """

        if label:
            print(label)
        display(self.t)
        display(self.x)
        display(self.y)
        display(self.z)
        return

    def simple_q(self: Qs) -> Qs:
        """
        Runs symboy.simplify() on each term, good for symbolic expression.

        Returns: Qs

        """

        self.t = sp.simplify(self.t)
        self.x = sp.simplify(self.x)
        self.y = sp.simplify(self.y)
        self.z = sp.simplify(self.z)
        return self

    def expand_q(self) -> Qs:
        """
        Runs expand on each term, good for symbolic expressions.

        Returns: Qs

        """
        """Expand each term."""

        self.t = sp.expand(self.t)
        self.x = sp.expand(self.x)
        self.y = sp.expand(self.y)
        self.z = sp.expand(self.z)
        return self

    def subs(self: Qs, symbol_value_dict: annotations.Dict) -> Qs:
        """
        Evaluates a quaternion using sympy values and a dictionary {t:1, x:2, etc}.

        Args:
            symbol_value_dict: Dict

        Returns: Qs

        """

        t1 = self.t.subs(symbol_value_dict)
        x1 = self.x.subs(symbol_value_dict)
        y1 = self.y.subs(symbol_value_dict)
        z1 = self.z.subs(symbol_value_dict)

        q_txyz = Qs(
            [t1, x1, y1, z1], q_type=self.q_type, representation=self.representation
        )

        return q_txyz


def scalar_q(q_1: Qs) -> Qs:
    """
    Returns the scalar_q part of a quaternion as a quaternion.

    $ \rm{scalar_q(q)} = (q + q^*)/2 = (t, 0) $

    Returns: Qs

    Args:
        q_1: Qs

    Returns: Qs

    """

    end_q_type = f"scalar_q({q_1.q_type})"

    s = Qs([q_1.t, 0, 0, 0], q_type=end_q_type, representation=q_1.representation)
    return s


def vector_q(q_1: Qs) -> Qs:
    """
    Returns the vector_q part of a quaternion.
    $ \rm{vector_q(q)} = (q\_1 - q\_1^*)/2 = (0, R) $

    Returns: Qs

    """

    end_q_type = f"vector_q({q_1.q_type})"

    v = Qs(
        [0, q_1.x, q_1.y, q_1.z],
        q_type=end_q_type,
        representation=q_1.representation,
    )
    return v


def t(q_1: Qs) -> np.array:
    """
    Returns the scalar_q t as an np.array.

    Returns: np.array

    """

    return np.array([q_1.t])


def xyz(q_1: Qs) -> np.array:
    """
    Returns the vector_q x, y, z as an np.array.

    Returns: np.array

    """

    return np.array([q_1.x, q_1.y, q_1.z])


def q0(q_type: str = "0", representation: str = "") -> Qs:
    """
    Return a zero quaternion.

    $ q\_0() = 0 = (0, 0) $

    Returns: Qs

    """

    return Qs([0, 0, 0, 0], q_type=q_type, representation=representation)


def q1(n: float = 1.0, q_type: str = "1", representation: str = "") -> Qs:
    """
    Return a real-valued quaternion multiplied by n.

    $ q\_1(n) = n = (n, 0) $

    Returns: Qs

    """

    return Qs([n, 0, 0, 0], q_type=q_type, representation=representation)


def qi(n: float = 1.0, q_type: str = "i", representation: str = "") -> Qs:
    """
    Return a quaternion with $ i * n $.

    $ q\_i(n) = n i = (0, n i) $

    Returns: Qs

    """

    return Qs([0, n, 0, 0], q_type=q_type, representation=representation)


def qj(n: float = 1.0, q_type: str = "j", representation: str = "") -> Qs:
    """
    Return a quaternion with $ j * n $.

    $ q\_j(n) = n j = (0, n j) $

    Returns: Qs

    """

    return Qs([0, 0, n, 0], q_type=q_type, representation=representation)


def qk(n: float = 1, q_type: str = "k", representation: str = "") -> Qs:
    """
    Return a quaternion with $ k * n $.

    $ q\_k(n) = n k =(0, n k) $

    Returns: Qs

    """

    return Qs([0, 0, 0, n], q_type=q_type, representation=representation)


def qrandom(low: float = -1.0, high: float = 1.0, distribution: str = "uniform", q_type: str = "?",
            representation: str = "") -> Qs:
    """
    Return a random-valued quaternion.
    The distribution is uniform, but one could add to options.
    It would take some work to make this clean so will skip for now.

    Args:
        low:
        high:
        distribution: str:    hove only implemented uniform distribution
        q_type:               ?
        representation:       Cartesian by default

    Returns: Qs

    """

    random_distributions = Bunch()
    random_distributions.uniform = np.random.uniform

    qr = Qs(
        [
            random_distributions[distribution](low=low, high=high),
            random_distributions[distribution](low=low, high=high),
            random_distributions[distribution](low=low, high=high),
            random_distributions[distribution](low=low, high=high),
        ],
        q_type=q_type,
        representation=representation,
    )
    return qr


def dupe(q_1: Qs) -> Qs:
    """
    Return a duplicate copy.

    Returns: Qs

    """

    du = Qs(
        [q_1.t, q_1.x, q_1.y, q_1.z],
        q_type=q_1.q_type,
        representation=q_1.representation,
    )
    return du


def equals(q_1: Qs, q_2: Qs, scalar: bool = True, vector: bool = True) -> bool:
    """
    Tests if q1 and q_2 quaternions are close to equal. If vector_q is set to False, will compare
    only the scalar_q. If scalar_q is set to False, will compare 3-vectors.

    $ q.equals(q\_2) = q == q\_2 = True $

    Args:
        q_1: Qs
        q_2: Qs
        scalar: bool    Will compare quaternion scalars
        vector: bool    Will compare quaternion 3-vectors

    Returns: bool

    """

    q_1.check_representations(q_2)

    q_1_t, q_1_x, q_1_y, q_1_z = (
        sp.expand(q_1.t),
        sp.expand(q_1.x),
        sp.expand(q_1.y),
        sp.expand(q_1.z),
    )
    q_2_t, q_2_x, q_2_y, q_2_z = (
        sp.expand(q_2.t),
        sp.expand(q_2.x),
        sp.expand(q_2.y),
        sp.expand(q_2.z),
    )

    if not scalar and not vector:
        raise ValueError("Equals needs either scalar_q or vector_q to be set to True")

    t_equals = math.isclose(q_1_t, q_2_t)
    x_equals = math.isclose(q_1_x, q_2_x)
    y_equals = math.isclose(q_1_y, q_2_y)
    z_equals = math.isclose(q_1_z, q_2_z)

    result = False

    if scalar and not vector and t_equals:
        result = True

    elif not scalar and vector and x_equals and y_equals and z_equals:
        result = True

    elif scalar and vector and t_equals and x_equals and y_equals and z_equals:
        result = True

    return result


def conj(q_1: Qs, conj_type: int = 0) -> Qs:
    """
    There are 4 types of conjugates.

    $ q.conj(0) = q^* =(t, -x, -y, -z) $
    $ q.conj(1) = (i q i)^* =(-t, x, -y, -z) $
    $ q.conj(2) = (j q j)^* =(-t, -x, y, -z) $
    $ q.conj(3) = (k q k)^* =(-t, -x, -y, z) $

    Args:
        q_1: Qs
        conj_type: int:   0-3 depending on who stays positive.

    Returns: Qs

    """

    end_q_type = f"{q_1.q_type}*"
    c_t, c_x, c_y, c_z = q_1.t, q_1.x, q_1.y, q_1.z
    cq = Qs()

    if conj_type % 4 == 0:
        cq.t = c_t
        if c_x != 0:
            cq.x = -1 * c_x
        if c_y != 0:
            cq.y = -1 * c_y
        if c_z != 0:
            cq.z = -1 * c_z

    elif conj_type % 4 == 1:
        if c_t != 0:
            cq.t = -1 * c_t
        cq.x = c_x
        if c_y != 0:
            cq.y = -1 * c_y
        if c_z != 0:
            cq.z = -1 * c_z
        end_q_type += "1"

    elif conj_type % 4 == 2:
        if c_t != 0:
            cq.t = -1 * c_t
        if c_x != 0:
            cq.x = -1 * c_x
        cq.y = c_y
        if c_z != 0:
            cq.z = -1 * c_z
        end_q_type += "2"

    elif conj_type % 4 == 3:
        if c_t != 0:
            cq.t = -1 * c_t
        if c_x != 0:
            cq.x = -1 * c_x
        if c_y != 0:
            cq.y = -1 * c_y
        cq.z = c_z
        end_q_type += "3"

    cq.q_type = end_q_type
    cq.representation = q_1.representation

    return cq


def conj_q(q_1: Qs, q_2: Qs) -> Qs:
    """
    Given a quaternion with 0s or 1s, will do the standard conjugate, first conjugate
    second conjugate, sign flip, or all combinations of the above.

    q.conj(q(1, 1, 1, 1)) = q.conj(0).conj(1).conj(2).conj(3)

    Args:
        q_1: Qs
        q_2: Qs    Use a quaternion to do one of 4 conjugates in combinations

    Returns: Qs

    """

    _conj = deepcopy(q_1)

    if q_2.t:
        _conj = conj(_conj, conj_type=0)

    if q_2.x:
        _conj = conj(_conj, conj_type=1)

    if q_2.y:
        _conj = conj(_conj, conj_type=2)

    if q_2.z:
        _conj = flip_signs(_conj)

    return _conj


def flip_signs(q_1: Qs) -> Qs:
    """
    Flip the signs of all terms.

    $ q.flip\_signs() = -q = (-t, -R) $

    Args:
        q_1: Qs

    Returns: Qs
    """

    end_q_type = f"-{q_1.q_type}"

    flip_t, flip_x, flip_y, flip_z = q_1.t, q_1.x, q_1.y, q_1.z

    flip_q = Qs(q_type=end_q_type, representation=q_1.representation)
    if flip_t != 0:
        flip_q.t = -1 * flip_t
    if flip_x != 0:
        flip_q.x = -1 * flip_x
    if flip_y != 0:
        flip_q.y = -1 * flip_y
    if flip_z != 0:
        flip_q.z = -1 * flip_z

    return flip_q


def vahlen_conj(q_1: Qs, conj_type: str = "-", q_type: str = "vc") -> Qs:
    """
    Three types of conjugates dash, apostrophe, or star as done by Vahlen in 1901.

    q.vahlen_conj("-") = q^* = (t, -x, -y, -z)

    q.vahlen_conj("'") = (k q k)^* = (t, -x, -y, z)

    q.vahlen_conj("*") = -(k q k)^* = (t, x, y, -z)

    Args:
        q_1: Qs
        conj_type: str:    3 sorts, dash apostrophe,
        q_type: str:

    Returns:

    """

    vc_t, vc_x, vc_y, vc_z = q_1.t, q_1.x, q_1.y, q_1.z
    c_q = Qs()

    if conj_type == "-":
        c_q.t = vc_t
        if vc_x != 0:
            c_q.x = -1 * vc_x
        if vc_y != 0:
            c_q.y = -1 * vc_y
        if vc_z != 0:
            c_q.z = -1 * vc_z
        q_type += "*-"

    if conj_type == "'":
        c_q.t = vc_t
        if vc_x != 0:
            c_q.x = -1 * vc_x
        if vc_y != 0:
            c_q.y = -1 * vc_y
        c_q.z = vc_z
        q_type += "*'"

    if conj_type == "*":
        c_q.t = vc_t
        c_q.x = vc_x
        c_q.y = vc_y
        if vc_z != 0:
            c_q.z = -1 * vc_z
        q_type += "*"

    c_q.q_type = f"{q_1.q_type}{q_type}"
    c_q.representation = q_1.representation

    return c_q


def _commuting_products(q_1: Qs, q_2: Qs) -> Dict:
    """
    Returns a dictionary with the commuting products. For internal use.

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Dict

    """

    s_t, s_x, s_y, s_z = q_1.t, q_1.x, q_1.y, q_1.z
    q_2_t, q_2_x, q_2_y, q_2_z = q_2.t, q_2.x, q_2.y, q_2.z

    products = {
        "tt": s_t * q_2_t,
        "xx+yy+zz": s_x * q_2_x + s_y * q_2_y + s_z * q_2_z,
        "tx+xt": s_t * q_2_x + s_x * q_2_t,
        "ty+yt": s_t * q_2_y + s_y * q_2_t,
        "tz+zt": s_t * q_2_z + s_z * q_2_t,
    }

    return products


def _anti_commuting_products(q_1: Qs, q_2: Qs) -> Dict:
    """
    Returns a dictionary with the three anti-commuting products. For internal use.

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Dict

    """

    s_x, s_y, s_z = q_1.x, q_1.y, q_1.z
    q_2_x, q_2_y, q_2_z = q_2.x, q_2.y, q_2.z

    products = {
        "yz-zy": s_y * q_2_z - s_z * q_2_y,
        "zx-xz": s_z * q_2_x - s_x * q_2_z,
        "xy-yx": s_x * q_2_y - s_y * q_2_x,
        "zy-yz": -s_y * q_2_z + s_z * q_2_y,
        "xz-zx": -s_z * q_2_x + s_x * q_2_z,
        "yx-xy": -s_x * q_2_y + s_y * q_2_x,
    }

    return products


def _all_products(q_1: Qs, q_2: Qs) -> Dict:
    """
    All products, commuting and anti-commuting products as a dictionary. For internal use.

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Dict

    """

    products = _commuting_products(q_1, q_2)
    products.update(_anti_commuting_products(q_1, q_2))

    return products


def square(q_1: Qs) -> Qs:
    """
    Square a quaternion.

    $ q.square() = q^2 = (t^2 - R.R, 2 t R) $

    Args:
        q_1: Qs

    Returns:
        Qs

    """

    end_q_type = f"{q_1.q_type}²"

    qxq = _commuting_products(q_1, q_1)

    sq_q = Qs(q_type=end_q_type, representation=q_1.representation)
    sq_q.t = qxq["tt"] - qxq["xx+yy+zz"]
    sq_q.x = qxq["tx+xt"]
    sq_q.y = qxq["ty+yt"]
    sq_q.z = qxq["tz+zt"]

    return sq_q


def norm_squared(q_1: Qs) -> Qs:
    """
    The norm_squared of a quaternion.

    $ q.norm\_squared() = q q^* = (t^2 + R.R, 0) $

    Returns: Qs

    """

    end_q_type = f"||{q_1.q_type}||²"

    qxq = _commuting_products(q_1, q_1)

    n_q = Qs(q_type=end_q_type, representation=q_1.representation)
    n_q.t = qxq["tt"] + qxq["xx+yy+zz"]

    return n_q


def norm_squared_of_vector(q_1: Qs):
    """
    The norm_squared of the vector_q of a quaternion.

    $ q.norm\_squared\_of\_vector_q() = ((q - q^*)(q - q^*)^*)/4 = (R.R, 0) $

    Returns: Qs
    """

    end_q_type = f"|V({q_1.q_type})|²"

    qxq = _commuting_products(q_1, q_1)

    nv_q = Qs(q_type=end_q_type, representation=q_1.representation)
    nv_q.t = qxq["xx+yy+zz"]

    return nv_q


def abs_of_q(q_1: Qs) -> Qs:
    """
    The absolute value, the square root of the norm_squared.

    $ q.abs_of_q() = \sqrt{q q^*} = (\sqrt{t^2 + R.R}, 0) $

    Returns: Qs

    """

    end_q_type = f"|{q_1.q_type}|"

    a = norm_squared(q_1)
    sqrt_t = a.t ** (1 / 2)
    a.t = sqrt_t
    a.q_type = end_q_type
    a.representation = q_1.representation

    return a


def normalize(q_1: Qs, n: float = 1.0, q_type: str = "U") -> Qs:
    """
    Normalize a quaternion to a given value n.

    $ q.normalized(n) = q (q q^*)^{-1} = q (n/\sqrt{q q^*}, 0) $

    Args:
        q_1: Qs
        n: float       Make the norm equal to n.
        q_type: str

    Returns: Qs

    """

    end_q_type = f"{q_1.q_type} {q_type}"

    abs_q_inv = inverse(abs_of_q(q_1))
    n_q = product(product(q_1, abs_q_inv), Qs([n, 0, 0, 0]))
    n_q.q_type = end_q_type
    n_q.representation = q_1.representation

    return n_q


def abs_of_vector(q_1: Qs) -> Qs:
    """
    The absolute value of the vector_q, the square root of the norm_squared of the vector_q.

    $ q.abs_of_vector() = \sqrt{(q\_1 - q\_1^*)(q\_1 - q\_1^*)/4} = (\sqrt{R.R}, 0) $

    Args:
        q_1: Qs

    Returns: Qs

    """

    end_q_type = f"|V({q_1.q_type})|"

    av = norm_squared_of_vector(q_1)
    sqrt_t = av.t ** (1 / 2)
    av.t = sqrt_t
    av.representation = q_1.representation
    av.q_type = end_q_type

    return av


def add(q_1: Qs, q_2: Qs) -> Qs:
    """
    Add two quaternions.

    $ q.add(q\_2) = q_1 + q\_2 = (t + t\_2, R + R\_2) $

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Qs

    """

    q_1.check_representations(q_2)

    add_q_type = f"{q_1.q_type}+{q_2.q_type}"

    t_1, x_1, y_1, z_1 = q_1.t, q_1.x, q_1.y, q_1.z
    t_2, x_2, y_2, z_2 = q_2.t, q_2.x, q_2.y, q_2.z

    add_q = Qs(q_type=add_q_type, representation=q_1.representation)
    add_q.t = t_1 + t_2
    add_q.x = x_1 + x_2
    add_q.y = y_1 + y_2
    add_q.z = z_1 + z_2

    return add_q


def dif(q_1: Qs, q_2: Qs) -> Qs:
    """
    Takes the difference of 2 quaternions.

    $ q.dif(q\_2) = q_1 - q\_2 = (t - t\_2, R - R\_2) $

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Qs

    """

    q_1.check_representations(q_2)

    end_dif_q_type = f"{q_1.q_type}-{q_2.q_type}"

    t_2, x_2, y_2, z_2 = q_2.t, q_2.x, q_2.y, q_2.z
    t_1, x_1, y_1, z_1 = q_1.t, q_1.x, q_1.y, q_1.z

    dif_q = Qs(q_type=end_dif_q_type, representation=q_1.representation)
    dif_q.t = t_1 - t_2
    dif_q.x = x_1 - x_2
    dif_q.y = y_1 - y_2
    dif_q.z = z_1 - z_2

    return dif_q


def product(q_1: Qs, q_2: Qs, kind: str = "", reverse: bool = False) -> Qs:
    """
    Form a product given 2 quaternions. Kind of product can be '' aka standard, even, odd, or even_minus_odd.
    Setting reverse=True is like changing the order.

    $ q.product(q_2) = q\_1 q\_2 = (t t_2 - R.R_2, t R_2 + R t_2 + RxR_2 ) $

    $ q.product(q_2, kind="even") = (q\_1 q\_2 + (q q\_2)^*)/2 = (t t_2 - R.R_2, t R_2 + R t_2 ) $

    $ q.product(q_2, kind="odd") = (q\_1 q\_2 - (q q\_2)^*)/2 = (0, RxR_2 ) $

    $ q.product(q_2, kind="even_minus_odd") = q\_2 q\_1 = (t t_2 - R.R_2, t R_2 + R t_2 - RxR_2 ) $

    $ q.product(q_2, reverse=True) = q\_2 q\_1 = (t t_2 - R.R_2, t R_2 + R t_2 - RxR_2 ) $

    Args:
        q_1: Qs
        q_2: Qs:
        kind: str:    can be blank, even, odd, or even_minus_odd
        reverse: bool:  if true, returns even_minus_odd

    Returns: Qs

    """

    q_1.check_representations(q_2)

    commuting = _commuting_products(q_1, q_2)
    q_even = Qs()
    q_even.t = commuting["tt"] - commuting["xx+yy+zz"]
    q_even.x = commuting["tx+xt"]
    q_even.y = commuting["ty+yt"]
    q_even.z = commuting["tz+zt"]

    anti_commuting = _anti_commuting_products(q_1, q_2)
    q_odd = Qs()

    if reverse:
        q_odd.x = anti_commuting["zy-yz"]
        q_odd.y = anti_commuting["xz-zx"]
        q_odd.z = anti_commuting["yx-xy"]

    else:
        q_odd.x = anti_commuting["yz-zy"]
        q_odd.y = anti_commuting["zx-xz"]
        q_odd.z = anti_commuting["xy-yx"]

    if kind == "":
        result = add(q_even, q_odd)
        times_symbol = "x"
    elif kind.lower() == "even":
        result = q_even
        times_symbol = "xE"
    elif kind.lower() == "odd":
        result = q_odd
        times_symbol = "xO"
    elif kind.lower() == "even_minus_odd":
        result = dif(q_even, q_odd)
        times_symbol = "xE-xO"
    else:
        raise Exception(
            "Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'."
        )

    if reverse:
        times_symbol = times_symbol.replace("x", "xR")

    result.q_type = f"{q_1.q_type}{times_symbol}{q_2.q_type}"
    result.representation = q_1.representation

    return result


def cross_q(q_1: Qs, q_2: Qs, reverse: bool = False) -> Qs:
    """
    Convenience function, calling product with kind="odd".
    Called 'cross_q' to imply it returns 4 numbers, not the standard 3.

    Args:
        q_1: Qs
        q_2: Qs
        reverse: bool

    Returns: Qs

    """
    return product(q_1, q_2, kind="odd", reverse=reverse)


def inverse(q_1: Qs, additive: bool = False) -> Qs:
    """
    The additive or multiplicative inverse of a quaternion. Defaults to 1/q, not -q.

    $ q.inverse() = q^* (q q^*)^{-1} = (t, -R) / (t^2 + R.R) $

    $ q.inverse(additive=True) = -q = (-t, -R) $

    Args:
        q_1: Qs
        additive: bool

    Returns: Qs

    """

    if additive:
        end_q_type = f"-{q_1.q_type}"
        q_inv = flip_signs(q_1)
        q_inv.q_type = end_q_type

    else:
        end_q_type = f"{q_1.q_type}⁻¹"

        q_conj = conj(q_1)
        q_norm_squared = norm_squared(q_1)

        if (not q_1.is_symbolic()) and (q_norm_squared.t == 0):
            return q0()

        q_norm_squared_inv = Qs([1.0 / q_norm_squared.t, 0, 0, 0])
        q_inv = product(q_conj, q_norm_squared_inv)
        q_inv.q_type = end_q_type
        q_inv.representation = q_1.representation

    return q_inv


def divide_by(q_1: Qs, q_2: Qs) -> Qs:
    """
    Divide one quaternion by another. The order matters unless one is using a norm_squared (real number).

    $ q.divided_by(q_2) = q\_1 q_2^{-1} = (t t\_2 + R.R\_2, -t R\_2 + R t\_2 - RxR\_2) $

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Qs

    """

    q_1.check_representations(q_2)

    end_q_type = f"{q_1.q_type}/{q_2.q_type}"

    q_div = product(q_1, inverse(q_2))
    q_div.q_type = end_q_type
    q_div.representation = q_1.representation

    return q_div


def triple_product(q_1: Qs, q_2: Qs, q_3: Qs) -> Qs:
    """
    Form a triple product given 3 quaternions, in left-to-right order: q1, q_2, q_3.

    $ q.triple_product(q_2, q_3) = q q_2 q_3 $

    $ = (t t\_2 t\_3 - R.R\_2 t\_3 - t R\_2.R|_3 - t\_2 R.R\_3 - (RxR_2).R\_3, $

    $ ... t t\_2 R\_3 - (R.R\_2) R\_3 + t t\_3 R\_2 + t\_2 t\_3 R $

    $ ... + t\_3 RxR\_2 + t R_2xR\_3 + t_2 RxR\_3 + RxR\_2xR\_3) $

    Args:
        q_1: Qs
        q_2: Qs:
        q_3: Qs:

    Returns: Qs

    """

    q_1.check_representations(q_2)
    q_1.check_representations(q_3)

    triple = product(product(q_1, q_2), q_3)
    triple.representation = q_1.representation

    return triple


# Quaternion rotation involves a triple product:  u R 1/u
def rotate(q_1: Qs, u: Qs) -> Qs:
    """
    Do a rotation using a triple product: u R 1/u.

    $ q.rotate(u) = u q u^{-1} $

    $ = (u^2 t - u V.R + u R.V + t V.V, $
    $ ... - u t V + (V.R) V + u^2 R + V t u + VxR u - u RxV - VxRxV) $

    Args:
        q_1: Qs
        u: Qs    pre-multiply by u, post-multiply by $u^{-1}$.

    Returns: Qs

    """

    q_1.check_representations(u)
    end_q_type = f"{q_1.q_type}*rot"

    q_rot = triple_product(u, q_1, inverse(u))
    q_rot.q_type = end_q_type
    q_rot.representation = q_1.representation

    return q_rot


def rotation_angle(q_1: Qs, q_2: Qs, origin: Qs = q0(), tangent_space_norm: float = 1.0, degrees: bool = False) -> Qs:
    """
    Returns the spatial angle between the origin and 2 points.

    $$ scalar(normalize(vector(q_1) vector(q_2)^*)) = \cos(a) $$

    The product of the 3-vectors is a mix of symmetric and anti-symmetric terms.
    The normalized scalar is $\cos(a)$. Take the inverse cosine to get the angle
    for the angle in the plane between q_1, the origin and q_2.

    I have a radical view of space-time. It is where-when everything happens. In space-time, all algebra
    operations are the same: $ 2 + 3 = 5 $ and $ 2 * 3 = 6 $. The same cannot be said about the tangent
    space of space-time because it is tangent space that can be 'curved'. My proposal for gravity is that
    changes in the tangent space measurements of time, dt, exactly cancel those of the tangent space
    measurements of space, dR. When one is making a measurement that involves gravity, the tangent space
    norm will not be equal to unity, but greater or lesser than unity.

    There are a number of bothersome qualities about this function. The scalar term doesn't matter in the
    slightest way. As a consequence, this is a purely spatial function.

    Args:
        q_1: Qs
        q_2: Qs
        origin: Qs    default is zero.
        tangent_space_norm: float   Will be different from unity in 'curved' tangent spaces
        degrees: float    Use degrees instead of radians

    Returns: Qs    only the scalar is possibly non-zero

    """

    q_1_shifted_vector = vector_q(dif(q_1, origin))
    q_2_shifted_vector = vector_q(dif(q_2, origin))

    q_1__q_2 = normalize(product(q_1_shifted_vector, conj(q_2_shifted_vector)), n=tangent_space_norm)
    angle = math.acos(q_1__q_2.t)

    if degrees:
        angle = angle * 180 / math.pi

    return Qs([angle, 0, 0, 0])

def rotation_and_or_boost(q_1: Qs, h: Qs) -> Qs:
    """
    The method for doing a rotation in 3D space discovered by Rodrigues in the 1840s used a quaternion triple
    product. After Minkowski characterized Einstein's work in special relativity as a 4D rotation, efforts were
    made to do the same with one quaternion triple product. That obvious goal was not achieved until 2010 by
    D. Sweetser and indpendently by M. Kharinov. Two other triple products need to be used like so:

    $ b.rotation_and_or_boost(h) = h b h^* + 1/2 ((hhb)^* -(h^* h^* b)^*) $

    The parameter h is NOT free from constraints. There are two constraints. If the parameter h is to do a
    rotation, it must have a norm of unity and have the first term equal to zero.

    $ h = (0, R), scalar_q(h) = 0, scalar_q(h h^*) = 1 $

    To do a boost which may or may not also do a rotation, then the parameter h must have a square whose first
    term is equal to zero:

    $ h = (\cosh(a), \sinh(a)), scalar_q(h^2) = 1 $

    There has been no issue about the ability of this function to do boosts. There has been a spirited debate
    as to whether the function can do rotations. Notice that the form reduces to the Rodrigues triple product.
    I consider this so elementary that I cannot argue the other side. Please see the wiki page or use this code
    to see for yourq_1.

    Args:
        q_1: Qs
        h: Qs

    Returns: Qs

    """
    q_1.check_representations(h)
    end_q_type = f"{q_1.q_type}rotation/boost"

    # if not h.is_symbolic():
    #     if math.isclose(h.t, 0):
    #         if not math.isclose(h.norm_squared().t, 1):
    #             h = h.normalize()
    #             h.print_state("To do a 3D rotation, h adjusted value so scalar_q(h h^*) = 1")

    #     else:
    #         if not math.isclose(h.square().t, 1):
    #             h = Qs.Lorentz_next_boost(h, Qs.q1())
    #             h.print_state("To do a Lorentz boost, h adjusted value so scalar_q(h²) = 1")

    triple_1 = triple_product(h, q_1, conj(h))
    triple_2 = conj(triple_product(h, h, q_1))
    triple_3 = conj(triple_product(conj(h), conj(h), q_1))

    triple_23 = dif(triple_2, triple_3)
    half_23 = product(triple_23, Qs([0.5, 0, 0, 0], representation=q_1.representation))
    triple_123 = add(triple_1, half_23)
    triple_123.q_type = end_q_type
    triple_123.representation = q_1.representation

    return triple_123


def Lorentz_next_rotation(q_1: Qs, q_2: Qs) -> Qs:
    """
    Given 2 quaternions, creates a new quaternion to do a rotation
    in the triple triple quaternion function by using a normalized cross product.

    $ Lorentz_next_rotation(q, q_2) = (q q\_2 - q\_2 q) / 2|(q q\_2 - (q\_2 q)^*)| = (0, QxQ\_2)/|(0, QxQ\_2)| $

    Args:
        q_1: Qs   any quaternion
        q_2: Qs   any quaternion whose first term equals the first term of q and
                  for the first terms of each squared.

    Returns: Qs

    """
    q_1.check_representations(q_2)

    if not math.isclose(q_1.t, q_2.t):
        raise ValueError(f"Oops, to be a rotation, the first values must be the same: {q_1.t} != {q_2.t}")

    if not math.isclose(square(q_1).t, square(q_2).t):
        raise ValueError(f"Oops, the squares of these two are not equal: {square(q_1).t} != {square(q_2).t}")

    next_rotation = normalize(product(q_1, q_2, kind="odd"))

    # If the 2 quaternions point in exactly the same direction, the result is zero.
    # That is unacceptable for closure, so return the normalized vector_q of one input.
    # This does create some ambiguity since q and q_2 could point in exactly opposite
    # directions. In that case, the first quaternion is always chosen.
    v_norm = norm_squared_of_vector(next_rotation)

    if v_norm.t == 0:
        next_rotation = normalize(vector_q(q_1))

    return next_rotation


def Lorentz_next_boost(q_1: Qs, q_2: Qs) -> Qs:
    """
    Given 2 quaternions, creates a new quaternion to do a boost/rotation
    using the triple triple quaternion product
    by using the scalar_q of an even product to form (cosh(x), i sinh(x)).

    $ Lorentz_next_boost(q, q_2) = q q\_2 + q\_2 q

    Args:
        q_1: Qs
        q_2: Qs

    Returns: Qs

    """
    q_1.check_representations(q_2)

    if not (q_1.t >= 1.0 and q_2.t >= 1.0):
        raise ValueError(f"Oops, to be a boost, the first values must both be greater than one: {q_1.t},  {q_2.t}")

    if not math.isclose(square(q_1).t, square(q_2).t):
        raise ValueError(f"Oops, the squares of these two are not equal: {square(q_1).t} != {square(q_2).t}")

    q_even = product(q_1, q_2, kind="even")
    q_s = scalar_q(q_even)
    q_v = normalize(vector_q(q_even))

    if np.abs(q_s.t) > 1:
        q_s = inverse(q_s)

    exp_sum = product(add(exp(q_s), flip_signs(exp(q_s))), q1(1.0 / 2.0))
    exp_dif = product(dif(exp(q_s), flip_signs(exp(q_s))), q1(1.0 / 2.0))

    boost = add(product(exp_sum, q_v), exp_dif)

    return boost


# Lorentz transformations are not exclusively about special relativity.
# The most general case is B->B' such that the first term of scalar_q(B²)
# is equal to scalar_q(B'²). Since there is just one constraint yet there
# are 4 degrees of freedom, rescaling
def Lorentz_by_rescaling(q_1: Qs, op, h: Qs = None, quiet: bool = True) -> Qs:
    end_q_type = f"{q_1.q_type} Lorentz-by-rescaling"

    # Use h if provided.
    unscaled = op(h) if h is not None else op()

    q_1_interval = square(q_1).t
    unscaled_interval = unscaled.square().t

    # Figure out if the interval is time-like, space-like, or light-like (+, -, or 0)
    # if q_1_interval:
    #    if q_1_interval > 0:
    #        q_1_interval_type = "time-like"
    #    else:
    #        q_1_interval_type = "space-like"
    # else:
    #    q_1_interval_type = "light-like"

    # if unscaled_interval:
    #    if unscaled_interval > 0:
    #        unscaled_interval_type = "time-like"
    #    else:
    #        unscaled_interval_type = "space-like"
    # else:
    #    unscaled_interval_type = "light-like"

    q_1_interval_type = (
        "light-like" if unscaled_interval == 0 else "not_light-like"
    )
    unscaled_interval_type = (
        "light-like" if unscaled_interval == 0 else "not_light-like"
    )

    # My house rules after thinking about this rescaling stuff.
    # A light-like interval can go to a light-like interval.
    # Only a light-like interval can transform to the origin.
    # A light-like interval cannot go to a time- or space-like interval or visa versa.
    # If any of these exceptions are met, then an identity transformaton is returned - deepcopy(q1).
    # A time-like interval can rescale to a time-like or space-like (via an 'improper rescaling') interval.
    # A space-like interval can rescale to a time-like or space-like interval interval.

    # For light-like to light-like, no scaling is required. I don't think boosting makes sense to return q1
    if (q_1_interval_type == "light-like") and (
            unscaled_interval_type == "light-like"
    ):
        return unscaled

    # When one is light-like but the other is not, return a copy of the
    # starting value (an identity transformation).

    if (q_1_interval_type == "light-like") and (
            unscaled_interval_type != "light-like"
    ):
        return deepcopy(q_1)

    if (q_1_interval_type != "light-like") and (
            unscaled_interval_type == "light-like"
    ):
        return deepcopy(q_1)

    # The remaining case is to handle is if time-like goes to space-like
    # or visa-versa. Use a sign flip to avoid an imaginary value from the square root.
    if q_1.is_symbolic():
        sign_flip = False
    else:
        sign_flip = True if q_1_interval * unscaled_interval < 0 else False

    if sign_flip:
        scaling = np.sqrt(-1 * q_1_interval / unscaled_interval)
    else:
        scaling = np.sqrt(q_1_interval / unscaled_interval)

    if unscaled.equals(q0()):
        print("zero issue") if not quiet else 0
        return deepcopy(q_1)

    if not q_1.is_symbolic() and not np.isclose(scaling, 1):
        print(f"scaling needed: {scaling}") if not quiet else 0

    scaled = unscaled.product(Qs([scaling, 0, 0, 0]))
    scaled.print_state("final scaled") if not quiet else 0
    scaled.square().print_state("scaled square") if not quiet else 0
    scaled.q_type = end_q_type

    return scaled


# g_shift is a function based on the space-times-time invariance proposal for gravity,
# which proposes that if one changes the distance from a gravitational source, then
# squares a measurement, the observers at two different hieghts agree to their
# space-times-time values, but not the intervals.
# g_form is the form of the function, either minimal or exponential
# Minimal is what is needed to pass all weak field tests of gravity
def g_shift(q_1: Qs, dimensionless_g, g_form="exp"):
    """Shift an observation based on a dimensionless GM/c^2 dR."""

    end_q_type = f"{q_1.q_type} gshift"

    if g_form == "exp":
        g_factor = sp.exp(dimensionless_g)
    elif g_form == "minimal":
        g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
    else:
        print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
        return q_1

    g_q = Qs(q_type=end_q_type)
    g_q.t = q_1.t / g_factor
    g_q.x = q_1.x * g_factor
    g_q.y = q_1.y * g_factor
    g_q.z = q_1.z * g_factor
    g_q.q_type = end_q_type
    g_q.representation = q_1.representation

    return g_q


def sin(q_1: Qs) -> Qs:
    """
    Take the sine of a quaternion

    $ q.sin() = (\sin(t) \cosh(|R|), \cos(t) \sinh(|R|) R/|R|)$

    Returns: Qs

    """

    end_q_type = f"sin({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Qs([math.sin(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sint = math.sin(q_1.t)
    cost = math.cos(q_1.t)
    sinhR = math.sinh(abs_v.t)
    coshR = math.cosh(abs_v.t)

    k = cost * sinhR / abs_v.t

    q_sin = Qs()
    q_sin.t = sint * coshR
    q_sin.x = k * q_1.x
    q_sin.y = k * q_1.y
    q_sin.z = k * q_1.z

    q_sin.q_type = end_q_type
    q_sin.representation = q_1.representation

    return q_sin


def cos(q_1: Qs) -> Qs:
    """
    Take the cosine of a quaternion.
    $ q.cos() = (\cos(t) \cosh(|R|), \sin(t) \sinh(|R|) R/|R|) $

    Returns: Qs

    """

    end_q_type = f"cos({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Qs([math.cos(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sint = math.sin(q_1.t)
    cost = math.cos(q_1.t)
    sinhR = math.sinh(abs_v.t)
    coshR = math.cosh(abs_v.t)

    k = -1 * sint * sinhR / abs_v.t

    q_cos = Qs()
    q_cos.t = cost * coshR
    q_cos.x = k * q_1.x
    q_cos.y = k * q_1.y
    q_cos.z = k * q_1.z

    q_cos.q_type = end_q_type
    q_cos.representation = q_1.representation

    return q_cos


def tan(q_1: Qs) -> Qs:
    """
    Take the tan of a quaternion.

     $ q.tan() = \sin(q) \cos(q)^{-1} $

     Returns: Qs

    Args:
        q_1: Qs

     """

    end_q_type = f"tan({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Qs([math.tan(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sinq = sin(q_1)
    cosq = cos(q_1)

    q_tan = divide_by(sinq, cosq)
    q_tan.q_type = end_q_type
    q_tan.representation = q_1.representation

    return q_tan


def sinh(q_1: Qs) -> Qs:
    """
    Take the sinh of a quaternion.

    $ q.sinh() = (\sinh(t) \cos(|R|), \cosh(t) \sin(|R|) R/|R|) $

    Returns: Qs

    """

    end_q_type = f"sinh({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Qs([math.sinh(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sinh_t = math.sinh(q_1.t)
    cos_r = math.cos(abs_v.t)
    cosh_t = math.cosh(q_1.t)
    sin_r = math.sin(abs_v.t)

    k = cosh_t * sin_r / abs_v.t

    q_sinh = Qs(q_type=end_q_type, representation=q_1.representation)
    q_sinh.t = sinh_t * cos_r
    q_sinh.x = k * q_1.x
    q_sinh.y = k * q_1.y
    q_sinh.z = k * q_1.z

    return q_sinh


def cosh(q_1: Qs) -> Qs:
    """
    Take the cosh of a quaternion.

    $ (\cosh(t) \cos(|R|), \sinh(t) \sin(|R|) R/|R|) $

    Returns: Qs

    """

    end_q_type = f"cosh({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Qs([math.cosh(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    cosh_t = math.cosh(q_1.t)
    cos_r = math.cos(abs_v.t)
    sinh_t = math.sinh(q_1.t)
    sin_r = math.sin(abs_v.t)

    k = sinh_t * sin_r / abs_v.t

    q_cosh = Qs(q_type=end_q_type, representation=q_1.representation)
    q_cosh.t = cosh_t * cos_r
    q_cosh.x = k * q_1.x
    q_cosh.y = k * q_1.y
    q_cosh.z = k * q_1.z

    return q_cosh


def tanh(q_1: Qs) -> Qs:
    """
    Take the tanh of a quaternion.

    $ q.tanh() = \sin(q) \cos(q)^{-1} $

    Returns: Qs

    """

    end_q_type = f"tanh({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Qs([math.tanh(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sinhq = sinh(q_1)
    coshq = cosh(q_1)

    q_tanh = divide_by(sinhq, coshq)
    q_tanh.q_type = end_q_type
    q_tanh.representation = q_1.representation

    return q_tanh


def exp(q_1: Qs) -> Qs:
    """
    Take the exponential of a quaternion.

    $ q.exp() = (\exp(t) \cos(|R|, \exp(t) \sin(|R|) R/|R|) $

    Returns: Qs
    """

    end_q_type = f"exp({q_1.q_type})"

    abs_v = abs_of_vector(q_1)
    et = math.exp(q_1.t)

    if abs_v.t == 0:
        return Qs([et, 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    cosR = math.cos(abs_v.t)
    sinR = math.sin(abs_v.t)
    k = et * sinR / abs_v.t

    q_exp = Qs(
        [et * cosR, k * q_1.x, k * q_1.y, k * q_1.z],
        q_type=end_q_type,
        representation=q_1.representation,
    )

    return q_exp


def ln(q_1: Qs) -> Qs:
    """
    Take the natural log of a quaternion.

    $ q.ln() = (0.5 \ln t^2 + R.R, \atan2(|R|, t) R/|R|) $

    Returns: Qs

    """
    end_q_type = f"ln({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:

        if q_1.t > 0:
            return Qs([math.log(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)
        else:
            # I don't understand this, but Mathematica does the same thing.
            return Qs([math.log(-q_1.t), math.pi, 0, 0], q_type=end_q_type, representation=q_1.representation)

    t_value = 0.5 * math.log(q_1.t * q_1.t + abs_v.t * abs_v.t)
    k = math.atan2(abs_v.t, q_1.t) / abs_v.t

    q_ln = Qs(
        [t_value, k * q_1.x, k * q_1.y, k * q_1.z],
        q_type=end_q_type,
        representation=q_1.representation,
    )

    return q_ln


def q_2_q(q_1: Qs, q_2: Qs) -> Qs:
    """Take the natural log of a quaternion.

    $ q.q_2_q(p) = \exp(\ln(q) * p) $

    Returns: Qs

    """

    q_1.check_representations(q_2)
    end_q_type = f"{q_1.q_type}^{q_2.q_type}"

    q2q = exp(product(ln(q_1), q_2))
    q2q.q_type = end_q_type
    q2q.representation = q_1.representation
    q2q.q_type = end_q_type

    return q2q


def trunc(q_1: Qs) -> Qs:
    """
    Truncates values.

    Returns: Qs

    """

    if not q_1.is_symbolic():
        q_1.t = math.trunc(q_1.t)
        q_1.x = math.trunc(q_1.x)
        q_1.y = math.trunc(q_1.y)
        q_1.z = math.trunc(q_1.z)

    return q_1


class QsStates(Qs):
    """
    A class made up of many quaternions. It also includes values for rows * columns = dimension(QsStates).
    To mimic language already in wide use in linear algebra, there are qs_types of scalar_q, bra, ket, op/operator
    depending on the rows and column numbers.

    Quaternion states are a semi-group with inverses. A semi-group has more than one possible identity element. For
    quaternion states, there are $2^{dim}$ possible identities.
    """
    columns: int

    QS_TYPES = ["scalar_q", "bra", "ket", "op", "operator"]

    def __init__(self, qs=None, qs_type: str = "ket", rows: int = 0, columns: int = 0):

        super().__init__()
        self.qs = qs
        self.qs_type = qs_type
        self.rows = rows
        self.columns = columns

        if qs_type not in self.QS_TYPES:
            print(
                "Oops, only know of these quaternion series types: {}".format(
                    self.QS_TYPES
                )
            )

        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = int(len(qs)), int(len(qs)), int(len(qs))

        self.set_qs_type(qs_type, rows, columns, copy=False)

    def set_qs_type(self: QsStates, qs_type: str = "", rows: int = 0, columns: int = 0, copy: bool = True) -> QsStates:
        """
        Set the qs_type to something sensible.

        Args:
            qs_type: str:    can be scalar_q, ket, bra, op or operator
            rows: int        number of rows
            columns:         number of columns
            copy:

        Returns: QsStates

        """

        # Checks.
        if rows and columns and rows * columns != self.dim:
            raise ValueError(
                f"Oops, check those values again for rows:{rows} columns:{columns} dim:{self.dim}"
            )

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
                qs_type = "scalar_q"
            elif new_q.rows == 1:
                qs_type = "bra"
            elif new_q.columns == 1:
                qs_type = "ket"
            else:
                qs_type = "op"

        if not qs_type:
            raise Exception(
                "Oops, please set rows and columns for this quaternion series operator. Thanks."
            )

        if new_q.dim == 1:
            qs_type = "scalar_q"

        new_q.qs_type = qs_type

        return new_q

    def bra(self: QsStates) -> QsStates:
        """
        Quickly set the qs_type to bra by calling set_qs_type() with rows=1, columns=dim and taking a conjugate.

        Returns: QsStates

        """

        if self.qs_type == "bra":
            return self

        bra = deepcopy(self).conj()
        bra.rows = 1
        bra.columns = self.dim

        bra.qs_type = "bra" if self.dim > 1 else "scalar_q"

        return bra

    def ket(self: QsStates) -> QsStates:
        """
        Quickly set the qs_type to ket by calling set_qs_type() with rows=dim, columns=1 and taking a conjugate.

        Returns: QsStates

        """

        if self.qs_type == "ket":
            return self

        ket = deepcopy(self).conj()
        ket.rows = self.dim
        ket.columns = 1

        ket.qs_type = "ket" if self.dim > 1 else "scalar_q"

        return ket

    def op(self: QsStates, rows: int, columns: int) -> QsStates:
        """
        Quickly set the qs_type to op by calling set_qs_type().

        Args:
            rows: int:
            columns: int:

        Returns: QsStates

        """

        if rows * columns != self.dim:
            raise Exception(
                f"Oops, rows * columns != dim: {rows} * {columns}, {self.dimensions}"
            )

        op_q = deepcopy(self)

        op_q.rows = rows
        op_q.columns = columns

        if self.dim > 1:
            op_q.qs_type = "op"

        return op_q

    def __str__(self: QsStates, quiet: bool = False) -> str:
        """
        Print out all the states.

        Args:
            quiet: bool   Suppress printing the qtype.

        Returns: str

        """

        states = ""

        for n, q in enumerate(self.qs, start=1):
            states = states + f"n={n}: {q.__str__(quiet)}\n"

        return states.rstrip()

    def print_state(self: QsStates, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
        """
        Utility for printing states as a quaternion series.

        Returns: None

        """

        print(label)

        # Warn if empty.
        if self.qs is None or len(self.qs) == 0:
            raise ValueError("Oops, no quaternions in the series.")

        for n, q in enumerate(self.qs):
            print(f"n={n + 1}: {q.__str__(quiet)}")

        print(f"{self.qs_type}: {self.rows}/{self.columns}")

        if spacer:
            print("")

    def equals(self: QsStates, q_2: QsStates, scalar: bool = True, vector: bool = True) -> bool:
        """
        Test if two states are equal, state by state. Will compare the full quaternion
        unless either scalar_q or vector_q is set to false.

        Args:
            scalar: bool
            vector: bool
            q_2: QsStates   A quaternion state to compare with self.

        Returns: QsStates

        """

        if self.dim != q_2.dim:
            return False

        result = True

        for selfq, q_2q in zip(self.qs, q_2.qs):
            if not selfq.equals(q_2q, scalar, vector):
                result = False

        return result

    def conj(self: QsStates, conj_type: int = 0) -> QsStates:
        """
        Take the conjugates of states, default is zero, but also can do 1 or 2.

        Args:
            conj_type: int   0-3 for which one remains positive.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.conj(conj_type))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def conj_q(self: QsStates, q_2: QsStates) -> QsStates:
        """
        Does four conjugate operators on each state.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.conj_q(q_2))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def display_q(self: QsStates, label: str = "") -> None:
        """
        Try to display algebra in a pretty LaTeX way.

        Args:
            label: str   Text to decorate printout.

        Returns: None

        """

        if label:
            print(label)

        for i, ket in enumerate(self.qs, start=1):
            print(f"n={i}")
            ket.display_q()
            print("")

    def simple_q(self: QsStates) -> QsStates:
        """
        Simplify the states using sympy.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.simple_q())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def subs(self: QsStates, symbol_value_dict) -> QsStates:
        """
        Substitutes values into a symbolic expresion.

        Args:
            symbol_value_dict: Dict   {t: 3, x: 4}

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.subs(symbol_value_dict))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def scalar(self: QsStates) -> QsStates:
        """
        Returns the scalar_q part of a quaternion as a quaternion.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.scalar_q())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def vector(self: QsStates) -> QsStates:
        """
        Returns the vector_q part of a quaternion.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.vector_q())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def xyz(self: QsStates) -> List:
        """
        Returns the 3-vector_q for each state.

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.xyz())

        return new_states

    @staticmethod
    def q_0(dim: int = 1, qs_type: str = "ket") -> QsStates:
        """
        Return zero dim quaternion states.

        print(q_0(3))
        n=1: (0, 0, 0, 0) 0
        n=2: (0, 0, 0, 0) 0
        n=3: (0, 0, 0, 0) 0

        Args:
            dim: int
            qs_type: str

        Returns: QsStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(q0())

        return QsStates(new_states, qs_type=qs_type)


    @staticmethod
    def q_1(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> QsStates:
        """
        Return n * 1 dim quaternion states.

        print(q1(n, 3))
        n=1: (n, 0, 0, 0) 1
        n=2: (n, 0, 0, 0) 1
        n=3: (n, 0, 0, 0) 1
        Args:
            n: float    real valued
            dim: int
            qs_type: str

        Returns: QsStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(Qs().q_1(n))

        q1 = QsStates(new_states, qs_type=qs_type)

        return q1

    @staticmethod
    def q_i(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> QsStates:
        """
        Return n * i dim quaternion states.

        print(qi(3))
        n=1: (0, n, 0, 0) i
        n=2: (0, n, 0, 0) i
        n=3: (0, n, 0, 0) i

        Args:
            n: float    n times i
            dim: int
            qs_type: str

        Returns: QsStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(Qs().q_i(n))

        return QsStates(new_states, qs_type=qs_type)

    @staticmethod
    def q_j(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> QsStates:
        """
        Return n * j dim quaternion states.

        print(qj(3))
        n=1: (0, 0, n, 0) j
        n=2: (0, 0, n, 0) j
        n=3: (0, 0, n, 0) j

        Args:
            n: float
            dim: int
            qs_type: str

        Returns: QsStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(Qs().q_j(n))

        return QsStates(new_states, qs_type=qs_type)

    @staticmethod
    def q_k(n: float = 1, dim: int = 1, qs_type: str = "ket") -> QsStates:
        """
        Return n * k dim quaternion states.

        print(qk(3))
        n=1: (0, 0, 0, n) 0
        n=2: (0, 0, 0, n) 0
        n=3: (0, 0, 0, n) 0

        Args:
            n: float
            dim: int
            qs_type: str

        Returns: QsStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(Qs().q_k(n))

        q0 = QsStates(new_states, qs_type=qs_type)

        return q0

    @staticmethod
    def q_random(low: float = -1.0, high: float = 1.0, distribution: str = "uniform", dim: int = 1,
                 qs_type: str = "ket", q_type: str = "?", representation: str = "") -> QsStates:
        """
        Return a random-valued quaternion.
        The distribution is uniform, but one could add to options.
        It would take some work to make this clean so will skip for now.

        Args:
            low: float
            high: float
            distribution: str     have only implemented uniform distribution
            dim: int              number of states
            qs_type: str          bra/ket/op
            q_type: str           ?
            representation:       Cartesian by default

        Returns: QHState

        """

        new_states = []

        for _ in range(dim):
            new_states.append(Qs().q_random(low=low, high=high, distribution=distribution, q_type=q_type,
                                            representation=representation))

        qr = QsStates(new_states, qs_type=qs_type)

        return qr

    def flip_signs(self: QsStates) -> QsStates:
        """
        Flip signs of all states.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.flip_signs())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def norm(self: QsStates) -> QsStates:
        """
        Norm of states.

        Returns: QsStates

        """

        new_states = []

        for bra in self.qs:
            new_states.append(bra.norm())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def normalize(self: QsStates, n: float = 1.0, **kwargs) -> QsStates:
        """
        Normalize all states.

        Args:
            **kwargs:
            n: float   number to normalize to, default is 1.0

        Returns: QsStates

        """

        new_states = []

        zero_norm_count = 0

        for bra in self.qs:
            if bra.norm_squared().t == 0:
                zero_norm_count += 1
                new_states.append(q0())
            else:
                new_states.append(bra.normalize(n, ))

        new_states_normalized = []

        non_zero_states = self.dim - zero_norm_count

        for new_state in new_states:
            new_states_normalized.append(
                new_state.product(Qs([math.sqrt(1 / non_zero_states), 0, 0, 0]))
            )

        return QsStates(
            new_states_normalized,
            qs_type=self.qs_type,
            rows=self.rows,
            columns=self.columns,
        )

    def orthonormalize(self: QsStates) -> QsStates:
        """
        Given a quaternion series, returns an orthonormal basis.

        Returns: QsStates

        """

        last_q = self.qs.pop(0).normalize(math.sqrt(1 / self.dim), )
        orthonormal_qs = [last_q]

        for q in self.qs:
            qp = q.conj().product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1 / self.dim), )
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q

        return QsStates(
            orthonormal_qs, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def determinant(self: QsStates) -> QsStates:
        """
        Calculate the determinant of a 'square' quaternion series.

        Returns: QsStates

        """

        if self.dim == 1:
            q_det = self.qs[0]

        elif self.dim == 4:
            ad = self.qs[0].product(self.qs[3])
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
            raise ValueError("Oops, don't know how to calculate the determinant of this one.")

        return q_det

    def add(self: QsStates, ket: QsStates) -> QsStates:
        """
        Add two states.

        Args:
            ket: QsStates

        Returns: QsStates

        """

        if (self.rows != ket.rows) or (self.columns != ket.columns):
            error_msg = "Oops, can only add if rows and columns are the same.\n"
            error_msg += f"rows are {self.rows}/{ket.rows}, col: {self.columns}/{ket.columns}"
            raise ValueError(error_msg)

        new_states = []

        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(add(bra, ket))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def summation(self: QsStates) -> QsStates:
        """
        Add them all up, return one quaternion. Not sure if this ever is meaningful.

        Returns: QsStates

        """

        result = None

        for q in self.qs:
            if result is None:
                result = q
            else:
                result = result.add(q)

        return result

    def dif(self: QsStates, ket: QsStates) -> QsStates:
        """
        Take the difference of two states.

        Args:
            ket: QsStates

        Returns: QsStates

        """

        new_states = []

        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def diagonal(self: QsStates, dim: int) -> QsStates:
        """
        Make a state dim * dim with q or qs along the 'diagonal'. Always returns an operator.

        Args:
            dim: int

        Returns: QsStates

        """

        diagonal = []

        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        elif self.qs is None:
            raise ValueError("Oops, the qs here is None.")
        else:
            raise ValueError("Oops, need the length to be equal to the dimensions.")

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))
                else:
                    diagonal.append(q0())

        return QsStates(diagonal, qs_type="op", rows=dim, columns=dim)

    def trace(self: QsStates) -> QsStates:
        """
        Return the trace as a scalar_q quaternion series.

        Returns: QsStates

        """

        if self.rows != self.columns:
            raise ValueError(f"Oops, not a square quaternion series: {self.rows}/{self.columns}")

        else:
            trace = self.qs[0]

        for i in range(1, self.rows):
            trace = trace.add(self.qs[i * (self.rows + 1)])

        return QsStates([trace])

    @staticmethod
    def identity(dim: int = 1, operator: bool = False, additive: bool = False, non_zeroes=None, qs_type: str = "ket") \
            -> QsStates:
        """
        Identity operator for states or operators which are diagonal.

        Args:
            dim: int
            operator: bool
            additive: bool
            non_zeroes:
            qs_type: str

        Returns: QsStates

        """

        if additive:
            id_q = [q0() for _ in range(dim)]

        elif non_zeroes is not None:
            id_q = []

            if len(non_zeroes) != dim:
                print(
                    "Oops, len(non_zeroes)={nz}, should be: {d}".format(
                        nz=len(non_zeroes), d=dim
                    )
                )
                return QsStates([q0()])

            else:
                for non_zero in non_zeroes:
                    if non_zero:
                        id_q.append(q1())
                    else:
                        id_q.append(q0())

        else:
            id_q = [Qs().q_1() for _ in range(dim)]

        if operator:
            q_1 = QsStates(id_q)
            ident = QsStates.diagonal(q_1, dim)

        else:
            ident = QsStates(id_q, qs_type=qs_type)

        return ident

    def product(self: QsStates, q_2: QsStates, kind: str = "", reverse: bool = False) -> QsStates:
        """
        Forms the quaternion product for each state.

        Args:
            q_2: QsStates
            kind: str
            reverse: bool

        Returns: QsStates

        """

        self_copy = deepcopy(self)
        q_2_copy = deepcopy(q_2)
        qs_left, qs_right = QsStates(), QsStates()

        # Diagonalize if need be.
        if ((self.rows == q_2.rows) and (self.columns == q_2.columns)) or (
                "scalar_q" in [self.qs_type, q_2.qs_type]
        ):

            if self.columns == 1:
                qs_right = q_2_copy
                qs_left = self_copy.diagonal(qs_right.rows)

            elif q_2.rows == 1:
                qs_left = self_copy
                qs_right = q_2_copy.diagonal(qs_left.columns)

            else:
                qs_left = self_copy
                qs_right = q_2_copy

        # Typical matrix multiplication criteria.
        elif self.columns == q_2.rows:
            qs_left = self_copy
            qs_right = q_2_copy

        else:
            print(
                "Oops, cannot multiply series with row/column dimensions of {}/{} to {}/{}".format(
                    self.rows, self.columns, q_2.rows, q_2.columns
                )
            )

        # Operator products need to be transposed.
        operator_flag = False
        if qs_left in ["op", "operator"] and qs_right in ["op", "operator"]:
            operator_flag = True

        outer_row_max = qs_left.rows
        outer_column_max = qs_right.columns
        shared_inner_max = qs_left.columns
        projector_flag = (
                (shared_inner_max == 1) and (outer_row_max > 1) and (outer_column_max > 1)
        )

        result = [
            [Qs().q_0(q_type="") for _i in range(outer_column_max)]
            for _j in range(outer_row_max)
        ]

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

                    result[outer_row][outer_column] = result[outer_row][
                        outer_column
                    ].add(
                        qs_left.qs[left_index].product(
                            qs_right.qs[right_index], kind=kind, reverse=reverse
                        )
                    )

        # Flatten the list.
        new_qs = [item for sublist in result for item in sublist]
        new_states = QsStates(new_qs, rows=outer_row_max, columns=outer_column_max)

        if projector_flag or operator_flag:
            return new_states.transpose()

        else:
            return new_states

    def inverse(self: QsStates, additive: bool = False) -> QsStates:
        """
        Inversing bras and kets calls inverse() once for each.
        Inversing operators is more tricky as one needs a diagonal identity matrix.

        Args:
            additive: bool

        Returns: QsStates

        """

        if self.qs_type in ["op", "operator"]:

            if additive:

                q_flip = self.inverse(additive=True)
                q_inv = q_flip.diagonal(self.dim)

            else:
                if self.dim == 1:
                    q_inv = QsStates(self.qs[0].inverse())

                elif self.qs_type in ["bra", "ket"]:

                    new_qs = []

                    for q in self.qs:
                        new_qs.append(q.inverse())

                    q_inv = QsStates(
                        new_qs,
                        qs_type=self.qs_type,
                        rows=self.rows,
                        columns=self.columns,
                    )

                elif self.dim == 4:
                    det = self.determinant()
                    detinv = det.inverse()

                    q0 = self.qs[3].product(detinv)
                    q_2 = self.qs[1].flip_signs().product(detinv)
                    q2 = self.qs[2].flip_signs().product(detinv)
                    q3 = self.qs[0].product(detinv)

                    q_inv = QsStates(
                        [q0, q_2, q2, q3],
                        qs_type=self.qs_type,
                        rows=self.rows,
                        columns=self.columns,
                    )

                elif self.dim == 9:
                    det = self.determinant()
                    detinv = det.inverse()

                    q0 = (
                        self.qs[4]
                            .product(self.qs[8])
                            .dif(self.qs[5].product(self.qs[7]))
                            .product(detinv)
                    )
                    q_2 = (
                        self.qs[7]
                            .product(self.qs[2])
                            .dif(self.qs[8].product(self.qs[1]))
                            .product(detinv)
                    )
                    q2 = (
                        self.qs[1]
                            .product(self.qs[5])
                            .dif(self.qs[2].product(self.qs[4]))
                            .product(detinv)
                    )
                    q3 = (
                        self.qs[6]
                            .product(self.qs[5])
                            .dif(self.qs[8].product(self.qs[3]))
                            .product(detinv)
                    )
                    q4 = (
                        self.qs[0]
                            .product(self.qs[8])
                            .dif(self.qs[2].product(self.qs[6]))
                            .product(detinv)
                    )
                    q5 = (
                        self.qs[3]
                            .product(self.qs[2])
                            .dif(self.qs[5].product(self.qs[0]))
                            .product(detinv)
                    )
                    q6 = (
                        self.qs[3]
                            .product(self.qs[7])
                            .dif(self.qs[4].product(self.qs[6]))
                            .product(detinv)
                    )
                    q7 = (
                        self.qs[6]
                            .product(self.qs[1])
                            .dif(self.qs[7].product(self.qs[0]))
                            .product(detinv)
                    )
                    q8 = (
                        self.qs[0]
                            .product(self.qs[4])
                            .dif(self.qs[1].product(self.qs[3]))
                            .product(detinv)
                    )

                    q_inv = QsStates(
                        [q0, q_2, q2, q3, q4, q5, q6, q7, q8],
                        qs_type=self.qs_type,
                        rows=self.rows,
                        columns=self.columns,
                    )

                else:
                    raise ValueError("Oops, don't know how to invert.")

        else:
            new_states = []

            for bra in self.qs:
                new_states.append(bra.inverse(additive=additive))

            q_inv = QsStates(
                new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
            )

        return q_inv

    def divide_by(self: QsStates, ket: QsStates, additive: bool = False) -> QsStates:
        """
        Take a quaternion and divide it by another using an inverse. Can only handle up to 3 states.

        Args:
            ket: QsStates
            additive: bool

        Returns: QsStates

        """

        new_states = []

        ket_inv = ket.inverse(additive)

        for bra, k in zip(self.qs, ket_inv.qs):
            new_states.append(bra.product(k))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def triple_product(self: QsStates, ket: QsStates, ket_2: QsStates) -> QsStates:
        """
        A quaternion triple product of states.

        Args:
            ket: QsStates
            ket_2: QsStates

        Returns: QsStates

        """

        new_states = []

        for bra, k, k2 in zip(self.qs, ket.qs, ket_2.qs):
            new_states.append(bra.product(k).product(k2))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def rotate(self: QsStates, ket: QsStates) -> QsStates:
        """
        Rotate one state by another.

        Args:
            ket: QsStates

        Returns: QsStates

        """

        new_states = []

        for bra, k in zip(self.qs, ket.qs):
            new_states.append(bra.rotate(k))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def rotation_and_or_boost(self: QsStates, ket: QsStates) -> QsStates:
        """
        Do state-by-state rotations or boosts.

        Args:
            ket: QsStates

        Returns: QsStates

        """

        new_states = []

        for bra, k in zip(self.qs, ket.qs):
            new_states.append(bra.rotation_and_or_boost(k))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    @staticmethod
    def Lorentz_next_rotation(q: QsStates, q_2: QsStates) -> QsStates:
        """
        Does multiple rotations of a QHState given another QHState of equal dimensions.

        Args:
            q: QsStates
            q_2: QHStaes

        Returns:

        """

        if q.dim != q_2.dim:
            raise ValueError(
                "Oops, this tool requires 2 quaternion states with the same number of dimensions."
            )

        new_states = []

        for ket, q2 in zip(q.qs, q_2.qs):
            new_states.append(Qs.Lorentz_next_rotation(ket, q2))

        return QsStates(
            new_states, qs_type=q.qs_type, rows=q.rows, columns=q.columns
        )

    @staticmethod
    def Lorentz_next_boost(q: QsStates, q_2: QsStates) -> QsStates:
        """
        Does multiple boosts of a QHState given another QHState of equal dimensions.

        Args:
            q: QsStates
            q_2: QsStates

        Returns: QsStates

        """

        if q.dim != q_2.dim:
            raise ValueError(
                "Oops, this tool requires 2 quaternion states with the same number of dimensions."
            )

        new_states = []

        for ket, q2 in zip(q.qs, q_2.qs):
            new_states.append(Qs.Lorentz_next_boost(ket, q2))

        return QsStates(
            new_states, qs_type=q.qs_type, rows=q.rows, columns=q.columns
        )

    def g_shift(self: QsStates, g_factor: float = 1.0, g_form="exp") -> QsStates:
        """
        Do the g_shift to each state.

        Args:
            g_factor: float
            g_form: str

        Returns: QsStates

        """

        new_states = []

        for bra in self.qs:
            new_states.append(bra.g_shift(g_factor, g_form))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    @staticmethod
    def bracket(bra: QsStates, op: QsStates, ket: QsStates) -> QsStates:
        """
        Forms <bra|op|ket>. Note: if fed 2 kets, will take a conjugate.

        Args:
            bra: QsStates
            op: QsStates
            ket: QsStates

        Returns: QsStates

        """

        flip = 0

        if bra.qs_type == "ket":
            bra = bra.bra()
            flip += 1

        if ket.qs_type == "bra":
            ket = ket.ket()
            flip += 1

        if flip == 1:
            print("fed 2 bras or kets, took a conjugate. Double check.")

        b = bra.product(op).product(ket)

        return b

    @staticmethod
    def braket(bra: QsStates, ket: QsStates) -> QsStates:
        """
        Forms <bra|ket>, no operator. Note: if fed 2 kets, will take a conjugate.

        Args:
            bra: QsStates
            ket: QsStates

        Returns: QsStates

        """

        flip = 0

        if bra.qs_type == "ket":
            bra = bra.bra()
            flip += 1

        if ket.qs_type == "bra":
            ket = ket.ket()
            flip += 1

        if flip == 1:
            print("fed 2 bras or kets, took a conjugate. Double check.")

        else:
            print("Assumes your <bra| already has been conjugated. Double check.")

        b = bra.product(ket)

        return b

    def op_q(self: QsStates, q: Qs, first: bool = True, kind: str = "", reverse: bool = False) -> QsStates:
        """
        Multiply an operator times a quaternion, in that order. Set first=false for n * Op

        Args:
            q: Qs
            first: bool
            kind: str
            reverse: bool

        Returns: QsStates

        """

        new_states = []

        for op in self.qs:

            if first:
                new_states.append(op.product(q, kind, reverse))

            else:
                new_states.append(q.product(op, kind, reverse))

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def square(self: QsStates) -> QsStates:
        """
        The square of each state.

        Returns: QsStates

        """

        new_states = []

        for bra in self.qs:
            new_states.append(bra.square())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def norm_squared(self: QsStates) -> QsStates:
        """
        Take the inner product, returning a scalar_q series.

        Returns: QsStates

        """

        norm_scalar = self.set_qs_type("bra").conj().product(self.set_qs_type("ket"))

        return norm_scalar

    def norm_squared_of_vector(self: QsStates) -> QsStates:
        """
        Take the inner product of the vector_q, returning a scalar_q.

        Returns: QsStates

        """

        vector_norm_scalar: QsStates = self.set_qs_type("bra").vector().conj().product(self.set_qs_type("ket").vector())

        return vector_norm_scalar

    def transpose(self: QsStates, m: int = None, n: int = None) -> QsStates:
        """
        Transposes a series.

        Args:
            m: int
            n: int

        Returns: QsStates

        """

        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m

        if n is None:
            n = int(self.dim / m)

        matrix = [[0 for _x in range(m)] for _y in range(n)]

        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = self.qs[mi * n + ni]

        qs_t = []

        for t in matrix:
            for q in t:
                qs_t.append(q)

        # Switch rows and columns.
        return QsStates(qs_t, rows=self.columns, columns=self.rows)

    def Hermitian_conj(self: QsStates, m: int = None, n: int = None, conj_type: int = 0) -> QsStates:
        """
        Returns the Hermitian conjugate.

        Args:
            m: int
            n: int
            conj_type: int    0-3

        Returns: QsStates

        """

        return self.transpose(m, n).conj(conj_type)

    def dagger(self: QsStates, m: int = None, n: int = None, conj_type: int = 0) -> QsStates:
        """
        Just calls Hermitian_conj()

        Args:
            m: int
            n: int
            conj_type: 0-3

        Returns: QsStates

        """

        return self.Hermitian_conj(m, n, conj_type)

    def is_square(self: QsStates) -> bool:
        """
        Tests if a quaternion series is square, meaning the dimenion is n^2.

        Returns: bool

        """

        return math.sqrt(self.dim).is_integer()

    def is_Hermitian(self: QsStates) -> bool:
        """
        Tests if a series is Hermitian.

        Returns: bool

        """

        hc = self.Hermitian_conj()

        return self.equals(hc)

    @staticmethod
    def sigma(kind: str = "x", theta: float = None, phi: float = None) -> QsStates:
        """
        Returns a sigma when given a type like, x, y, z, xy, xz, yz, xyz, with optional angles theta and phi.

        Args:
            kind: str  x, y, z, xy, etc
            theta: float   an angle
            phi: float     an angle

        Returns:

        """

        q0, q_2, qi = q0(), q1(), qi()

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

        x_factor = q_2.product(Qs([sin_theta * cos_phi, 0, 0, 0]))
        y_factor = qi.product(Qs([sin_theta * sin_phi, 0, 0, 0]))
        z_factor = q_2.product(Qs([cos_theta, 0, 0, 0]))

        sigma = Bunch()
        sigma.x = QsStates([q0, x_factor, x_factor, q0], "op")
        sigma.y = QsStates([q0, y_factor, y_factor.flip_signs(), q0], "op")
        sigma.z = QsStates([z_factor, q0, q0, z_factor.flip_signs()], "op")

        sigma.xy = sigma.x.add(sigma.y)
        sigma.xz = sigma.x.add(sigma.z)
        sigma.yz = sigma.y.add(sigma.z)
        sigma.xyz = sigma.x.add(sigma.y).add(sigma.z)

        if kind not in sigma:
            raise ValueError("Oops, I only know about x, y, z, and their combinations.")

        return sigma[kind].normalize()


    def sin(self: QsStates) -> QsStates:
        """
        sine of states.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.sin())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def cos(self: QsStates) -> QsStates:
        """
        cosine of states.

        Returns:

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.cos())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def tan(self: QsStates) -> QsStates:
        """
        tan() of states.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.tan())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def sinh(self: QsStates) -> QsStates:
        """
        sinh() of states.

        Returns:

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.sinh())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def cosh(self: QsStates) -> QsStates:
        """
        cosh() of states.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.cosh())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def tanh(self: QsStates) -> QsStates:
        """
        tanh() of states.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.tanh())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def exp(self: QsStates) -> QsStates:
        """
        exponential of states.

        Returns: QsStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.exp())

        return QsStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )
