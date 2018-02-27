#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
#Daniel Chang
#2260161
#Chang254@mail.chapman.edu
#Myranda Hoggatt
#2285495
#hogga102@mail.chapman.edu
#Devon Ball
#2313078
#dball@chapman.edu
#PHYS220 Spring 2018
#CW04 Exercise 3
###

"""Classwork 04 Test Functions
This suite of functions tests the functionality of the CW04 solutions.
"""

import nose
import numpy as np
import cw04

def test_gauss_list():
    """test_gauss_list()
    Tests whether Gaussian values are correct for domain points -1, 0, and 1,
    using the reference list implementation.
    """
    x,g = cw04.gen_gaussian_list(-1,1,3)
    desired = [0.24197072451914337, 0.3989422804014327, 0.24197072451914337]
    print("Obtained:",g)
    print("Desired:",desired)
    # For comparing floating point values, nose has useful helper functions
    # to ensure they are equal up to a numerical precision tolerance
    nose.tools.assert_almost_equal(g, desired)

def test_gauss_array():
    """test_gauss_array()
    Tests whether Gaussian values are correct for domain points -1, 0, and 1,
    using the numpy array implementation.
    """
    x,g = cw04.gen_gaussian_array(-1,1,3)
    desired = np.array([0.24197072451914337, 0.3989422804014327, 0.24197072451914337])
    print("Obtained:",g)
    print("Desired:",desired)
    # Numpy has built-in testing functions to iterate over arrays and compare
    # values up to certain tolerances
    np.testing.assert_almost_equal(g, desired)

def test_sinc_list():
    """test_sinc_list()
    Tests whether sinc(x) values are correct for domain points -1, 0, and 1,
    using the reference list implementation.
    """
    x,sc = cw04.gen_sinc_list(-1,1,3)
    desired = [0.8414709848078965, 1, 0.8414709848078965]
    print("Obtained:",sc)
    print("Desired:",desired)
    # For comparing floating point values, nose has useful helper functions
    # to ensure they are equal up to a numerical precision tolerance
    nose.tools.assert_almost_equal(sc, desired)

def test_sinc_array():
    """test_sinc_array()
    Tests whether sinc values are correct for domain points -1, 0, and 1,
    using the numpy array implementation.
    """
    x,sf = cw04.gen_sinc_array(-1,1,3)
    desired = [0.8414709848078965, 1, 0.8414709848078965]
    print("Obtained:",sf)
    print("Desired:",desired)
    # For comparing floating point values, nose has useful helper functions
    # to ensure they are equal up to a numerical precision tolerance
    nose.tools.assert_almost_equal(sf, desired)


def test_sinf_list():
    """test_sinf_list()
    Tests whether sinf(x) values are correct for domain points -1, 0, and 1,
    using the reference list implementation.
    """
    x,sf = cw04.gen_sinc_list(-1,1,3)
    desired = [-0.8414709848078965, 1, 0.8414709848078965]
    print("Obtained:",sf)
    print("Desired:",desired)
    # For comparing floating point values, nose has useful helper functions
    # to ensure they are equal up to a numerical precision tolerance
    nose.tools.assert_almost_equal(sf, desired)

def test_sinf_array():
    """test_sinf_array()
    Tests whether sinf(x) values are correct for domain points -1, 0, and 1,
    using the numpy array implementation.
    """
    x,sf = cw04.gen_sinf_array(-1,1,3)
    desired = [-0.8414709848, 1, 0.8414709848]
    print("Obtained:",sf)
    print("Desired:",desired)
    # For comparing floating point values, nose has useful helper functions
    # to ensure they are equal up to a numerical precision tolerance
    nose.tools.assert_almost_equal(sf, desired)

