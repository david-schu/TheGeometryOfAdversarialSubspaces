#!/usr/bin/env python
# -*- coding: utf8 -*-

"""subspace_utils.py: Utilities for the implementation of the 
'Gradient Aligned Adversarial Subspace' method in 
'The Space of Transferable Adversarial Examples' by Florian Tramèr, 
Nicolas Papernot, Ian Goodfellow, Dan Boneh and Patrick McDaniel."""

__author__      = "Florian Tramèr, Ian Goodfellow"

import numpy as np
rng = np.random.RandomState([2017, 3, 1])


def random_rotation_matrix(g):
    """
    Returns a rotation matrix that rotates g to occupy only the first
    component of the resulting vector.
    Because many such rotations are possible, generates a random rotation
    matrix.
    """

    assert len(g.shape) == 1
    n = g.shape[0]
    Q = np.zeros((n, n))
    Q[0, :] = g / np.sqrt(np.dot(g, g))
    for i in range(1, n):
        q = rng.randn(n)
        # If we orthogonalize more than once, reduce numerical error
        for reps in range(2):
            for j in range(i):
                u = Q[j, :]
                q = q - np.dot(q, u) * u
                q = q - np.dot(q, u) * u
            q = q / np.sqrt(np.dot(q, q))
            Q[i, :] = q
    return Q

def raw_attack(Q, g, grad_coeff, num_attacks):
    """
    The simplest attack interface: find attacks that have norm one and put the desired
    coefficient along the gradient.

    g: gradient vector for a single example to attack
    adv_coeff: the component of the perturbation that should be aligned with g
    num_attacks may be up to g.shape[0]

    Returns a minibatch of num_attacks orthogonal perturbations
    """
    assert len(g.shape) == 1
    assert isinstance(grad_coeff, float)
    if np.square(grad_coeff) > 1.:
        raise ValueError("Non-adversarial components don't have enough mass to "
                         "make the attacks orthogonal.")
    assert isinstance(num_attacks, int)
    n = g.shape[0]

    assert num_attacks < n

    out = np.zeros((num_attacks, n))
    out[:, 0] = grad_coeff
    neutralizer = np.identity(num_attacks)
    neutralizer -= 1. / num_attacks

    target_norm = np.sqrt(1 - np.square(grad_coeff))
    actual_norm = np.sqrt((num_attacks - 1.) / num_attacks)

    if target_norm != actual_norm:
        neutralizer *= target_norm / actual_norm

    out[:, 1:1 + num_attacks] = neutralizer

    out = np.dot(out, Q)

    return out

def ortho_attack(Q, g, desired_grad_coeff):
    """
    Wraps raw_attack to adjust the grad coeff and number of attacks to guarantee orthogonality.
    Attacks still have norm 1.
    """
    fractional_num_attacks = 1. / np.square(desired_grad_coeff)
    num_attacks = int(np.floor(fractional_num_attacks))
    grad_coeff = 1. / np.sqrt(num_attacks)
    # print("Desired, actual: ", desired_grad_coeff, grad_coeff)
    return raw_attack(Q, g, grad_coeff, num_attacks)

def test_Q_unit_norm():
    dim = 10
    g = rng.randn(dim)
    Q = random_rotation_matrix(g)
    row_norms = np.sqrt(np.square(Q).sum(axis=1))
    assert np.allclose(row_norms, np.ones(dim))

def test_Q_projects_g_to_first():
    dim = 10
    g = rng.randn(dim)
    Q = random_rotation_matrix(g)
    r = np.dot(Q, g)
    assert np.allclose(r[0], np.sqrt(np.square(g).sum()))
    if not np.allclose(r[1:], 0. * r[1:]):
        print(r[1:])
        raise ValueError()

def test_Q_is_orthogonal():
    dim = 10
    g = rng.randn(dim)
    Q = random_rotation_matrix(g)
    prod = np.dot(Q.T, Q)
    assert np.allclose(prod, np.identity(dim))

def test_attack_unit_norm():
    dim = 10
    g = rng.randn(dim)
    Q = random_rotation_matrix(g)
    grad_coeff = .1
    etas = raw_attack(Q, g, grad_coeff, dim - 1)
    norms = np.sqrt(np.square(etas).sum(axis=1))
    assert np.allclose(norms, 1), norms

def test_attack_has_right_grad_coeff():
    dim = 10
    g = rng.randn(dim)
    u = g / np.sqrt(np.square(g).sum())
    Q = random_rotation_matrix(g)
    grad_coeff = .1
    etas = raw_attack(Q, g, grad_coeff, dim - 1)
    prods = np.dot(etas, u)
    assert np.allclose(prods, grad_coeff), prods

def test_attacks_ortho():
    dim = 10
    g = rng.randn(dim)
    Q = random_rotation_matrix(g)
    desired_grad_coeff = 0.5
    etas = ortho_attack(Q, g, desired_grad_coeff)
    prod = np.dot(etas, etas.T)
    assert np.allclose(prod, np.identity(etas.shape[0])), prod

if __name__ == "__main__":
    test_Q_unit_norm()
    test_Q_projects_g_to_first()
    test_Q_is_orthogonal()
    test_attack_unit_norm()
    test_attack_has_right_grad_coeff()
    test_attacks_ortho()
