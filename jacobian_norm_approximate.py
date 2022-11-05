"""Tensorflow implementation of
   https://github.com/facebookresearch/jacobian_regularizer/blob/main/jacobian/jacobian.py

   Author: Suyong Choi (suyong@korea.ac.kr)
   In tensorflow, batch_jacobian could be used if dimension isn't high.
"""

import tensorflow as tf

def random_vector(B, C):
    """Creates a random vector of dimension C with a norm of 1

    Args:
        B (int): batch size
        C (int): dimension feature output
    """
    v = tf.random.normal(shape = [B, C])
    vnorm = tf.norm(v, axis = 1, keepdims=True)
    return v/vnorm


def geomcomplexity_approx(z, x, nproj=1):
    """Calculate approximate jacobian according to arXiv:1908.02729 Algorithm 1
       and return the square of Frobenius norm of the Jacobian

    Args:
        x (tensor): feature inputs
        z (tensor): feature outputs
    """
    assert(nproj>0)
    zshape = tf.shape(z)
    C = zshape[1]
    B = zshape[0]
    Cfloat = tf.cast(C, tf.float32)
    Bfloat = tf.cast(B, tf.float32)
    J2 = 0.0

    for _ in range(nproj):
        v = random_vector(B, C)
        Jv = tf.gradients(tf.reduce_sum(z*v), x)
        J2 += Cfloat* tf.norm(Jv)**2 / (nproj*Bfloat)
    
    return J2


