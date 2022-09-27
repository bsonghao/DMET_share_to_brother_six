#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD calculation.
'''
#geometry of molecules (in Angstrom)
HF = 'H 0 0 0; F 0 0 1.1'

H2O = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''

import pyscf

mol = pyscf.M(
    atom = HF,
    basis = 'ccpvdz')

mf = mol.RHF().run()

mycc = mf.CCSD().run()
print('CCSD correlation energy', mycc.e_corr)
