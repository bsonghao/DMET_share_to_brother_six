# third party imports
from pyscf import gto, scf, ao2mo
import numpy as np

def extract_Hamiltonian_parameters(mo_flag, mean_field, mol_HF):
    """ extract 1-electron integral Hamiltonian parameters """
    # 1-electron integral in AO basis
    h_core_AO = mol_HF.intor('int1e_kin_sph') + mol_HF.intor('int1e_nuc_sph')
    print("1-electron integral (in AO basis):\n{:}".format(h_core_AO.shape))

    if not mo_flag:
        return h_core_AO

    # 1-electron integral in MO basis
    h_core_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, h_core_AO, mean_field.mo_coeff)
    print("1-electron integral (in MO basis):\n{:}".format(h_core_MO.shape))
    return h_core_MO


def calculate_overlap_matrix(mo_flag, mean_field, mol_HF):
    """ extract overlap matrix from PySCF Hartree Fock calculations """
    # overlap matrix in AO basis
    S_AO = mol_HF.intor('int1e_ovlp_sph')
    if not mo_flag:
        return S_AO

    # overlap matrix in MO basis
    S_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, S_AO, mean_field.mo_coeff)
    return S_MO


def calculate_electron_repulsion_integrals(mo_flag, mean_field, molecular_HF):
    """ extract 2-electron integrals from PySCF input"""
    # 2-electron integral in AO basis
    eri_AO = molecular_HF.intor('int2e_sph', aosym=1)
    print("2-electron integral (in AO basis):\n{:}".format(eri_AO.shape))

    if not mo_flag:
        return eri_AO

    # 2-electron integral in MO basis
    eri_MO = ao2mo.incore.full(eri_AO, mean_field.mo_coeff)
    print("2-electron integral (in MO basis):\n{:}".format(eri_AO.shape))

    return eri_MO


def calculate_fock_matrix(mo_flag, mean_field, molecular_HF):
    """ extract converge Fock matrix from Hartree Fock calculation in PySCF """
    # Fock matrix in AO basis
    fock_AO = mean_field.get_fock()
    print("Fock matrix (in AO basis):\n{:}".format(fock_AO.shape))

    if not mo_flag:
        return fock_AO

    # Fock matrix in MO basis
    fock_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, fock_AO, mean_field.mo_coeff)
    print("Fock matrix (in MO basis):\n{:}".format(fock_MO.shape))
    return fock_MO


def calculate_1_electron_density_matrix(mo_flag, mean_field):
    """ extract one electronic reduced density matrix from PySCF HF calculation """
    # 1 RDM in AO basis
    RDM_1_AO = mean_field.make_rdm1()
    print("1-rdm(in AO basis):\n{:}".format(RDM_1_AO.shape))

    if mo_flag:
        # 1-rdm matrix in MO basis
        RDM_1_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, RDM_1_AO, mean_field.mo_coeff)
        print("1 rdm (in MO basis):\n{:}".format(RDM_1_MO.shape))
        return RDM_1_MO
