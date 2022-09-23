# system imports
import argparse

# third party imports
from pyscf import gto, scf, ao2mo
import numpy as np

# local imports
from Hartree_Fock import check_integral


def extract_Hamiltonian_parameters(mo_flag, mean_field, mol_HF):
    """ extract 1-electron integral Hamiltonian parameters """

    # atomic orbitals
    h_core_AO = mol_HF.intor('int1e_kin_sph') + mol_HF.intor('int1e_nuc_sph')
    print("1-electron integral (in AO basis):\n{:}".format(h_core_AO.shape))

    if not mo_flag:
        return h_core_AO

    # molecular orbitals
    h_core_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, h_core_AO, mean_field.mo_coeff)
    print("1-electron integral (in MO basis):\n{:}".format(h_core_MO.shape))
    return h_core_MO


def calculate_overlap_matrix(mo_flag, mean_field, mol_HF):
    """ x """

    # atomic orbitals
    S_AO = mol_HF.intor('int1e_ovlp_sph')

    if not mo_flag:
        return S_AO

    # molecular orbitals
    S_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, S_AO, mean_field.mo_coeff)
    return S_MO


def calculate_electron_repulsion_integrals(mo_flag, mean_field, molecular_HF):
    """ 2-electron integral """

    # atomic orbitals
    eri_AO = molecular_HF.intor('int2e_sph', aosym=1)
    print("2-electron integral (in AO basis):\n{:}".format(eri_AO.shape))

    if not mo_flag:
        return eri_AO

    # molecular orbitals
    eri_MO = ao2mo.incore.full(eri_AO, mean_field.mo_coeff)
    print("2-electron integral (in MO basis):\n{:}".format(eri_AO.shape))

    return eri_MO


def calculate_fock_matrix(mo_flag, mean_field, molecular_HF):
    """ """

    # atomic orbitals
    fock_AO = mean_field.get_fock()
    print("Fock matrix (in AO basis):\n{:}".format(fock_AO.shape))

    if not mo_flag:
        return fock_AO

    # molecular orbitals
    fock_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, fock_AO, mean_field.mo_coeff)
    print("Fock matrix (in MO basis):\n{:}".format(fock_MO.shape))
    return fock_MO


def prepare_parsed_arguments():
    """ Wrapper for argparser setup """

    formatclass = argparse.ArgumentDefaultsHelpFormatter

    # parse the arguments
    parser = argparse.ArgumentParser(description="test against HF", formatter_class=formatclass)
    parser.add_argument(
        '--basis',
        type=str,
        default='ao',
        metavar='HF basis',
        help='The type of basis to use for calculating the HF energy. Should be either ao or mo'
    )

    # initialize
    pargs = parser.parse_args()

    # error checking
    assert pargs.basis == 'ao' or pargs.basis == 'mo', f"Supply either 'ao' or 'mo' not {pargs.basis}"

    return pargs


def main():
    """ x """

    # process the users input
    pargs = prepare_parsed_arguments()

    mo_flag = bool(pargs.basis == 'mo')

    # geometry of molecules (in Angstrom)
    HF = 'H 0 0 0; F 0 0 1.1'

    H2O = '''
    O 0 0      0
    H 0 -2.757 2.587
    H 0  2.757 2.587'''

    atom = HF
    molecule = "HF"

    # setup model input using gaussian-t-orbitals?
    molecular_HF = gto.M(
        atom=atom,  # in Angstrom
        basis='ccpvdz',
        symmetry=1,
    )

    # run HF calculate
    mean_field = scf.HF(molecular_HF)
    mean_field.kernel()

    h_core = extract_Hamiltonian_parameters(mo_flag, mean_field, molecular_HF)

    S_matrix = calculate_overlap_matrix(mo_flag, mean_field, molecular_HF)

    er_integral = calculate_electron_repulsion_integrals(mo_flag, mean_field, molecular_HF)

    fock_matrix = calculate_fock_matrix(mo_flag, mean_field, molecular_HF)

    # energy expectation value (HF energy)
    E_Hartree_Fock = mean_field.e_tot
    print("energy expectation value (HF energy) (in Hartree):{:.5f}".format(E_Hartree_Fock))

    # total number of electrons
    OccupationNumber = mean_field.mo_occ / 2
    nof_electron = sum(OccupationNumber)
    print("total number of electrons: {:}".format(nof_electron))
    print("occupation number:\n{:}".format(OccupationNumber))

    # get NuclearRepulsionEnergy
    NR_energy = mean_field.energy_nuc()

    # run TFCC & thermal NOE calculation
    model = check_integral(
        E_Hartree_Fock,
        h_core,
        fock_matrix,
        er_integral,
        nof_electron,
        S_matrix,
        OccupationNumber,
        NR_energy,
        molecule=molecule,
        mo_basis=mo_flag
    )

    # thermal field transform
    model.my_SCF()


if (__name__ == '__main__'):
    main()
