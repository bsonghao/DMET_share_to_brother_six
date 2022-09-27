# system imports
from os.path import abspath, join, dirname
import sys

# third party imports
from pyscf import gto, scf, ao2mo
import numpy as np

# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/DMET_share_to_brother_six/'))
sys.path.insert(0, project_dir)

# local import
from project.extract_Hamiltonian_parameters import *
from project.Hartree_Fock import *

def main():
    """ main function to run DMET calculation  """
    # Boolean flags
    mo_flag = True
    debug_flag = False

    # geometry of molecules (in Angstrom)
    HF = 'H 0 0 0; F 0 0 1.1'

    H2O = '''
    O 0 0      0
    H 0 -2.757 2.587
    H 0  2.757 2.587'''

    atom = HF
    molecule = "HF"

    # setup model input using gaussian-type-orbitals
    molecular_HF = gto.M(
        atom=atom,  # in Angstrom
        basis='ccpvdz',
        symmetry=1,
    )

    # run HF calculate from PySCF
    mean_field = scf.HF(molecular_HF)
    mean_field.kernel()

    # extract Hamiltonian parameters from PySCF input
    h_core = extract_Hamiltonian_parameters(mo_flag, mean_field, molecular_HF)
    S_matrix = calculate_overlap_matrix(mo_flag, mean_field, molecular_HF)
    er_integral = calculate_electron_repulsion_integrals(mo_flag, mean_field, molecular_HF)
    fock_matrix = calculate_fock_matrix(mo_flag, mean_field, molecular_HF)

    # extract information from Hartree Fock calculations of PySCF
    ## energy expectation value (HF energy)
    E_Hartree_Fock = mean_field.e_tot
    print("energy expectation value (HF energy) (in Hartree):{:.5f}".format(E_Hartree_Fock))

    ## total number of electrons
    OccupationNumber = mean_field.mo_occ / 2
    nof_electron = sum(OccupationNumber)
    print("total number of electrons: {:}".format(nof_electron))
    print("occupation number:\n{:}".format(OccupationNumber))

    ## get NuclearRepulsionEnergy
    NR_energy = mean_field.energy_nuc()

    ## get 1-electron reduced density matrix
    RDM_1 = calculate_1_electron_density_matrix(mean_field)


    if debug_flag:
        # compare PySCF HF calculation to my naive HF calculation
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
        model.my_SCF()


if (__name__ == '__main__'):
    main()
