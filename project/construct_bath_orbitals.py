class Constuct_Bath(object):
    """
    Construct_Bath object construct DMET bath orbitals based on low level
    quantum chemistry (mean field) calculation
    """
    def __init__(self, rdm1, impurity_list):
        """
        rdm1: reduced 1-electron density matrix get from mean field calculation
        impurity_list: labels of fragment orbitals (manually defined)
        """
        self.rdm1 = rdm1
        self.impurity_list = impurity_list

    def get_bath(self):
        """construct DMET bath"""
        return 
