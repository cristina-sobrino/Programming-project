"""
Created on Thu Oct 17 18:04:26 2024

@author: Cristina Sobrino
"""
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

import time
import numpy as np
from numpy import linalg


#···································
#······  READING THE INPUT   ·······
#···································


name = input("Please, write the name of the file:")


def read_file(archivo):
    """This function takes the input name and extracts the coordinates and a list of the atoms connected by bonds.

    Inputs:
        archivo (String): it is the name of the input file

    Returns:
        coord_vector (List[float]): A 1D vector of the coordinates as (x1,y1,z1,x2,y2,z3,...)
        atom_vector (List[int]): A 1D vector of the atom indices
        bond_matrix (List[List[Int]]): A 2D array of the bond indices
    """

    with open(archivo, "r", encoding="utf-8") as file:
        next(file)
        lines = file.readlines()

        bond_matrix= []
        coord_vector = []
        atom_vector = []

        for i in lines:
            parts = i.strip().split()

            if len(parts) > 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    if not parts[3].isdigit(): #to differenciate the coordinates from the bond matrix we only get the coordinates that have letter after them
                        atoms_i = str(parts[3])
                        coord_vector.append(x)
                        coord_vector.append(y)
                        coord_vector.append(z)
                        atom_vector.append(atoms_i)

                    else:
                        #the bond matrix is the one that only has numbers and not strings.
                        bond_matrix.append([parts[0], parts[1]])

                except ValueError:
                    continue


        return coord_vector, atom_vector, bond_matrix

coord_m, atoms, bonds_m = read_file(name)


#···································
#······  BOND CALCULATION    ·······
#···································

def bond_analysis(bonds, coord):
    """This function takes the bond vector and computes the length of the bonds.

    Inputs:
        bonds (List[float]): A list of bonds where each bond is defined by indices of the two atoms involved
        coord (List[float]): A 1D vector of the coordinates as (x1,y1,z1,x2,y2,z3,...)

    Returns:
        bond_eq_length (List[float]): equilibrium bond length for each bond
        force_cte (List[float]): force constants for each bond
        vector_rba (List[float]): bond vectors between atom pairs
        bond_input_lengths (List[float]): bond length at current structure

    """

    bond_eq_length = [] ; vector_rba_x = [] ; vector_rba_y = [] ; vector_rba_z = [] ; vector_rba = []
    bonds2 = []
    r = 0
    i = 0
    h = 0
    bond = 0
    force_cte = []
    bond_input_lengths = []

    #First, we create a new list of the bonds, but substituting the atom number (that depend on the input) with the atom symbol so the bond is identified uniquely.
    for i in range(len(bonds)):
        a1 = int(bonds[i][0])
        bonds2.append(atoms[a1-1])
        a2 = int(bonds[i][1])
        bonds2.append(atoms[a2-1])
    #now bonds2 has a vector form so is easier to iterate: ['C', 'C', 'C', 'H', 'C', 'H']

    #the second step is to compute the actual bonds in the input
    while bond < len(bonds):

        atom1 = (int(bonds[bond][0]) - 1)*3 #because python starts to count at zero but mol2 file does not
        atom2 = (int(bonds[bond][1]) - 1)*3 #because each atom has three coordinates
        r_ab = np.sqrt( ((float(coord[atom1])) - float(coord[atom2]))**2 + (float(coord[atom1+1]) - float(coord[atom2+1]))**2 + (float(coord[atom1+2]) - float(coord[atom2+2]))**2 )
        bond_input_lengths.append(r_ab)

        #this will be usefull for later
        r_ba_x = float(coord[atom1]) - float(coord[atom2])
        vector_rba_x.append(r_ba_x)

        r_ba_y = float(coord[atom1+1]) - float(coord[atom2+1])
        vector_rba_y.append(r_ba_y)

        r_ba_z = float(coord[atom1+2]) - float(coord[atom2+2])
        vector_rba_z.append(r_ba_z)

        vector_rba.append([[r_ba_x], [r_ba_y], [r_ba_z]])

        bond += 1
    #from this loop we get a list of the input bond lengths

    #the third step is to create a new list but in this case with the bond length.
    while h < len(bonds2):
        if bonds2[h] and bonds2[h+1] == 'C':
            r = 1.53
            kb = 300.
        if bonds2[h] == 'C' and bonds2[h+1] == 'H':
            r = 1.11
            kb = 350.
        if bonds2[h+1] == 'C' and bonds2[h] == 'H':
            r = 1.11
            kb = 350.

        h += 2 #because a bond always involve two atoms!
        bond_eq_length.append(r)
        force_cte.append(kb)

    return bond_eq_length, force_cte, bond_input_lengths, vector_rba

eq_bonds,kb_list, real_bonds, v_rba = bond_analysis(bonds_m,coord_m)

bond_vector = []
#the bond vector has a form that is [1,2,1,3,1,4,...]

for i in range(len(bonds_m)):
    a = bonds_m[i][0]
    b = bonds_m[i][1]
    bond_vector.append(a)
    bond_vector.append(b)



def E_stretch_calc(real_bonds_m):
    """ This a function that computes the stretching energy. It returns a list
    of the contribution of each atom and the total energy.

    Inputs:
        real_bonds_m (List[float]): bond length at current structure

    Returns:
        E_stret_lt (List[float]): list of energy contribution for each atom
        E_stret ([float]): stretching contribution of the total energy

    """

    E_stret = 0 ; E_stret_lt = []

    for i in range(len(real_bonds_m)):
        E_stretching_cont = kb_list[i]*(real_bonds_m[i] - eq_bonds[i])**2
        E_stret += E_stretching_cont
        E_stret_lt.append(E_stretching_cont)

    return E_stret_lt, E_stret

E_stret_list, E_stretching = E_stretch_calc(real_bonds)



#···································
#······  ANGLE CALCULATION   ·······
#···································


def angle_def(bond_vectors):
    """ This is a function that takes the bond vector - that is: [1,2,1,3,1,4] - meaning atoms 1 and 2
    are bonded, 1 and 3 are bonded, etc. and creates an angle vector in that equal form.

    Inputs:
        bond_vectors (List[Int]): 1D array with the atom indices involved in the bonds.

    Returns:
        angle_list2 (List[Int]): list with the atom indices involved in the angles.
    """


    angle_list2 = []
    appended_angles = []

    for i in range(len(bond_vectors)):
        element = bond_vectors[i]

        for j in range(len(bond_vectors)):
            if element == bond_vectors[j] and i != j:
                if (j % 2) == 0: #if the number is even, then it is the first atom of the bond
                    other_j = j + 1
                else:
                    other_j = j - 1
                if (i % 2) == 0:
                    other_i = i + 1
                else:
                    other_i = i -1

                possible_angle = sorted([element, bond_vectors[other_i],bond_vectors[other_j]])
                if possible_angle not in appended_angles:
                    appended_angles.append(possible_angle)                    # print(possible_angle)
                    angle_list2.append(element) #central element
                    angle_list2.append(bond_vectors[other_i]) #external 1
                    angle_list2.append(bond_vectors[other_j]) #external 2


    return angle_list2

angle_list = angle_def(bond_vector)

def angle_character():
    """ This function takes the angle_list2 and characterize the angles.
    Return lists are equilibrium angles and force constants

    Inputs:
        all the necessary variables are defined outside of the function

    Returns:
        angle_eq (List[float]): equilibrium angles for each angle (from tiny force field)
        force_cte_a (List[float]): equilibirum forces for each angle (from tiny force field)

    """

    angle_atoms = [] ; h = 0 ; angle_eq = [] ; force_cte_a = []
    for n in range(len(angle_list)):
        a1 = int(angle_list[n])
        angle_atoms.append(atoms[a1-1])

    #characterization of the angle
    while h < len(angle_atoms):
        if angle_atoms[h] and angle_atoms[h+1] and angle_atoms[h+2] == 'C':
            theta_eq = 109.5 * (np.pi/180)
            ka = 60.

        if angle_atoms[h] == 'H' and angle_atoms[h+1] == 'C' and angle_atoms[h+2] == 'C'or angle_atoms[h] == 'C' and angle_atoms[h+1] == 'C' and angle_atoms[h+2] == 'H':
            theta_eq = 109.5 * (np.pi/180)
            ka = 35.

        if angle_atoms[h+1] == 'H' and angle_atoms[h] == 'C' and angle_atoms[h+2] == 'H':
            theta_eq = 109.5 * (np.pi/180)  #important to do the operation in radians
            ka = 35.
        h += 3
        angle_eq.append(theta_eq)
        force_cte_a.append(ka)
    return angle_eq, force_cte_a

theta_0, k_a = angle_character()


def angle_math(coord):
    """This function computes the angles of the molecule. It takes the coordinates and
    returns a list of the R vectors as well as lists of angles and atoms.

    Inputs:
        coord (List[float]): 1D array of the coordinates

    Returns:
        thetas (List[float]) : 1D array of the angles at current structure
        r_ba_lt (List[float]) : 1D array containing the r_BA vectors
        r_bc_lt    (List[float]) : 1D array containing the r_BC vectors

    """

    m = 0 ; thetas = [] ; r_bc_lt = [] ; r_ba_lt = []

    while m < len(angle_list):

        #first we define the atoms for each angle
        atomA = (int(angle_list[m+1])-1)*3

        atomB = (int(angle_list[m])-1)*3 #central atom is stored the first one

        atomC = (int(angle_list[m+2])-1)*3 #multiplied by three so that I extract the coordinates correctly


        #applying the formula
        r_ba = [ float(coord[atomA]) - float(coord[atomB]), float(coord[atomA+1]) - float(coord[atomB+1]), float(coord[atomA+2]) - float(coord[atomB+2]) ]
        r_ba_lt.append(r_ba)
        r_bc = [ float(coord[atomC]) - float(coord[atomB]), float(coord[atomC+1]) - float(coord[atomB+1]), float(coord[atomC+2]) - float(coord[atomB+2]) ]
        r_bc_lt.append(r_bc)

        numerator = np.dot(r_ba,r_bc)
        r_ba_norm = np.sqrt((float(coord[atomA]) - float(coord[atomB]))**2 + (float(coord[atomA+1]) - float(coord[atomB+1]))**2 + (float(coord[atomA+2]) - float(coord[atomB+2]))**2 )
        r_bc_norm = np.sqrt((float(coord[atomB]) - float(coord[atomC]))**2 + ( float(coord[atomB+1]) - float(coord[atomC+1]))**2 + ( float(coord[atomB+2]) - float(coord[atomC+2]) )**2 )

        theta = (np.arccos(numerator/(r_ba_norm * r_bc_norm)))
        thetas.append(theta)

        m += 3
    return thetas, r_ba_lt, r_bc_lt


theta_a, r_ba_list, r_bc_list = angle_math(coord_m)


def E_bending_calc(theta):
    """This function computes the bending energy taking as input the list of angles.

    Inputs:
        theta (List[float]): angles at current structure

    Returns:
        E_bending ([float]): total contribution of the bending energy
        E_bend_list (List[float]): lsit of the bending contributions to the energy of each atom
    """

    E_bending = 0 ; E_bend_list = []

    for i in range(len(theta)):
        E_bending_cont = k_a[i]*(theta[i] - theta_0[i])**2
        E_bending += E_bending_cont
        E_bend_list.append(E_bending_cont)
    return E_bending, E_bend_list


E_bend, E_b_list = E_bending_calc(theta_a)

#··········································
#······          DIHEDRALS          ·······
#··········································


def dihedral_def():
    """This function computes the dihedral matrix in the form: [  [[1,2], [2,6], [1,3]],  [[1,2], [2,5], [1,4]]  ]

    Returns:
        dihedral_list (List[float]): array of the atom indices involved in the dihedrals

    """

    dihedral_list2 = [] ; dihedral_list = []
    dihedral_list3 = []
    for i in range(len(bonds_m)):
        for j in range(len(bonds_m)):
            for k in range(len(bonds_m)):
                for l in range(0,2):
                    for m in range(0,2):
                        for n in range(0,2):
                            if bonds_m[i][l] == bonds_m[j][m] and i != j and bonds_m[i][1-l] == bonds_m[k][n] and i != k and k != j :   #if they share at least one element, then there must be a dihedral
                                #for the dihedral 1-3-2-6, the bonds would be [1,2], [2,6], [1,3]
                                element_sorted = sorted([bonds_m[j],bonds_m[i],bonds_m[k]])
                                if element_sorted not in dihedral_list2:
                                    dihedral_list2.append(element_sorted)
                                    dihedral_list.append([bonds_m[j][1-m],bonds_m[i][l],bonds_m[i][1-l], bonds_m[k][1-n]])

                                    dihedral_list3.append(int(bonds_m[j][1-m])-1)
                                    dihedral_list3.append(int(bonds_m[i][l])-1)
                                    dihedral_list3.append(int(bonds_m[i][1-l])-1)
                                    dihedral_list3.append(int(bonds_m[k][1-n])-1)

    return dihedral_list, dihedral_list3
dihedral_mat, dihedral_v = dihedral_def()


def dihe_energy(coord):
    """This function computes the torsional contribution to the energy
    Inputs: 
        E_dihedral (float): contribution of the torsion to the energy 
        E_dihe_list (List[float]): list of torsional contributions of each atom
        phi_list: (List[float]): list of torsional angles in radians
        vector_dh_lt (list[float]): list of R_ij vectors (R_AB, R_BC, ...)
    """


    E_dihedral = 0; E_dihe_list = [] ; torsion_rab = [] ; torsion_rbc = [] ; torsion_rcd = []
    torsion_rac = [] ; torsion_rbd = [] ; phi_list = [] ; vector_dh_lt = []

    for i in range(len(dihedral_mat)):

        atomA = (int(dihedral_mat[i][0])-1)*3
        atomB = (int(dihedral_mat[i][1]) -1)*3
        atomC = (int(dihedral_mat[i][2]) -1)*3 #B and C are the shared atoms
        atomD = (int(dihedral_mat[i][3])-1)*3

#         applying the formula
        r_ab = [ float(coord[atomB]) - float(coord[atomA]), float(coord[atomB+1]) - float(coord[atomA+1]), float(coord[atomB+2]) - float(coord[atomA+2]) ]
        torsion_rab.append(r_ab)
        r_bc = [ float(coord[atomC]) - float(coord[atomB]), float(coord[atomC+1]) - float(coord[atomB+1]), float(coord[atomC+2]) - float(coord[atomB+2]) ]
        torsion_rbc.append(r_bc)
        r_cd = [ float(coord[atomD]) - float(coord[atomC]), float(coord[atomD+1]) - float(coord[atomC+1]), float(coord[atomD+2]) - float(coord[atomC+2]) ]
        torsion_rcd.append(r_cd)
        r_ac = [ float(coord[atomC]) - float(coord[atomA]), float(coord[atomC+1]) - float(coord[atomA+1]), float(coord[atomC+2]) - float(coord[atomA+2]) ]
        torsion_rac.append(r_ac)
        r_bd = [ float(coord[atomD]) - float(coord[atomB]), float(coord[atomD+1]) - float(coord[atomB+1]), float(coord[atomD+2]) - float(coord[atomB+2]) ]
        torsion_rbd.append(r_bd)
        vector_dh_lt.append([r_ab,r_bc,r_cd,r_ac,r_bd])


        t = np.cross(r_ab, r_bc)
        u = np.cross(r_bc,r_cd)
        v = np.cross(t,u)

        t_norm = linalg.norm(t)
        u_norm = linalg.norm(u)
        r_bc_norm = linalg.norm(r_bc)

        cos = np.dot(t,u)/(t_norm*u_norm)
        sin = np.dot(r_bc,v)/(r_bc_norm*t_norm*u_norm)

        phi = np.arctan2(sin,cos)
        phi_list.append(phi)

        #the energy:
        A_phi = 0.3
        n = 3
        E_dihe = A_phi*(1 + np.cos(n*phi))
        E_dihe_list.append(E_dihe)
        E_dihedral += A_phi*(1 + np.cos(n*phi))
        i += 1
        
    return E_dihedral, E_dihe_list, vector_dh_lt, phi_list



E_tot_dihedral, E_dihe_lt,vector_dh_lt, phi_rad = dihe_energy(coord_m)


n_internals = len(phi_rad) + len(theta_a) + len(real_bonds)
#··········································
#······       Lennard-Jones         ·······
#··········································

def con_mat1():
    """This function creates the connectivity matrix.
    
    Inputs (globally defined):
        angle_list (List[Str]): this function iterates over the angle list to extract connectivity information for each possible pair of atoms
        
    Returns: 
        mat (List[Int]): connectivity matrix
    
    """

    #first we create a null matrix with the correct dimensions
    mat = [[0 for j in range(len(atoms))] for j in range(len(atoms))]

    #then we fill with a "1" if they are bonded (that is if they are next to each other in the bond list)
    #or with a "2" if they are connected through a shared atom (they form a bond)
    
    for i in range(0,len(angle_list),3):
        element1 = int(angle_list[i+1])-1 #bc angle list stars to count at 1, but the matrix index starts at 0.
        element2 = int(angle_list[i])-1
        element3 = int(angle_list[i + 2])-1
        mat[element1][element2] = 1 #they are bonded
        mat[element1][element3] = 2 #they are bonded to the same atom
        mat[element2][element1] = 1
        mat[element2][element3] = 1
        mat[element3][element1] = 2
        mat[element3][element2] = 1
        
    return mat

con_mat = con_mat1()

def pair_list():
    
    """ This function creates a list of all the possible pairs of atoms in the molecule.
    Returns:
        pair_list (List[Str]): list of all the possible pairs of atoms
    """
    
    pair_list = []
    for j in range(len(con_mat)):
        for i in range(len(con_mat)):
            if con_mat[i][j] == 0 and i != j and [j+1,i+1] not in pair_list :
                pair_list.append([i+1,j+1])

    return pair_list


pair_dvw = pair_list()



def vdw_calc(coord):

    """This function computes the VDW energy.
    Inputs:
        coord (List[float]): A 1D vector of the coordinates as (x1,y1,z1,x2,y2,z3,...)
    
    Returns:    
        distance_pair (List[float]): A 1D vector of the length of the vectors between atoms
        E_vdw_lt (List[float]): A 1D vector of the VDW contribution of each atom to the energy
        E_vdw [float]: Total VDW contribution to the energy
        sigma_list (List[float]): list of the sigma constants (from tiny force field)
        epsilon_list (List[float]): list of the epsilon constants (from tiny force field)
    """

    distance_pair = [] ; sigma_list = [] ; epsilon_list = []
    pair = 0
    # print(len(atoms))
    while pair < len(pair_dvw):
        atom1 = int(pair_dvw[pair][0]) #because python starts to count at zero but mol2 file does not
        atom2 = int(pair_dvw[pair][1]) #because each atom has three coordinates
        r_ab = np.sqrt( ((float(coord[(atom1-1)*3])) - float(coord[(atom2-1)*3]))**2 + (float(coord[(atom1-1)*3+1]) - float(coord[(atom2-1)*3+1]))**2 + (float(coord[(atom1-1)*3+2]) - float(coord[(atom2-1)*3+2]))**2 )
        distance_pair.append(r_ab)

        # print(atom1, atoms[atom1-1])
        if atoms[atom1-1] == 'C':
            sigma1 = 1.75
            epsilon1 = 0.07
        if atoms[atom2-1] == 'C':
            sigma2 = 1.75
            epsilon2 = 0.07
        if atoms[atom1-1] == 'H':
            sigma1 = 1.20
            epsilon1 = 0.03
        if atoms[atom2-1] == 'H':
            sigma2 = 1.20
            epsilon2 = 0.03

        sigma_t = 2*np.sqrt(sigma1*sigma2)
        sigma_list.append(sigma_t)
        epsilon_t = np.sqrt(epsilon1*epsilon2)
        epsilon_list.append(epsilon_t)
        pair += 1


    i = 0
    E_vdw = 0 ; E = 0 ; E_vdw_lt = []
    for i in range(len(sigma_list)):
        E = 4*(epsilon_list[i])*((sigma_list[i]/distance_pair[i])**12 - (sigma_list[i]/distance_pair[i])**6)
        E_vdw_lt.append(E)
        E_vdw += 4*(epsilon_list[i])*((sigma_list[i]/distance_pair[i])**12 - (sigma_list[i]/distance_pair[i])**6)

    return distance_pair, E_vdw_lt, E_vdw, sigma_list, epsilon_list

pair_distance, E_vdw_list, E_vdw_tot, sigma_vdw, epsilon_vdw = vdw_calc(coord_m)


#··········································
#······          ENERGIES           ·······
#··········································

def ene_tot(coords_i):
    """This function joins together all the other energy functions

    Inputs:
         coords_i (List[float]): the coordinates of the new structure for which the energy will be computed.

    Returns:
        total_n_energy (float): total energy of the new structure

    """

    #Stretching contribution
    eq_bonds_new, kb_list_new, real_bonds_new, v_rba_new = bond_analysis(bonds_m,coords_i)
    E_stret_list_new, E_stretching_new = E_stretch_calc(real_bonds_new)

    #Bending contribution
    theta_a_new, r_ba_list_new, r_bc_list_new = angle_math(coords_i)
    E_bend_new, E_b_list_new = E_bending_calc(theta_a_new)

    #Dihedral contribution
    E_tot_dihedral_new, E_dihe_lt_new, lt, phi_rad_new = dihe_energy(coords_i)

    #WVD contribution
    pair_distance_new, E_vdw_list_new, E_vdw_tot_new, sigma_vdw_new, epsilon_vdw_new = vdw_calc(coords_i)

    #Total energy
    total_n_energy = E_vdw_tot_new + E_tot_dihedral_new +sum(E_b_list_new) + sum(E_stret_list_new)

    return total_n_energy


def cart_to_internal(coords_i):

    """ This function takes cartesian coordinates and converts them into internal coordinates.

    Inputs:
        coords_i (List[float]): 1D cartesian coordinates of the structure to be converted

    Returns
        internal_coord (List[float]): 1D array with the new internal coordinates

    """

    #stretching coordinates
    eq_bonds_cart, kb_list_cart, real_bonds_cart, v_rba_cart = bond_analysis(bonds_m,coords_i)

    #bending coordinates
    theta_a_cart, r_ba_list_cart, r_bc_list_cart = angle_math(coords_i)

    #Dihedral coordinates
    E_tot_dihedral_cart, E_dihe_lt_cart, lt_cart, phi_rad_cart = dihe_energy(coords_i)

    internal_coord = real_bonds_cart + theta_a_cart + phi_rad_cart

    return internal_coord



#··········································
#······    STRETCHING GRADIENTS     ·······
#··········································

def grad_stret(real_bonds_m, v_rba):
    """This function computes the gradient of the stretching energy

    Inputs:
        real_bonds_m (List[float]): lengths of the bonds at current structure
        v_rba (List[float]): lengths of the R_BA vectors
        
    Returns:
        stret_list (List[float]): list of the energy contributions to the energy of each atom.
    """

    g_list_stret = []

    for i in range(len(real_bonds_m)):
        g_x = - 2 * kb_list[i] * (real_bonds_m[i] - eq_bonds[i]) * \
            (v_rba[i][0] / linalg.norm(v_rba[i]))
        g_y = - 2 * kb_list[i] * (real_bonds_m[i] - eq_bonds[i]) * \
            (v_rba[i][1] / linalg.norm(v_rba[i]))
        g_z = - 2 * kb_list[i] * (real_bonds_m[i] - eq_bonds[i]) * \
            (v_rba[i][2] / linalg.norm(v_rba[i]))
        g_list_stret.append([g_x, g_y, g_z])

        #g is the contribution of each bond to the gradient, but we want the contribution to be by atoms

    at_b = []
    at_a = []
    for j in range(0, len(bond_vector), 2):
        at_b.append(bond_vector[j])
        at_a.append(bond_vector[j+1])

    stret_x = 0
    stret_y = 0
    stret_z = 0
    elemento_list = []
    stret_list = []
    for i in range(len(atoms)):
        el = str(i+1)
        stret_x, stret_y, stret_z = 0, 0, 0  # Reset for each atom

        if el not in elemento_list:
            elemento_list.append(el)

            for j in range(len(at_b)):

                if el == at_b[j]:
                    stret_x += -g_list_stret[j][0]
                    stret_y += -g_list_stret[j][1]
                    stret_z += -g_list_stret[j][2]
            for k in range(len(at_a)):

                if el == at_a[k]:

                    stret_x += g_list_stret[k][0]
                    stret_y += g_list_stret[k][1]
                    stret_z += g_list_stret[k][2]

        if stret_x != 0 or stret_y != 0 or stret_z != 0:
            stret_list.append([float(stret_x), float(stret_y), float(stret_z)])

    return stret_list

stretch_grad_cont = grad_stret(real_bonds, v_rba)

#··········································
#······     BENDING GRADIENTS       ·······
#··········································
def grad_angles(r_ba_list,r_bc_list, theta_a):

    """This function computes the bending gradient of each atom type: atom A (external), atom B (central), atom C.
    It then creates three different lists for each atom type.
    Inputs:
        r_ba_list (List[float]): lengths of the R_BA vectors
        r_bc_list (List[float]): lengths of the R_BC vectors
        theta_a (List[float]): angles at current structure
        
    Returns:
        grad_list_a (List[float]): list of the gradient contributions for atom type A
        grad_list_b (List[float]): list of the gradient contributions for atom type B
        grad_list_c (List[float]): list of the gradient contributions for atom type C
    """

    i = 0
    grad_list_a = []
    grad_a = 0
    grad_b = 0
    grad_c = 0
    grad_list_b = []
    grad_list_c = []
    derivada_list = []

    while i < (len(theta_a)):
        norm_rba = linalg.norm(r_ba_list[i])
        norm_rbc = linalg.norm(r_bc_list[i])
        p = np.cross(r_ba_list[i],r_bc_list[i])


        #for x coordinate of atom 1:
        derivada_a = - np.cross(r_ba_list[i],p) / ((norm_rba**2)*linalg.norm(p))

        derivada_list.append(derivada_a)
        grad_a = 2*k_a[i]*(theta_a[i] - theta_0[i])*(derivada_a)
        grad_list_a.append(grad_a)


        #for x coordinate of atom 2:
        derivada_b = np.cross(r_ba_list[i],p) / ((norm_rba**2)*linalg.norm(p)) + (-np.cross(r_bc_list[i],p) / ((norm_rbc**2)*linalg.norm(p)))
        derivada_list.append(derivada_b)


        grad_b = 2*k_a[i]*(theta_a[i] - theta_0[i])*(derivada_b)
        grad_list_b.append(grad_b)


         #for x coordinate of atom 3:
        derivada_c = (np.cross(r_bc_list[i],p) / ((norm_rbc**2)*linalg.norm(p)))
        derivada_list.append(derivada_c)

        grad_c = 2*k_a[i]*(theta_a[i] - theta_0[i])*(derivada_c)
        grad_list_c.append(grad_c)

        i += 1

    return grad_list_a, grad_list_b, grad_list_c

grad_list_a, grad_list_b, grad_list_c = grad_angles(r_ba_list,r_bc_list, theta_a)


def bending_grad(grad_list_a, grad_list_b, grad_list_c):
    """This function creates a list containing the total contribution of each atom to the bending gradient.
    Inputs:
        grad_list_a (List[float]): list of the gradient contributions for atom type A
        grad_list_b (List[float]): list of the gradient contributions for atom type B
        grad_list_c (List[float]): list of the gradient contributions for atom type C
    
    Returns:
        cont_list (List[float]): 1D array of the gradient contribution for each atom (as the coordinates).
    """

    cont_list = []
    elemento_list = []
    elemento = None
    i = j = k = l = 0
    cont_y = 0
    cont_z = 0

    angle = 0
    at_b = []
    at_c = []
    at_a = []
    while angle < (len(angle_list)):
        at_b.append(angle_list[angle])
        anglea = angle + 1
        at_a.append(angle_list[anglea])
        anglec = angle + 2
        at_c.append(angle_list[anglec])
        angle += 3
    i = 0

    for i in range(len(atoms)):
        elemento = str(i+1)
        cont_x = 0
        cont_y = 0
        cont_z = 0
        if elemento not in elemento_list:
            elemento_list.append(elemento)

            for j in range(len(at_c)):
                if elemento == at_c[j]:
                    cont_x += -grad_list_c[j][0]
                    cont_y += -grad_list_c[j][1]
                    cont_z += -grad_list_c[j][2]

            for k in range(len(at_a)):
                if elemento == at_a[k]:
                    cont_x += -grad_list_a[k][0]
                    cont_y += -grad_list_a[k][1]
                    cont_z += -grad_list_a[k][2]

            for l in range(len(at_b)):
                if elemento == at_b[l]:
                    cont_x += -grad_list_b[l][0]
                    cont_y += -grad_list_b[l][1]
                    cont_z += -grad_list_b[l][2]

        # if cont_x != 0 or cont_y != 0 or cont_z != 0:
            cont_list.append([cont_x, cont_y, cont_z])

    return cont_list

bending_grad_cont = bending_grad(grad_list_a, grad_list_b, grad_list_c)


#··········································
#······    TORSIONAL GRADIENTS      ·······
#··········································

def grad_torsion(vector_dh_lt, phi_rad):
    """This function creates a list containing the total contribution of each atom to the torsional gradient.
    Inputs:
        vector_dh_lt (List[float]): list of R_ij vectors (R_AB, R_BC, ...)
        phi_rad (List[float]): list of angles in radians
        
    Returns:
        grad_list_a, grad_list_b, grad_list_c, grad_list_d (List[float]): contribution of each atom type for a dihedral
        labeled as A-B-C-D.
        
    """
    
    
    i = 0
    grad_list_a = []
    grad_a = 0
    grad_b = 0
    grad_c = 0
    grad_list_b = []
    grad_list_c = []
    grad_list_d = []
    A_phi = 0.3
    n = 3

    while i < (len(phi_rad)):

        norm_rbc = linalg.norm(vector_dh_lt[i][1])

        t = np.cross(vector_dh_lt[i][0],vector_dh_lt[i][1])
        u = np.cross(vector_dh_lt[i][1],vector_dh_lt[i][2])


        #for x coordinate of atom 1:
        derivada_a = np.cross((np.cross(t,vector_dh_lt[i][1]) / ((norm_rbc)*(linalg.norm(t)**2))), vector_dh_lt[i][1] )

        grad_a = -n*A_phi*(np.sin(n*phi_rad[i]))*(derivada_a)
        grad_list_a.append(grad_a)


        #for x coordinate of atom 2:
        frac1 = np.cross(t,vector_dh_lt[i][1])  / ((norm_rbc)*(linalg.norm(t)**2))
        frac2 = np.cross(-u,vector_dh_lt[i][1])  / ((norm_rbc)*(linalg.norm(u)**2))
        derivada_b = np.cross(vector_dh_lt[i][3],frac1) + np.cross(frac2,vector_dh_lt[i][2])

        grad_b = -n*A_phi*(np.sin(n*phi_rad[i]))*(derivada_b)
        grad_list_b.append(grad_b)


         #for x coordinate of atom 3:
        frac1c = np.cross(t,vector_dh_lt[i][1])  / ((norm_rbc)*(linalg.norm(t)**2))
        frac2c = np.cross(-u,vector_dh_lt[i][1])  / ((norm_rbc)*(linalg.norm(u)**2))
        derivada_c = np.cross(frac1c,vector_dh_lt[i][0]) + np.cross(vector_dh_lt[i][4],frac2c)

        grad_c = -n*A_phi*(np.sin(n*phi_rad[i]))*(derivada_c)
        grad_list_c.append(grad_c)


        #for x coordinate of atom 4:
        derivada_d = np.cross((np.cross(-u,vector_dh_lt[i][1]) / ((norm_rbc)*(linalg.norm(u)**2))), vector_dh_lt[i][1] )

        grad_d = -n*A_phi*(np.sin(n*phi_rad[i]))*(derivada_d)
        grad_list_d.append(grad_d)


        i += 1
    return grad_list_a, grad_list_b, grad_list_c, grad_list_d

grad_t_a, grad_t_b, grad_t_c, grad_t_d = grad_torsion(vector_dh_lt, phi_rad)


def torsion_grad(grad_t_a, grad_t_b, grad_t_c, grad_t_d):
    """This function creates a list containing the total contribution of each atom to the torsional gradient.
    Inputs:
        grad_t_a, grad_t_b, grad_t_c, grad_t_d (List[float]): contribution of each atom type for a dihedral
        labeled as A-B-C-D.
    
    """
    
    
    cont_list = []
    elemento_list = []
    elemento = None
    i = j = k = l = 0
    cont_y = 0
    cont_z = 0

    torsion = 0
    at_b1 = []
    at_c1 = []
    at_a1 = []
    at_d1 = []
    while torsion < (len(dihedral_v)):
        at_a1.append(dihedral_v[torsion])
        torsionb = torsion + 1
        at_b1.append(dihedral_v[torsionb])
        torsionc = torsion + 2
        at_c1.append(dihedral_v[torsionc])
        torsiond = torsion + 3
        at_d1.append(dihedral_v[torsiond])
        torsion += 4
    i = 0

    for i in range(len(atoms)):
        elemento = str(i)
        cont_x = 0
        cont_y = 0
        cont_z = 0
        if elemento not in elemento_list:
            elemento_list.append(elemento)

            for j in range(len(at_b1)):

                if int(elemento) == at_b1[j]:
                    cont_x += grad_t_b[j][0]
                    cont_y += grad_t_b[j][1]
                    cont_z += grad_t_b[j][2]

            for k in range(len(at_a1)):
                if int(elemento) == at_a1[k]:
                    cont_x += grad_t_a[k][0]
                    cont_y += grad_t_a[k][1]
                    cont_z += grad_t_a[k][2]

            for l in range(len(at_c1)):
                if int(elemento) == at_c1[l]:
                    cont_x += grad_t_c[l][0]
                    cont_y += grad_t_c[l][1]
                    cont_z += grad_t_c[l][2]

            for m in range(len(at_d1)):
                if int(elemento) == at_d1[m]:
                    cont_x += grad_t_d[m][0]
                    cont_y += grad_t_d[m][1]
                    cont_z += grad_t_d[m][2]

        # if cont_x != 0 or cont_y != 0 or cont_z != 0:
            cont_list.append([cont_x, cont_y, cont_z])

        cont_x = 0
    return cont_list

torsion_grad_list = torsion_grad(grad_t_a, grad_t_b, grad_t_c, grad_t_d )

#··········································
#······        VDW GRADIENTS        ·······
#··········································

def grad_vdw(coord):
    """This function computes a list of gradient contribution of the VDW energy per pair.
    It takes the coordinates as a vector and it returns a list of the effective pairs.
    
    Inputs:
        coord (List[float]): 1D array of the coordinates at current structure.
    
    Returns:
        grad_vdw_list (List[float]): list with the VDW contributions to the gradient per pair.    
    """
    grad_vdw_list = []

    for i in range(len(pair_dvw)):
        at1 = int(pair_dvw[i][0])  - 1
        at2 = int(pair_dvw[i][1]) - 1

        #coordinates
        dist_vector = np.array([coord[at1*3] - coord[at2*3], coord[at1*3 + 1] - coord[at2*3 + 1], coord[at1*3 + 2] - coord[at2*3 + 2] ])
        r = linalg.norm(dist_vector)
        A = 4*epsilon_vdw[i]*(sigma_vdw[i]**12)
        B = 4*epsilon_vdw[i]*(sigma_vdw[i]**6)
        grad_pair = dist_vector*( (-12*A/(r**14)) + (6*B/r**8))
        grad_vdw_list.append(grad_pair)

    return grad_vdw_list

grad_pair_list = grad_vdw(coord_m)


def vdw_lt(grad_pair_list):
    """This function computes a list of gradient contribution of the VDW energy per atom.
    It takes the coordinates as a vector and it returns a list of the effective pairs.
    
    Inputs:
        grad_pair_list (List[float]): list with the VDW contributions to the gradient per pair.
    
    Returns:
        pair_at_list (List[float]): list with the VDW contributions to the gradient per atom.    
    """
    
    at1_l = []
    at2_l = []
    pair_at_list = []

    for p in range(0, len(pair_dvw)):
        at1_l.append(pair_dvw[p][0])
        at2_l.append(pair_dvw[p][1])

    el_list = []
    for i in range(len(atoms)):
        el = str(i+1)
        pair_x, pair_y, pair_z = 0, 0, 0  # Reset for each atom

        for j in range(len(at1_l)):

            if el == str(at1_l[j]):
                el_list.append(el)
                pair_x += grad_pair_list[j][0]
                pair_y += grad_pair_list[j][1]
                pair_z += grad_pair_list[j][2]

        for k in range(len(at2_l)):

            if el == str(at2_l[k]):
                el_list.append(el)
                pair_x += -grad_pair_list[k][0]
                pair_y += -grad_pair_list[k][1]
                pair_z += -grad_pair_list[k][2]

        if pair_x != 0 or pair_y != 0 or pair_z != 0:
            pair_at_list.append(
                [int(el), float(pair_x), float(pair_y), float(pair_z)])

    for i in range(1, len(pair_at_list), 1):
        if str(i) not in el_list:
            pair_at_list.append(
                [int(i), float(0.000), float(0.000), float(0.000)])
        pair_at_list = sorted(pair_at_list)

    return pair_at_list

pair_vdw_grad = vdw_lt(grad_pair_list)




def grad_n(coords_i):
    """This is a function that joints all the necessary functions for computing gradients.
    It takes the coordinates (coords_i) in a vector form.
    It returns a vector-like list of the gradients for each atom. """

    #We compute the dihedrals
    E_tot_dihedraln, E_dihe_ltn, lt, phi_rad = dihe_energy(coords_i)

    #Computation of the bonds
    eq_bonds,kb_list, real_bondsn, v_rba = bond_analysis(bonds_m,coords_i)
    stretch_grad_cont = grad_stret(real_bondsn, v_rba)

    #Computation of the angles
    theta_an, r_ba_list, r_bc_list = angle_math(coords_i)

    grad_list_a, grad_list_b, grad_list_c = grad_angles(r_ba_list,r_bc_list,theta_an)
    bending_grad_cont = bending_grad(grad_list_a, grad_list_b, grad_list_c)

    #Computation of the torsional gradients
    grad_t_a, grad_t_b, grad_t_c, grad_t_d = grad_torsion(lt, phi_rad)
    torsion_grad_list = torsion_grad(grad_t_a, grad_t_b, grad_t_c, grad_t_d )

    #Computation of the WDV gradients
    grad_pair_list = grad_vdw(coords_i)
    pair_vdw_grad = vdw_lt(grad_pair_list)
    tot_gr_list = []

    if not torsion_grad_list:  # if torsion_grad_list is empty
        torsion_grad_list = [[0.0, 0.0, 0.0] for i in range(len(atoms))]
    if not pair_vdw_grad:
        pair_vdw_grad = [[0.0, 0.0, 0.0, 0.0] for i in range(len(atoms))]

    for i in range(len(atoms)):

        x_val_gr = bending_grad_cont[i][0] + stretch_grad_cont[i][0] + \
            torsion_grad_list[i][0] + pair_vdw_grad[i][1]

        y_val_gr = bending_grad_cont[i][1] + stretch_grad_cont[i][1] + \
            torsion_grad_list[i][1] + pair_vdw_grad[i][2]

        z_val_gr = bending_grad_cont[i][2] + stretch_grad_cont[i][2] + \
            torsion_grad_list[i][2] + pair_vdw_grad[i][3]


        tot_gr_list.append(x_val_gr)
        tot_gr_list.append(y_val_gr)
        tot_gr_list.append(z_val_gr)
    return tot_gr_list



#··········································
#······        TOTAL GRADIENT       ·······
#··········································
tot_gr = []
tot_gr_list = []

if not torsion_grad_list:  # if torsion_grad_list is empty
    torsion_grad_list = [[0.0, 0.0, 0.0] for i in range(len(atoms))]
if not pair_vdw_grad:
    pair_vdw_grad = [[0.0,0.0,0.0,0.0] for i in range(len(atoms))]

for i in range(len(atoms)):


    x_gr = bending_grad_cont[i][0] + stretch_grad_cont[i][0] + torsion_grad_list[i][0] + pair_vdw_grad[i][1]
    y_gr = bending_grad_cont[i][1] + stretch_grad_cont[i][1] + torsion_grad_list[i][1] + pair_vdw_grad[i][2]
    z_gr = bending_grad_cont[i][2] + stretch_grad_cont[i][2] + torsion_grad_list[i][2] + pair_vdw_grad[i][3]

    tot_gr.append([x_gr, y_gr, z_gr])
    tot_gr_list.append(x_gr)
    tot_gr_list.append(y_gr)
    tot_gr_list.append(z_gr)


tot_gr_list = grad_n(coord_m)

#··········································
#······        BFGS ALGORITHM       ·······
#··········································


a = time.time()

output_file2 = name.replace(".mol2", "") + "_CSF_out2.out"

with open(output_file2, 'w', encoding='utf-8') as file, open('opt.molden', 'w', encoding='utf-8') as file2:

    print("="*50, file=file)
    print("      OPTIMIZATION IN CARTESIAN COORDINATES        ", file=file)
    print("      Program by Cristina Sobrino Fernández        ", file=file)
    print("="*50, file=file)
    print(file=file)

    #Initial guess for the hessian matrix (diagonal elements equal to 1/300)
    M = np.zeros((len(coord_m),len(coord_m))) #square matrix of 3Nx3N
    M[:len(coord_m), :len(coord_m)] = np.eye(len(coord_m)) * (1/300)

    #initialization of the variables needed for the algorithm
    grms = 1
    coord_n0 = coord_m #initial coordinates
    coord_n1 = np.zeros(len(coord_m)) #new coordinates
    j = 0
    coord_list = []
    grms_list = []
    energy_list = []
    grad_n0 = grad_n(coord_m)

    print("Initial geometry (Å)", file=file)
    for i in range(0,len(coord_n1),3):
        print(f"{atoms[int(i/3)]} {coord_n0[i]:15.6f} {coord_n0[i+1]:15.6f} {coord_n0[i+2]:15.6f}", file=file)

    print(file=file)
    print("Initial gradient (Kcal/mol/Å)", file=file)

    for i in range(0,len(grad_n0),3):
        print(f"{atoms[int(i/3)]} {grad_n0[i]:15.6f} {grad_n0[i+1]:15.6f} {grad_n0[i+2]:15.6f}", file=file)
    print(file=file)
    print("Potential energy at input structure", f"{ene_tot(coord_n0):15.6f}", "Kcal/mol", file=file)
    print(file=file)

    while grms > 0.001 and j < 500:  #max 200 optimization cycles

        print("·"*30, file=file)
        print("······· Cycle number", j+1,"·······", file=file)
        print("·"*30, file=file)
        print("", file=file)
        
        #Update the gradient
        tot_gr_list = grad_n(coord_n0)
        
        #Update the line search
        pk = -np.matmul(M,tot_gr_list)

        print("Predicted structure change",file=file)
        print(file=file)
        for i in range(0,len(grad_n0),3):
            print(f"{atoms[int(i/3)]} {pk[i]:15.6f} {pk[i+1]:15.6f} {pk[i+2]:15.6f}", file=file)


        alpha = 0.8
        contador = 1
        cond1 = 1
        cond2 = 0

        #line search algorithm


        print(file=file)
        print("Line search:", file=file)
        print(len("Line search:")*"-", file=file)


        while cond1 >= cond2: #first wolfe rule
            alpha = (0.8)**(contador) 

            # Convert to NumPy array to sum the vectors
            coord_n1 = np.array(coord_n1)  
            coord_n0 = np.array(coord_n0)
            pk = np.array(pk)
            
            #update the coordinates
            coord_n1 = coord_n0 + alpha * pk
            
            #compute the two sides of the equation of the wolfe rule
            cond1 = ene_tot(coord_n1)
            cond2 = ene_tot(coord_n0) + 0.1*alpha*np.dot(pk,tot_gr_list)
            contador += 1

            print(f"{'alpha'} {alpha:15.4f} {'Energy'} {cond1:15.8f}" ,file=file)


        energy_list.append(cond1)
        coord_list.append(coord_n1.copy())
        print(file=file)
        print("New structure (Å)", file=file)

        for i in range(0,len(coord_n1),3):
            print(f"{atoms[int(i/3)]} {coord_n1[i]:15.6f} {coord_n1[i+1]:15.6f} {coord_n1[i+2]:15.6f}", file=file)



        #compute the gradient at the new structure
        gr = grad_n(coord_n1)
        print(file=file)
        print("New gradient (Kcal/mol/Å)", file=file)

        for i in range(0,len(gr),3):
            print(f"{atoms[int(i/3)]} {gr[i]:15.6f} {gr[i+1]:15.6f} {gr[i+2]:15.6f}", file=file)


        #compute grms and check if optimization converged
        grms = np.sqrt((1/(3*len(atoms)))*(np.dot(gr,gr)))
        grms_list.append(grms)
        print(file=file)
        print("Energy:", f"{cond1:15.8f}", file=file)
        print("                 Value    Threshold    Converged?", file=file)
        if grms > 0.001:
            print(f"{'GRMS:  '} {grms:15.4f} {'   '} {0.001} {'     No    '} ", file=file)
        else:
            print(f" {'GRMS  '} {grms:20.4f} {'   '} {0.001} {'   YES   '}" , file=file)
            print("·····································", file=file)
            print("····   OPTIMIZATION CONVERGED    ····", file=file)
            print("·····································", file=file)
        print(file=file)

        sk = alpha * pk
        wk = np.matmul(M,sk)

        yk = np.zeros(len(gr))


        for i in range(len(gr)):
            yk[i] = gr[i] - grad_n0[i]

        vk = np.matmul(M,yk)

        sk_yk = np.dot(sk, yk)
        yk_vk = np.dot(yk, vk)

        #compute the hessian
        M = M + ((sk_yk + yk_vk) * np.outer(sk, sk) / (sk_yk**2)) - ((np.outer(vk, sk) + np.outer(sk, vk)) / sk_yk)

        coord_n0 = coord_n1.copy()
        grad_n0 = gr.copy()

        j += 1

    # print("Optimization not converged", file=file)
    print(file=file)
    print(file=file)
    print(file=file)
    print("&& FINAL ENERGY", f"{ene_tot(coord_n1):.8f}", file=file)
    print(file=file)
    print("                         FINAL STRUCTURE:", file=file)
    print(f"{'Number':^6} {'Element':^10} {'X':^15} {'Y':^15} {'Z':^15}", file=file)
    print("·" * 68, file=file)
    for i in range(0,len(coord_n1),3):
        print(f"{int(i/3)+1:<6} {atoms[int(i/3)]:^10} {coord_n1[i]:15.6f} {coord_n1[i+1]:15.6f} {coord_n1[i+2]:15.6f}", file=file)


    print("[Molden Format]", file=file2)
    print("[GEOCONV]", file=file2)
    print("energy", file=file2)
    for i in range(len(energy_list)):
        print(energy_list[i], file=file2)
    print("rms-force", file=file2)
    for i in range(len(grms_list)):
        print(grms_list[i], file=file2)
    print("[GEOMETRIES] (XYZ)", file=file2)

    for k in range(len(coord_list)):
        print(len(atoms), file=file2)
        print(cond1, file=file2)
        for i in range(len(atoms)):
            print(f"{atoms[int(i)]} {coord_list[k][i*3]:15.6f} {coord_list[k][i*3+1]:15.6f} {coord_list[k][i*3+2]:15.6f}", file=file2)


b = time.time()
et= b-a
print("Optimization process in cartesian coordinates took", et)




print("The optimization process converged normally, the output is ready.")


#··········································
#······         WILSON ELEMENTS     ·······
#··········································

def wilson_bending(r_ba_list,r_bc_list, theta_a):
    """ This function computes the bending contribution to the Wilson matrix
    Inputs: 
        r_ba_list,r_bc_list (List[float]): R_BA and R_BC vector lengths
        theta_a (List[float]): angles at current structure
    
    Returns:
        derivada_list (List[float]): derivative of the bending elements
    
    """

    i = 0

    derivada_list = []

    while i < (len(theta_a)):
        norm_rba = linalg.norm(r_ba_list[i])
        norm_rbc = linalg.norm(r_bc_list[i])
        p = np.cross(r_ba_list[i],r_bc_list[i])

        #for x coordinate of atom 1:
        derivada_a = np.cross(r_ba_list[i],p) / ((norm_rba**2)*linalg.norm(p))
        derivada_list.append(derivada_a)

        #for x coordinate of atom 2:
        derivada_b = -np.cross(r_ba_list[i],p) / ((norm_rba**2)*linalg.norm(p)) + (np.cross(r_bc_list[i],p) / ((norm_rbc**2)*linalg.norm(p)))
        derivada_list.append(derivada_b)

         #for x coordinate of atom 3:
        derivada_c = -(np.cross(r_bc_list[i],p) / ((norm_rbc**2)*linalg.norm(p)))
        derivada_list.append(derivada_c)

        i += 1


    return derivada_list


bend_dern = wilson_bending(r_ba_list, r_bc_list, theta_a)


def wilson_torsion(vector_dh_lt, phi_rad):
    """ This function computes the torsional contribution to the Wilson matrix
    Inputs: 
        vector_dh_lt (List[float]): R_BA, R_BC, ... vector lengths
        phi_rad (List[float]): dihedral angles at current structure
    
    Returns:
        derivada_torsion (List[float]): derivative of the torsional elements
    """
    
    i = 0
    derivada_torsion = []

    while i < (len(phi_rad)):

        norm_rbc = linalg.norm(vector_dh_lt[i][1])
        t = np.cross(vector_dh_lt[i][0], vector_dh_lt[i][1])
        u = np.cross(vector_dh_lt[i][1], vector_dh_lt[i][2])

        #for x coordinate of atom 1:
        derivada_a = np.cross((np.cross(
        t, vector_dh_lt[i][1]) / ((norm_rbc)*(linalg.norm(t)**2))), vector_dh_lt[i][1])
        derivada_torsion.append(derivada_a)

        #for x coordinate of atom 2:
        frac1 = np.cross(t, vector_dh_lt[i][1]) / \
                             ((norm_rbc)*(linalg.norm(t)**2))
        frac2 = np.cross(-u, vector_dh_lt[i][1]) / \
                             ((norm_rbc)*(linalg.norm(u)**2))
        derivada_b = np.cross(
                vector_dh_lt[i][3], frac1) + np.cross(frac2, vector_dh_lt[i][2])
        derivada_torsion.append(derivada_b)

        #for x coordinate of atom 3:
        frac1c = np.cross(
                t, vector_dh_lt[i][1]) / ((norm_rbc)*(linalg.norm(t)**2))
        frac2c = np.cross(-u, vector_dh_lt[i][1]) / \
                              ((norm_rbc)*(linalg.norm(u)**2))
        derivada_c = np.cross(
                frac1c, vector_dh_lt[i][0]) + np.cross(vector_dh_lt[i][4], frac2c)
        derivada_torsion.append(derivada_c)

        #for x coordinate of atom 4:
        derivada_d = np.cross(
                (np.cross(-u, vector_dh_lt[i][1]) / ((norm_rbc)*(linalg.norm(u)**2))), vector_dh_lt[i][1])
        derivada_torsion.append(derivada_d)

        i += 1
    return derivada_torsion

deriv_torn = wilson_torsion(vector_dh_lt, phi_rad)

internal_coord = int(len(bonds_m)+len(angle_list)/3+len(dihedral_mat))




def wilson_mat(real_bonds_m, v_rba, bend_der, deriv_tor):
    """ This function computes the Wilson B matrix
    Inputs: 
        v_rba (List[float]): R_BA vector lengths
        bend_der, deriv_tor (List[float]): torsional and bending contributions 
        real_bonds_m (List[float]): bond lengths at current structure
    
    Returns:
        b_s (List[List[float]]): Wilson B matrix
    """
    

    internal_coord = int(len(bonds_m)+len(angle_list)/3+len(dihedral_mat))

    b_s = np.zeros((internal_coord,len(coord_m)))

    for j in range(0,len(bond_vector),2):
        #stretching elements

        j_2 = int(j/2)
        element1 = int(bond_vector[j]) - 1
        element2 = int(bond_vector[j+1]) -1
        norm_v_rba = linalg.norm(v_rba[j_2])  
        b_s[j_2][element1 * 3] = v_rba[j_2][0] / norm_v_rba
        b_s[j_2][element1 * 3 + 1] = v_rba[j_2][1] / norm_v_rba
        b_s[j_2][element1 * 3 + 2] = v_rba[j_2][2] / norm_v_rba

        b_s[j_2][element2 * 3] = v_rba[j_2][0] / norm_v_rba*(-1)
        b_s[j_2][element2 * 3 + 1] = v_rba[j_2][1] / norm_v_rba*(-1)
        b_s[j_2][element2 * 3 + 2] = v_rba[j_2][2] / norm_v_rba*(-1)
    

    stret_elements=len(real_bonds)

    #bending elements
    for k in range(0,int(len(angle_list)/3)):

        k_3 = int(k)+stret_elements


        element_a = int(angle_list[k*3+1]) - 1
        element_b = int(angle_list[k*3]) - 1
        element_c = int(angle_list[k*3+2]) - 1



        b_s[k_3][element_a*3]   = bend_der[(k)*3][0]
        b_s[k_3][element_a*3+1] = bend_der[(k)*3][1]
        b_s[k_3][element_a*3+2] = bend_der[(k)*3][2]

        b_s[k_3][element_b*3]   =  bend_der[(k)*3+1][0]
        b_s[k_3][element_b*3+1] =  bend_der[(k)*3+1][1]
        b_s[k_3][element_b*3+2] =  bend_der[(k)*3+1][2]

        b_s[k_3][element_c*3]   =  bend_der[(k)*3+2][0]
        b_s[k_3][element_c*3+1] =  bend_der[(k)*3+2][1]
        b_s[k_3][element_c*3+2] =  bend_der[(k)*3+2][2]


    for l in range(0,int(len(dihedral_v)/4)):

        l_4 = int(l)+stret_elements + int(len(angle_list)/3)


        element_a = int(dihedral_v[l*4])
        element_b = (dihedral_v[l*4+1])
        element_c = (dihedral_v[l*4+2])
        element_d = (dihedral_v[l*4+3])


        b_s[l_4][element_a*3]   = deriv_tor[(l)*4][0]
        b_s[l_4][element_a*3+1] = deriv_tor[(l)*4][1]
        b_s[l_4][element_a*3+2] = deriv_tor[(l)*4][2]

        b_s[l_4][element_b*3]   =  deriv_tor[(l)*4+1][0]
        b_s[l_4][element_b*3+1] =  deriv_tor[(l)*4+1][1]
        b_s[l_4][element_b*3+2] =  deriv_tor[(l)*4+1][2]

        b_s[l_4][element_c*3]   =  deriv_tor[(l)*4+2][0]
        b_s[l_4][element_c*3+1] =  deriv_tor[(l)*4+2][1]
        b_s[l_4][element_c*3+2] =  deriv_tor[(l)*4+2][2]

        b_s[l_4][element_d*3]   =  deriv_tor[(l)*4+3][0]
        b_s[l_4][element_d*3+1] =  deriv_tor[(l)*4+3][1]
        b_s[l_4][element_d*3+2] =  deriv_tor[(l)*4+3][2]


    return b_s

B = wilson_mat(real_bonds, v_rba, bend_dern, deriv_torn)
Bt = np.transpose(B)



def bgt(Bw):
    """ This function computes the inverse of the G and B matrices
    Inputs:
        Bw (List[List[float]]): Wilson B matrix
    Returns:
        B_inverse (List[List[float]]): Inverse of the Wilson B matrix
        G_inverse (List[List[float]]): Inverse of the G matrix
    """

    Bt = np.transpose(Bw)

    G = np.matmul(Bw, Bt)


    threshold = 1e-12

    G[np.abs(G) < threshold] = 0


    eigenvaluesG, eigenvectorsG = np.linalg.eigh(G)

    eigenvaluesG[np.abs(eigenvaluesG) < threshold] = 0

    Lambda_inv_values = []

    for e in eigenvaluesG:
        if e > threshold:
            Lambda_inv_values.append(1/e)
        else:
            Lambda_inv_values.append(0)

    Lambda_inv = np.diag(Lambda_inv_values)

    G_inverse = eigenvectorsG @ Lambda_inv @ eigenvectorsG.T


    B_inverse = G_inverse @ Bw
    return B_inverse, G_inverse



#··························································
#······        INTERNAL COORDINATES ALGORITHM       ·······
#··························································


#new wilson b matrix
def wilson_b(coords_i):
    """This function joins all the Wilson-related functions so it only depends on the coordinates.
    Inputs:
        coords_i (List[float]): 1D array of the coordinates
    
    Returns:
        Bn (List[List[float]]): Wilson B matrix at current structure
    
    """    

    eq_bonds,kb_list, real_bondsn, v_rba = bond_analysis(bonds_m,coords_i)

    theta_an, r_ba_list, r_bc_list = angle_math(coords_i)

    E_tot_dihedraln, E_dihe_ltn, lt, phi_rad = dihe_energy(coords_i)

    bend_der1 = wilson_bending(r_ba_list, r_bc_list, theta_an)

    deriv_tor1 = wilson_torsion(lt, phi_rad)

    Bn = wilson_mat(real_bondsn, v_rba, bend_der1, deriv_tor1)

    return Bn


output_file3 = name.replace(".mol2", "") + "_CSF_out3.out"

with open(output_file3, 'w', encoding='utf-8') as file, open('opt-internal.molden', 'w', encoding='utf-8') as file2:

    def cartesian_search(initial_coordinates, qk, sk_int, Bt, G_inverse):
        """ This function finds the optimal cartesian coordinates from a set of internal coordinates.
        
        Inputs:
            initial_coordinates (List[float]): 1D array of the coordinates
            qk (List[float]): set of internal coordinates
            sk_int (List[float]): desired change to be carried out on the internal coordinates
            Bt (List[List[float]]): Inverse of the Wilson B matrix
            G_inverse (List[List[float]]): Inverse of the G matrix
        
        Returns:
            x_new (List[float]): updated set of optimal cartesian coordinates
        
        
        """
        

        print("  ", file=file)
        print("·."*16, file=file)
        print("·.·.·. Cartesian fitting  ·.·.·.",file=file)
        print("·."*16,file=file)
        print("  ", file=file)


        max_x = 1 ; it = 1
        sqk = sk_int.copy()
        xk = initial_coordinates.copy()
        q_new = qk + sk_int

        for i in range(len(real_bonds)+len(theta_a)):
            sqk[i] = q_new[i] - qk[i]

        for i in range(len(real_bonds)+len(theta_a), len(sqk)):
            sqk[i] = q_new[i] - qk[i]
            if sqk[i] > np.pi:
                sqk[i] = sqk[i] - 2*np.pi
            if sqk[i] < -np.pi:
                sqk[i] = sqk[i] + 2*np.pi

        x_new = xk + Bt @ G_inverse @ sqk
        q_current = cart_to_internal(x_new)

        qk = q_current.copy()

        xk = x_new.copy()

        while max_x > 0.00001:

            print("Iteration number", it, file=file)
            print("----------------", file=file)

            for i in range(len(real_bonds)+len(theta_a)):
                sqk[i] = q_new[i] - qk[i]

            for i in range(len(real_bonds)+len(theta_a), len(sqk)):
                sqk[i] = q_new[i] - qk[i]
                if sqk[i] > np.pi:
                    sqk[i] = sqk[i] - 2*np.pi
                if sqk[i] < -np.pi:
                    sqk[i] = sqk[i] + 2*np.pi

            x_new = xk + Bt @ G_inverse @ sqk
            q_current = cart_to_internal(x_new)
            print("Current set of internals coordinates", file=file)
            for i in q_current:
                print(f"{i:.4f}", end="   ", file=file)

            print("  ", file=file)

            qk = q_current.copy()

            max_x = max(abs(x_new-xk))
            print(file=file)
            print("                             Value    Threshold    Converged?", file=file)
            if max_x > 0.00001:
                print(f"{'Max change in x:  '} {max_x:15.5f} {'   '} {0.00001} {'     No    '} ", file=file)
            else:
                print(f" {'Max change in x:  '} {max_x:20.5f} {'   '} {0.00001} {'   YES   '}" , file=file)
            print(file=file)
            print(file=file)
            xk = x_new.copy()
            it += 1

            # to avoid infinite loop
            if it > 10:
                print("Maximum iterations reached!")
                break


        return x_new

    a = time.time()

    print("="*45, file=file)
    print("   OPTIMIZATION IN INTERNAL COORDINATES     ", file=file)
    print("   Program by Cristina Sobrino Fernández    ", file=file)
    print("="*45, file=file)
    print(file=file)

    print("Input structure (Å):", file=file)
    print("", file=file)
    print(f"{'Number':^6} {'Element':^10} {'X':^15} {'Y':^15} {'Z':^15}", file=file)
    print("·" * 68, file=file)

    for i in range(len(atoms)):
        elemento = atoms[i]
        x_val_int = coord_m[i*3]
        y_val_int = coord_m[i*3+1]
        z_val_int = coord_m[i*3+2]

        print(f"{i+1:<6}  {elemento:^10} {x_val_int:>15.4f} {y_val_int:>15.4f} {z_val_int:>15.4f}", file=file)
    print(file=file)

    #Initial guess for the hessian
    M = np.zeros((int(n_internals),int(n_internals))) #square matrix of 3Nx3N
    #np.eye creates diagonal elements
    M[:len(real_bonds), :len(real_bonds)] = np.eye(len(real_bonds)) * (1/600)
    M[len(real_bonds):len(real_bonds)+len(theta_a), len(real_bonds):len(real_bonds)+len(theta_a)] = np.eye(len(theta_a)) * (1/150)
    M[len(real_bonds)+len(theta_a):len(real_bonds)+len(theta_a)+len(phi_rad), len(real_bonds)+len(theta_a):len(real_bonds)+len(theta_a)+len(phi_rad)] = np.eye(len(phi_rad)) * (1/80)

    #Initial internal gradient
    B_inverse, G_inverse = bgt(B)
    cartesian_gradient = grad_n(coord_m)
    internal_gradient = G_inverse @ B @ cartesian_gradient

    pk_int = -np.matmul(M,internal_gradient)

    sk_int = pk_int

    #initialization of variables
    xk0 = coord_m.copy()
    qk = real_bonds + theta_a + phi_rad
    sqk = sk_int.copy()
    xk = coord_m.copy()
    ite = 1
    grms = 1

    while grms > 0.001 and ite < 100:

        #STEP ONE
        print("                      ", file=file)
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::", file=file)
        print(":::           Optimization cycle number", ite,"       :::",file=file)
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::", file=file)
        print("                 ", file=file)

        print("Potential energy", f"{ene_tot(xk0):15.8f} Kcal/mol", file=file)
        print(file=file)
        B1 = wilson_b(xk0)
        Bt = np.transpose(B1)

        B_inverse, G_inver = bgt(B1)
        cartesian_gradient = grad_n(xk0)
        internal_gradient1 = G_inver @ B1 @ cartesian_gradient
        print("Gradient in terms of the internal coordinates:", file=file)
        for i in internal_gradient1:
            print(f"{i:.6f}", end="   ", file=file)
        print(file=file)


        #STEP 2
        pk_i = - M @ internal_gradient1

        #STEP 3
        sk_int = pk_i
        print(file=file)
        print("Predicted update step s_k in internal coordinates:", file=file)
        for i in sk_int:
            print(f"{i:.4f}", end="   ", file=file)


        rms = np.sqrt((sk_int@sk_int)/n_internals)


        l_qk = np.sqrt(np.sum(sk_int**2) / n_internals)
        if l_qk > 0.02:
            print(file=file)
            print("Predicted step is too long", file=file)
            sk_int = sk_int*(0.02/l_qk)

            print(file=file)
            print("Scaled update step in internal coordinates:", file=file)
            for i in sk_int:
                print(f"{i:.4f}", end="   ", file=file)

        print("Maximum step is:", f"{rms:10.6f}", file=file)
        print(file=file)
        #STEP 4 is performed in cartesian_search function


        #STEP 5
        xn = cartesian_search(xk0, qk, sk_int, Bt, G_inver)
        print("Cartesian fitting is converged. The new set of cartesian coordinates is:", file=file)
        print(f"{'Number':^6} {'Element':^10} {'X':^15} {'Y':^15} {'Z':^15}", file=file)
        print("·" * 68, file=file)

        for i in range(len(atoms)):
            elemento = atoms[i]
            x_val = xn[i*3]
            y_val = xn[i*3+1]
            z_val = xn[i*3+2]

            print(f"{i+1:<6}  {elemento:^10} {x_val:>15.4f} {y_val:>15.4f} {z_val:>15.4f}", file=file)

        print(file=file)
        qc = cart_to_internal(xn)
        print("New set of internals", file=file)
        for i in qc:
            print(f"{i:.6f}", end="   ", file=file)
        print(file=file)
        print("Predicted max step is", f"{rms:15.4f}", file=file)
        print(file=file)

        #STEP 6
        B1 = wilson_b(xn)
        B_inverse, G_inver = bgt(B1)

        cartesian_gradient2 = grad_n(xn)
        internal_gradient2 = G_inver @ B1 @ cartesian_gradient2

        print("Gradient in terms of the internal coordinates:", file=file)
        for i in internal_gradient2:
            print(f"{i:.6f}", end="   ", file=file)
        print(file=file)


        y_qk = internal_gradient2 - internal_gradient1
        v_qk = M @ y_qk

        q1 = cart_to_internal(xn)


        for i in range(len(real_bonds)+len(theta_a)):
            sqk[i] = q1[i] - qk[i]

        for i in range(len(real_bonds)+len(theta_a), len(sqk)):
            sqk[i] = q1[i] - qk[i]
            if sqk[i] > np.pi:
                sqk[i] = sqk[i] - 2*np.pi
            if sqk[i] < -np.pi:
                sqk[i] = sqk[i] + 2*np.pi


        #STEP 7
        M_n = M + ((np.dot(sqk, y_qk) + np.dot(y_qk, v_qk)) * np.outer(sqk, sqk)) / (np.dot(sqk, y_qk)**2) - (np.outer(v_qk, sqk) + np.outer(sqk, v_qk)) / np.dot(sqk, y_qk)
        M = M_n.copy()


        grms = np.sqrt((1/(3*len(atoms)))*(np.dot(cartesian_gradient2,cartesian_gradient2)))

        print(file=file)
        print("&& Energy after cycle", ite,":", f"{ene_tot(xn):10.8f} Kcal/mol", file=file)
        print(file=file)
        print("                 Value    Threshold    Converged?", file=file)
        if grms > 0.001:
            print(f"{'GRMS:  '} {grms:15.4f} {'   '} {0.001} {'     No    '} ", file=file)
        else:
            print(f" {'GRMS  '} {grms:20.4f} {'   '} {0.001} {'   YES   '}" , file=file)
            print(file=file)
            print("·····································", file=file)
            print("····   OPTIMIZATION CONVERGED    ····", file=file)
            print("·····································", file=file)
            print(file=file)
        print(file=file)



        ite += 1


        #We redefine variables
        xk0 = xn.copy()
        qk = q1.copy()
        internal_gradient = internal_gradient2.copy()


    print("                         FINAL STRUCTURE:", file=file)
    print(f"{'Number':^6} {'Element':^10} {'X':^15} {'Y':^15} {'Z':^15}", file=file)
    print("·" * 68, file=file)

    for i in range(len(atoms)):
        elemento = atoms[i]
        x_val_final = xn[i*3]
        y_val_final = xn[i*3+1]
        z_val_final = xn[i*3+2]

        print(f"{i+1:<6}  {elemento:^10} {x_val_final:>15.6f} {y_val_final:>15.6f} {z_val_final:>15.6f}", file=file)
    print("&& FINAL ENERGY: ", f"{ene_tot(xn):10.8f} Kcal/mol", file=file)

    b = time.time()
    eto = b - a
    print("Optimization in internal coordinates took", eto, "seconds")



#··········································
#······  FORMATTING OF THE OUTPUT   ·······
#··········································


with open('output.txt', 'w', encoding='utf-8') as file:

    print("Input name: ", name, file=file)
    print("The molecule has ", len(atoms), "atoms", file=file)

    def cartesian_format(atoms_inp, coord_inp):
        print(f"{'·································'.center(60)}", file=file)
        print(f"{'···  Input orientation in Å  ···'.center(60)}", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Number':^6} {'Element':^10} {'X':^15} {'Y':^15} {'Z':^15}", file=file)
        print("·" * 68, file=file)

        for i in range(len(atoms_inp)):
            elemento = atoms_inp[i]
            x_val_cart = coord_inp[i*3]
            y_val_cart = coord_inp[i*3+1]
            z_val_cart = coord_inp[i*3+2]

            print(f"{i+1:<6}  {elemento:^10} {x_val_cart:>15.4f} {y_val_cart:>15.4f} {z_val_cart:>15.4f}", file=file)

    cartesian_format(atoms, coord_m)

    print("Number of coordinates:", file=file)
    print(f"{'Stretching:      '} {len(bonds_m)} {'  Bending:     '} {len(angle_list)/3 :.0f} {' Torsion:     '} {len(dihedral_mat)}", file=file)
    print(f"{'Internal:     '} {len(bonds_m)+len(angle_list)/3+len(dihedral_mat):.0f} {' Cartesian:     '} {len(atoms)*3}" , file=file)
    print(" ", file=file)
    print("  ", file=file)
    print(f"{'·································'.center(60)}", file=file)
    print(f"{'···        INPUT ENERGIES     ···'.center(60)}", file=file)
    print(f"{'·································'.center(60)}", file=file)
    print(" ", file=file)
    print("  ", file=file)
    print("&& Total stretching energy is", f"{sum(E_stret_list):.6f}", file=file)
    print("&& Total bending energy is", f"{sum(E_b_list):.6f}", file=file)
    print("&& Total dihedral energy is", f"{E_tot_dihedral:.6f}", file=file)
    print("&& Total VDW energy is", f"{E_vdw_tot:.6f}", file=file)
    print("&& Total potential energy is", f"{(E_vdw_tot+E_tot_dihedral+sum(E_b_list)+sum(E_stret_list)):.6f}", file=file)



    def bonds_format(bonds_lt, atoms_lt, real_bond, E_stret_lt):
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(f"{'···        Bond matrix        ···'.center(60)}", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Bonded centers':^20} {'Distance (Å)':^10} {'Energy contribution (kcal/mol)':^10}", file=file)
        print("·" * 60, file=file)

        for i in range(len(bonds_lt)):
            e1 = int(bonds_lt[i][0])
            e2 = int(bonds_lt[i][1])
            e1_s = atoms_lt[e1-1]
            e2_s = atoms_lt[e2-1]
            dist = real_bond[i]
            cont = E_stret_lt[i]

            print(f"{e1:^2} {e1_s:^2} {'......'} {e2:^2} {e2_s:^2} {dist:^15.4f} {cont:^15.4f}", file=file)
        print("&& Total stretching energy is", sum(E_stret_lt), file=file)

    bonds_format(bonds_m, atoms, real_bonds, E_stret_list)

    def angle_format(theta_a_p, angle_list_p, atoms_p):
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(f"{'···        Angle matrix       ···'.center(60)}", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Bonded centers':^20} {'Angle (radians)':^10} {'Angle (degrees)':^10} {'Energy contribution (kcal/mol)':^10}", file=file)
        print("·" * 70, file=file)

        for i in range(len(theta_a_p)):
            e1 = int(angle_list_p[i*3])
            e2 = int(angle_list_p[i*3+1])
            e3 = int(angle_list_p[i*3+2])
            e1_s = atoms_p[e1-1]
            e2_s = atoms_p[e2-1]
            e3_s = atoms_p[e3-1]
            ang = theta_a_p[i]
            cont = E_b_list[i]

            print(f"{e2:^2} {e2_s:^2} {'  ---  '} {e1:^2} {e1_s:^2} {'  ---  '} {e3:^2} {e3_s:^2} {ang:^15.4f} {cont:^15.4f}", file=file)
        print("&& Total bending energy is", sum(E_b_list), file=file)

    angle_format(theta_a, angle_list, atoms)

    def dihe_format():
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(f"{'···       Dihedral matrix     ···'.center(60)}", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Bonded centers':^20} {'Angle (degrees)':^10} {'Angle (radians)':^10} {'Energy contribution (kcal/mol)':^10}", file=file)
        print("·" * 70, file=file)

        for dihedral in range(len(dihedral_mat)):
            # angle_degrees = phi_lt[dihedral]
            angle_radians = phi_rad[dihedral]
            energy_contribution = E_dihe_lt[dihedral]

            print(f"{dihedral_mat[dihedral][0]:^2} -- {dihedral_mat[dihedral][1]:^2} -- {dihedral_mat[dihedral][2]:^2} -- {dihedral_mat[dihedral][3]:^2} {angle_radians:^10.4f} {energy_contribution:^15.4f}", file=file)
        print("There are", len(dihedral_mat), "dihedral angles", file=file)
        print("&& Total dihedral energy is", E_tot_dihedral, file=file)

    dihe_format()

    def vdw_format():
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(f"{'···         VDW matrix       ···'.center(60)}", file=file)
        print(f"{'·································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Not bonded centers':^20} {'Distance (A)':^10} {'Energy contribution (kcal/mol)':^10}", file=file)
        print("·" * 70, file=file)

        for pair in range(len(pair_dvw)):
            distance = f"{pair_distance[pair]:.4f}".center(15)
            energy_cont = f"{E_vdw_list[pair]:.4f}".center(20)

            print(f"{pair_dvw[pair][0]:^4} {atoms[int(pair_dvw[pair][0])-1]:^4} -- {pair_dvw[pair][1]:^4} {atoms[int(pair_dvw[pair][1])-1]:^4}  {distance:^15} {energy_cont:^15}", file=file)
        print("&& Total VDW energy is", E_vdw_tot, file=file)

    vdw_format()



    print(" ", file=file)
    print("  ", file=file)
    print(f"{'·················································'.center(60)}", file=file)
    print(f"{'···       Analytical gradient of energy      ···'.center(60)}", file=file)
    print(f"{'·················································'.center(60)}", file=file)
    print(" ", file=file)
    print("  ", file=file)
    print(f"{'Elemento':<10} {'X':>15} {'Y':>15} {'Z':>15}", file=file)
    print("-" * 60, file=file)
    for i in range(len(atoms)):
        
        elemento = str(i+1)
        x_val = bending_grad_cont[i][0] + stretch_grad_cont[i][0] + torsion_grad_list[i][0] + pair_vdw_grad[i][1]
        y_val = bending_grad_cont[i][1] + stretch_grad_cont[i][1] + torsion_grad_list[i][1] + pair_vdw_grad[i][2]
        z_val = bending_grad_cont[i][2] + stretch_grad_cont[i][2] + torsion_grad_list[i][2] + pair_vdw_grad[i][3]

        print(f"{elemento:<2} {atoms[i]:<5} {x_val:>15.6f} {y_val:>15.6f} {z_val:>15.6f}", file=file)




    def stret_grad_format():
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'···················································'.center(60)}", file=file)
        print(f"{'···  Analytical gradient of stretching energy   ···'.center(60)}", file=file)
        print(f"{'···················································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Elemento':<10} {'X':>15} {'Y':>15} {'Z':>15}", file=file)
        print("-" * 60, file=file)
        for i in range(len(atoms)):
            elemento = str(i+1)
            x_val_stret = stretch_grad_cont[i][0]
            y_val_stret = stretch_grad_cont[i][1]
            z_val_stret = stretch_grad_cont[i][2]

            print(f"{elemento:<2} {atoms[i]:<5} {x_val_stret:>15.6f} {y_val_stret:>15.6f} {z_val_stret:>15.6f}", file=file)

    stret_grad_format()





    def bending_grad_format():
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·················································'.center(60)}", file=file)
        print(f"{'···  Analytical gradient of bending energy    ···'.center(60)}", file=file)
        print(f"{'·················································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Elemento':<10} {'X':>15} {'Y':>15} {'Z':>15}", file=file)
        print("-" * 60, file=file)
        for i in range(len(atoms)):
            elemento = str(i+1)
            x_val_bend = bending_grad_cont[i][0]
            y_val_bend = bending_grad_cont[i][1]
            z_val_bend = bending_grad_cont[i][2]

            print(f"{elemento:<2} {atoms[i]:<5} {x_val_bend:>15.6f} {y_val_bend:>15.6f} {z_val_bend:>15.6f}", file=file)

    bending_grad_format()



    def torsion_grad_format():
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·················································'.center(60)}", file=file)
        print(f"{'···  Analytical gradient of torsion energy    ···'.center(60)}", file=file)
        print(f"{'·················································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Elemento':<10} {'X':>15} {'Y':>15} {'Z':>15}", file=file)
        print("-" * 60, file=file)
        try:
            for i in range(len(atoms)):
                elemento = str(i+1)
                x_val_tor = torsion_grad_list[i][0]
                y_val_tor = torsion_grad_list[i][1]
                z_val_tor = torsion_grad_list[i][2]

                print(f"{elemento:<2} {atoms[i]:<5} {x_val_tor:>15.6f} {y_val_tor:>15.6f} {z_val_tor:>15.6f}", file=file)
        except:
            print("There are no dihedral angles in this molecule", file=file)
    torsion_grad_format()



    def vdw_grad_format():
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'·················································'.center(60)}", file=file)
        print(f"{'···    Analytical gradient of VDW energy      ···'.center(60)}", file=file)
        print(f"{'·················································'.center(60)}", file=file)
        print(" ", file=file)
        print("  ", file=file)
        print(f"{'Elemento':<10} {'X':>15} {'Y':>15} {'Z':>15}", file=file)
        print("-" * 60, file=file)
        try:
            for i in range(len(pair_vdw_grad)):
                print(f"{pair_vdw_grad[i][0]:<2} {atoms[i]:<5} {pair_vdw_grad[i][1]:>15.6f} {pair_vdw_grad[i][2]:>15.6f} {pair_vdw_grad[i][3]:>15.6f}", file=file)
        except:
            print("There was an error during the VDW calculation", file=file)
    vdw_grad_format()


    print(" ", file=file)
    print("  ", file=file)
    print(f"{'·································'.center(60)}", file=file)
    print(f"{'···     NORMAL TERMINATION    ···'.center(60)}", file=file)
    print(f"{'·································'.center(60)}", file=file)
