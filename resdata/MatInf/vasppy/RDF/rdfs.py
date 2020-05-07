from pymatgen import Structure, Lattice

a = 5.6402
lattice = Lattice.from_parameters(a, a, a, 90.0, 90.0, 90.0)
print(lattice)
structure = Structure.from_spacegroup(sg='Fm-3m', lattice=lattice,
                                      species=['Na', 'Cl'],
                                      coords=[[0, 0, 0], [0.5, 0, 0]])
print(structure)

from vasppy.rdf import RadialDistributionFunction

indices_Na = [i for i, site in enumerate(structure) if site.species_string is 'Na']
indices_Cl = [i for i, site in enumerate(structure) if site.species_string is 'Cl']
print(indices_Na)
print(indices_Cl)

rdf_nana = RadialDistributionFunction(structures=[structure],
                                      indices_i=indices_Na)
rdf_clcl = RadialDistributionFunction(structures=[structure],
                                      indices_i=indices_Cl)
rdf_nacl = RadialDistributionFunction(structures=[structure],
                                      indices_i=indices_Na,
                                      indices_j=indices_Cl)

import matplotlib.pyplot as plt

plt.plot(rdf_nana.r, rdf_nana.rdf, 'k', label='Na-Na')
plt.plot(rdf_clcl.r, rdf_clcl.rdf, 'b:', label='Cl-Cl')
plt.plot(rdf_nacl.r, rdf_nacl.rdf, 'g--', label='Na-Cl')
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

plt.plot(rdf_nana.r, rdf_nana.smeared_rdf(), 'k', label='Na-Na')
plt.plot(rdf_clcl.r, rdf_clcl.smeared_rdf(sigma=0.05), 'b:', label='Cl-Cl')
plt.plot(rdf_nacl.r, rdf_nacl.smeared_rdf(sigma=0.05), 'g--', label='Na-Cl')
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

rdf_nana = RadialDistributionFunction.from_species_strings(structures=[structure],
                                                           species_i='Na')
rdf_clcl = RadialDistributionFunction.from_species_strings(structures=[structure],
                                                           species_i='Cl')
rdf_nacl = RadialDistributionFunction.from_species_strings(structures=[structure],
                                                           species_i='Na',
                                                           species_j='Cl')
plt.plot(rdf_nana.r, rdf_nana.smeared_rdf(), 'k', label='Na-Na')
plt.plot(rdf_clcl.r, rdf_clcl.smeared_rdf(sigma=0.07), 'b:', label='Cl-Cl')
plt.plot(rdf_nacl.r, rdf_nacl.smeared_rdf(sigma=0.07), 'g--', label='Na-Cl')
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

from pymatgen.io.vasp import Xdatcar

xd = Xdatcar('NaCl_800K_MD_XDATCAR')

rdf_nana_800K = RadialDistributionFunction.from_species_strings(
    structures=xd.structures,
    species_i='Na'
)
rdf_clcl_800K = RadialDistributionFunction.from_species_strings(
    structures=xd.structures,
    species_i='Cl'
)
rdf_nacl_800K = RadialDistributionFunction.from_species_strings(
    structures=xd.structures,
    species_i='Na',
    species_j='Cl'
)

plt.plot(rdf_nana_800K.r, rdf_nana_800K.rdf, 'k', label='Na-Na')
plt.plot(rdf_clcl_800K.r, rdf_clcl_800K.rdf, 'b:', label='Cl-Cl')
plt.plot(rdf_nacl_800K.r, rdf_nacl_800K.rdf, 'g--', label='Na-Cl')
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

struct_1 = struct_2 = struct_3 = structure
rdf_nacl_mc = RadialDistributionFunction(structures=[struct_1, struct_2, struct_3],
                                         indices_i=indices_Na, indices_j=indices_Cl,
                                         weights=[34, 27, 146])
