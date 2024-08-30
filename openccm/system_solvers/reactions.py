########################################################################################################################
# Copyright 2024 the authors (see AUTHORS file for full list).                                                         #
#                                                                                                                      #
#                                                                                                                      #
# This file is part of OpenCCM.                                                                                        #
#                                                                                                                      #
#                                                                                                                      #
# OpenCCM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public  #
# License as published by the Free Software Foundation,either version 2.1 of the License, or (at your option)          #
# any later version.                                                                                                   #
#                                                                                                                      #
# OpenCCM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied        #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                                     #
# See the GNU Lesser General Public License for more details.                                                          #
#                                                                                                                      #
# You should have received a copy of the GNU Lesser General Public License along with OpenCCM. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

r"""
Functions related to parsing user-provided reaction mechanisms and create a numba compiled function which is then used
for the simulations.
"""

import inspect
from pyparsing import Combine, Group, Literal, ParseResults, Suppress, Word, ZeroOrMore, nums
from typing import List, Dict, Optional, Tuple

from ..config_functions import ConfigParser

import numpy as np
import sympy as sp


def generate_reaction_system(config_parser: ConfigParser, _ddt_reshape_shape: Optional[Tuple[int, int, int]]) -> None:
    """
    Main function that handles support for systems of chemical reactions. In order, this function:
    1. Performs an initial parsing of the reactions configuration file based on the two main headers, [REACTIONS] and [RATES].
    2. Parses reactions (and associated rates) in the same species order as they appear specified in the main configuration file.
    3. Creates a runtime-generated function containing the differential reaction system of equations for each specie which contributes to overall mass balance.

    Parameters
    ----------
    * config_parser:      OpenCCM ConfigParser which contains configuration file information and location (i.e. relative path).
    * _ddt_reshape_shape: Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it. Used by `create_reaction_code`
    """
    input_file    = config_parser.get_item(['SIMULATION', 'reactions_file_path'], str)
    rxn_species   = config_parser.get_list(['SIMULATION', 'specie_names'],        str)
    rxn_file_path = config_parser.get_item(['SETUP', 'working_directory'],        str) + '/reaction_code_gen.py'

    # Create dummy file if no reactions specified
    if input_file == 'None':
        with open(rxn_file_path, 'w') as file:
            file.write("from numba import njit\n\n\n")
            file.write("@njit\n")
            file.write("def reactions(C, _ddt):\n")
            file.write("    return\n\n")
        return

    # If reactions are specified, read each line in reactions configuration file
    with open(input_file, 'r') as file:
        all_lines = file.readlines()
    
    # Get [REACTIONS] and [RATES] header locations (since this can be unordered)
    reactions_header = all_lines.index('[REACTIONS]\n')
    rates_header     = all_lines.index('[RATES]\n')

    # Do not assume that reactions come before rates (as provided in the examples)
    if reactions_header < rates_header:
        all_reactions = all_lines[reactions_header+1:rates_header]
    else: 
        all_reactions = all_lines[reactions_header+1:]
    # Strip any preceeding, middle, and trailing whitespaces
    all_reactions = [rxn.strip() for rxn in all_reactions if rxn.strip()]
        
    # Generate reaction system if applicable (i.e. if any reactions are given)
    if all_reactions:
        # Again handle ordering of reactions config file headers
        if reactions_header < rates_header:
            all_rates = all_lines[rates_header+1:]
        else: 
            all_rates = all_lines[rates_header+1:reactions_header]
        # Strip any preceeding, middle, and trailing whitespaces
        all_rates = [rate.strip() for rate in all_rates if rate.strip()]

        # Get a dictionary with all reaction information first 
        rxn_book, dummy_map = organize_reactions_input(rxn_species, all_reactions, all_rates)

        # Send all reaction and rate information into parser to create system of equations
        reaction_eqns = parse_reactions(dummy_map, rxn_book)

    # Otherwise, run a tracer experiment (species present but no chemical reactions occur)
    else:
        reaction_eqns = []

    # Generate runtime function containing differential reaction system of equations for mass balance.
    create_reaction_code(rxn_species, reaction_eqns, rxn_file_path, _ddt_reshape_shape)


def organize_reactions_input(specie_order: List[str], all_reactions: List[str], all_rates: List[str]) \
        -> Tuple[
            Dict[str, List[str]],
            Dict[str, str]
           ]:
    """
    This function first performs several checks to ensure that the reactions config file was handled correctly by the user. 
    If successful, it breaks down the provided reactions and rates into an organized dictionary where each key is a reaction ID and the values are a length 2 list, where list is [reaction, rate].
    It also transforms the given chemical species into dummy specie names, which is much easier for the parser in parse_reactions() to handle. 

    Parameters
    ----------
    * specie_order:     The ordered species as written in the main configuration file.
                        Must match the same types of species present in reactions config file.
    * all_reactions:    The partially-parsed reactions as extracted from reactions configuration file.
    * all_rates:        The partially-parsed reaction rates as extracted from reactions configuration file.

    Returns
    -------
    * rxn_book:     A dictionary that contains each reaction and their chemical species / rate constants.
    * dummy_map:    Mapping between original specie name and internal dummy name
    """ 

    # Parse the reaction ids (i.e. R1, R2 etc) from the actual reactions and associated rates. These are needed for proper input checking.
    rate_ids = [rate.split(':')[0].strip() for rate in all_rates]
    rxn_ids = [rxn.split(':')[0].strip() for rxn in all_reactions]
    rxn_rxns = [rxn.split(':')[1].replace(" ", "") for rxn in all_reactions]

    # This section enforces proper input of reactions configuration file.
    # 1. Duplicate reaction or rate labels are present (i.e., 2 or more definitions for a reaction or rate expression). These definitions must be unique
    if (len(rxn_ids) != np.unique(rxn_ids).size) or (len(rate_ids) != np.unique(rate_ids).size):
        raise RuntimeError("Duplicate reaction and/or rate labels detected. Check reactions configuration file.")
    # 2. If the number of unique reaction IDs does not equal rate IDs, then there are missing reactions or rates
    elif np.unique(rxn_ids).size != np.unique(rate_ids).size:
        raise RuntimeError("Number of reactions does not match number of rates (or their reaction IDs are different). Check reactions configuration file.")
    # 3. Catches duplicate reactions listed in the reactions config file
    elif len(rxn_rxns) != np.unique(rxn_rxns).size:
        raise RuntimeError("Duplicate reactions detected. Check reactions configuration file.")
    # 4. Each reaction does not have a related rate
    elif sorted(rxn_ids) != sorted(rate_ids):
        raise RuntimeError("Cannot find rate for one or more reactions. Check that reaction and rate IDs align.")
    # If no issues, delete the reaction ids and temp reactions as they are no longer needed
    else:
        del rxn_ids, rxn_rxns

    # Prepare rate constants by separating from reaction IDs (i.e. 'R1: 5' --> '5')
    rate_vals = [rate.split(':')[1].replace(' ', '') for rate in all_rates]

    # Create rate dictionary with appropriate reaction IDs
    rate_book = dict(zip((rate_ids), (rate_vals)))

    # Get dictionary for reactions with rates which will complete the parsing of reactions config file
    rxn_book = {} 
    for _, rxn in enumerate(all_reactions):
        curr_key = rxn.split(':')[0]
        rxn_book.setdefault(curr_key, [])
        
        rxn_book[curr_key].append(rxn.split(':')[1].strip())
        rxn_book[curr_key].append(rate_book.get(curr_key))

    # Create mapping of species to dummy specie names which is much easier for parsing process
    dummy_map = {specie_order[i]: f"#{i}#" for i in range(len(specie_order))}
    rxn_book_keys = list(rxn_book.keys())
    for key in rxn_book_keys:
        # get current reaction
        rxn = rxn_book[key][0]
        for spec in specie_order:
            # replace species with dummy specie names
            rxn = rxn.replace(spec, dummy_map[spec])
        
        # assign transformed reaction back to master reaction book
        rxn_book[key][0] = rxn

    return rxn_book, dummy_map


def parse_reactions(dummy_map: Dict[str, str], rxn_book: Dict[str, List[str]]) -> List[str]:
    """
    Parses all reactions and rates and creates the differential reaction system of equations.
    Also enforces specific ordering of system of equations based on species in specie_order (i.e. from main config file).
    See the Notes section below for further description of expectations and limitations of the reactions configuration parser.

    Notes
    -----
    -   The reaction parser only supports forward (written) reactions (i.e. use of ->).
        Reversible reactions must be written as two independent reactions with their associated rate constants.
    -   The reactions parser can parse the general reaction: aA1 + bB2 + [...] -> gG7 + hH8 + [...].
    -   All chemical species can be alphanumeric with underscores (if desired),
        but cannot contain +/- symbols (i.e. for ions) in the specie name.
        -   We recommend using "_m" or "_p" for + and - ions, respectively.
            I.e.: Na+ should be written as Na_p in the reactions and main configuration file.

    An appropriate reactions configuration file for the reversible reaction N2O4 <-> 2NO2 with k_f = 1e-2 and k_r = 3e-4 (these are hypothetical rates) would be:
        [REACTIONS]
        R1: N2O4 -> 2NO2
        R2: 2NO2 -> N2O4

        [RATES]
        R1: 1e-2
        R2: 3e-4

    Parameters
    ----------
    * dummy_map: Mapping between real specie name and dummy name used for parsing.
                    Order of keys must match that of species present in reactions config file.
    * rxn_book:     A dictionary that contains each reaction and their chemical species / rate constants.

    Returns
    -------
    * reaction_eqns:    List containing the system of differential equations in the same order of species
                        as they appear in specie_order. I.e. if specie_order = [a, b] --> reaction_eqns = [da/dt, db/dt].
    """

    def to_dict(rxn_parsed: ParseResults) -> Dict[str, str]:
        """
        Convert the pyparsing result into a dictionary.
        Supports one of two formats for each item in rxn_parsed:
        1. Explicit coefficient: ['a', 'A']
        2. Implicit coefficient: ['A']

        where:
        - A: A chemical specie in the reaction
        - a: coefficient for specie A

        When converting to the dictionary version all implicit coefficients will be made explicit using a value of '1'.

        Parameters
        ----------
        rxn_parsed: Pyparsing result with potential implicit coefficients

        Returns
        -------
        * rxn_dict: Mapping between specie and its coefficient.
        """

        rxn_dict = {}
        for term in rxn_parsed:
            if len(term) == 1:  # Implicit coefficient
                rxn_dict[term[0]] = '1'
            else:  # Explicit coefficient
                rxn_dict[term[1]] = term[0]

        return rxn_dict

    def update_lhs_rhs(side_dict: Dict[str, str], reactants_dict: Dict[str, str], rate_const: str, de_eqn: Dict[str, str], negative_sign: bool) -> None:
        """
        Convert the species, coefficients, and reaction rate

        Parameters
        ----------
        side_dict:      The mapping between species and their stoichiometric coefficients for this half of the reaction.
        reactant_dict:  The mapping between species and their stoichiometric coefficients for the reactants of this reaction.
        rate_const:     The rate constant for this reaction.
        lhs:            The left hand side of the differential equation system. d_specie / dt.
        rhs:            The contribution to the right hand side of the differential equation system built from.
        negative_sign:  Whether a netgative sign needs to be added to the rate, such as for products.

        Returns
        -------
        * Nothing. `lhs` and `rhs` are updated in-place.
        """
        sign = '-' if negative_sign else '+'

        for specie, coeff in side_dict.items():
            if specie not in de_eqn.keys():
                de_eqn[specie] = ''

            if rate_const[:2] == 'z=':  # 0th order reaction
                de_eqn[specie] += sign + coeff + '*' + rate_const[2:]
            else:
                rxn_rate = '*'.join([_s + '**' + _c for _s, _c in reactants_dict.items()])
                de_eqn[specie] += sign + coeff + '*' + rate_const + '*' + rxn_rate

    # The following pyparsing structure outlines the expected reaction structure from reactions config file.
    rxn_arrow               = Literal('->')
    coefficient             = ZeroOrMore(Word(nums, '.' + nums))  # Match integers & floats
    molecule                = Combine(Literal('#') + Word(nums) + Literal('#'))
    molecule_w_coefficient  = Group(coefficient + molecule)
    all_molecules           = ZeroOrMore(molecule_w_coefficient + Suppress('+') | molecule_w_coefficient)
    reactants               = all_molecules + Suppress(rxn_arrow)
    products                = Suppress(reactants) + all_molecules

    # The different differential equation systems
    # Key are the species and values are their net reaction rate
    de_eqns: Dict[str, str] = {}

    # Iterate through rxn_book and get system of equations
    for _, (rxn_id, rxn_info) in enumerate(rxn_book.items()):
        rate_constant = rxn_info[1]

        # Parse and convert to dict the current reaction split by '->', i.e., separate reactants and products.
        # Convert to dictionary and make explicit any implicit coefficients
        reactants_dict  = to_dict(reactants.parseString(rxn_info[0]))
        products_dict   = to_dict(products.parseString(rxn_info[0]))

        # Update the lhs and rhs of differential equation reaction system for the reactants:
        update_lhs_rhs(reactants_dict,  reactants_dict, rate_constant, de_eqns, negative_sign=True)
        update_lhs_rhs(products_dict,   reactants_dict, rate_constant, de_eqns, negative_sign=False)

    # Return to original specie names, now that reaction parsing is complete
    for name_o, name_d in dummy_map.items():
        # Replace in keys
        de_eqns[name_o] = de_eqns.pop(name_d)
        # Replace within reaction rates
        for _name, eqn in de_eqns.items():
            de_eqns[_name] = eqn.replace(name_d, name_o)

    # Enforce order of differential equations in same species appearance as given in main configuration file
    # This is done implicitly since python maintains dictionary keys in insertion order, and dummy_map's keys are in the correct order
    reaction_eqns = []
    for _s, _rxn in de_eqns.items():
        reaction_eqns.append(_rxn)
  
    return reaction_eqns


def create_reaction_code(rxn_species:           List[str],
                         reaction_eqns:         List[str],
                         rxn_file_path:         str,
                         _ddt_reshape_shape:    Optional[Tuple[int, int, int]] = None
                         ) -> None:
    """
    Symbolically creates and simplifies the rhs of the system of differential equations for reaction environment using sympy.
    Then, this rhs is translated into source code and written into a separate .py file at runtime.

    Parameters
    ----------
    * rxn_species:          Name of species in the order that they appear in the main config file.
    * reaction_eqns:        Output of `parse_reactions` which gives system of differential reaction equations appearing in the same specie order as rxn_species.
    * rxn_file_path:        The path to the file in which to save the generated reactions.
    * _ddt_reshape_shape:   Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it.
    """
    # If an empty list was passed for reaction_eqns, it means no reactions are involved in simulation (i.e. tracer experiment). 
    # Therefore take species given and add dummy *0 so that the reaction terms are not contributing to mass balance as desired.
    if not reaction_eqns:
        reaction_eqns = [specie + '*0' for specie in rxn_species]

    # Convert string ODE system to sympy expression type ODE system
    rhs_symb = [sp.parse_expr(eqn) for eqn in reaction_eqns]

    # Convert into function and then to string (i.e. source code)
    the_source = inspect.getsource(sp.lambdify([rxn_species], rhs_symb, 'numpy'))

    # Change function name to reactions
    the_source = the_source.replace("_lambdifygenerated","reactions")

    # Change Dummy variable name
    dummy_name = the_source[the_source.find('(')+1:the_source.find(')')]
    the_source = the_source.replace(dummy_name, 'specie_concentrations')

    # Add _ddt parameter
    close_bracket = the_source.find(')')
    source_left, source_right = the_source[:close_bracket], the_source[close_bracket:]
    if _ddt_reshape_shape:
        source_right = source_right.replace('specie_concentrations', f'specie_concentrations.reshape({_ddt_reshape_shape})')
    the_source = source_left + ', _ddt' + source_right

    # Remove the unneeded return statement
    the_source = the_source[0:the_source.find('return')].strip() + '\n\n'

    # Create a view into _ddt that we can easily work with
    # (n_species, n_pfr * points_per_pfr) -> (n_species, n_pfr, points_per_pfr)
    if _ddt_reshape_shape:  # Only for PFRs
        the_source +=  '    # Reshape concentrations and _ddt to apply reaction only to non-inlet nodes\n'
        the_source += f'    _ddt = _ddt.reshape({_ddt_reshape_shape})\n'
        the_source += '    \n'

    # Unroll _ddt update for each specie
    rhs_strs = [str(eqn) for eqn in rhs_symb]
    if _ddt_reshape_shape:
        for specie in rxn_species:
            for i, rhs in enumerate(rhs_strs):
                rhs_strs[i] = rhs.replace(specie, specie + '[..., 1:]')

    for id_specie, rhs in enumerate(rhs_strs):
        the_source += f'    # Update for specie {rxn_species[id_specie]}\n'
        if _ddt_reshape_shape:
            the_source += f'    _ddt[{id_specie}, :, 1:] += {rhs} \n\n'  # RXN does not apply to inlet node.
        else:
            the_source += f'    _ddt[{id_specie}, :] += {rhs} \n\n'

    # Write system of differential reaction equations to runtime-generated file
    with open(rxn_file_path, 'w') as file:
        file.write("from numba import njit\n")
        file.write("import numpy as np\n")
        file.write("\n")
        file.write("\n")
        file.write('@njit\n')
        file.write(the_source)
