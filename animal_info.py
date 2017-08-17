############################################################################
# animal_info.py
#
# This is the set of defaults for connecting to the animal files.
#
# Created by Badr F. Albanna 4/20/14
# Last edited by Badr F. Albanna 01/12/15
# Copyright 2014 Badr F. Albanna
############################################################################

import numpy

###########
# Animals #
###########

# This will change to variable sets, first item are the variables he second
# are contingencies used to condition the final average.
# [({keys : names  }, {keys : names }),
#       ({keys : names  }, {keys : names }), ...]
VAR_AC_08282013 = {'F' : 'Foil', 'T' : 'Target', 'T+': 'correct_target', 'T-' : "incorrect_target", 'NPT' : 'Hits'}
VAR_AC_10232013 = {'F' : 'foil', 'T' : 'target', 'T+': 'correct_target', 'T-' : "incorrect_target", 'NPT' : 'hits'}
VAR_AC_10292013_1 = {'F' : 'foil', 'T' : 'target', 'T+': 'correct_target', 'T-' : "incorrect_target", .5 : 'DIO_65439', 1 : 'DIO_63423', 2 : 'DIO_61375', 4 : 'target', 8 : 'DIO_57279', 16 : 'DIO_49087', 32 : 'DIO_32703'}
VAR_AC_10292013_2 = {'F' : 'foil', 'T' : 'target', 'F+' : 'correct_reject', 'F-' : 'false_alarms', 'T+' : 'correct_target', 'T-' : 'incorrect_target', 'A' : 'action', 'NA' : 'no_action', 'C' : 'correct', 'I' : 'incorrect', 'NPT' : 'Hits', 'NPF' : 'npf', .5 : 'DIO_65439', 1 : 'DIO_63423', 2 : 'DIO_61375', 4 : 'target', 8 : 'DIO_57279', 16 : 'DIO_49087', 32 : 'DIO_32703' }
VAR_AC_11262013 = {'F' : 'foil', 'T' : 'target', 'F+' : 'correct_reject', 'F-' : 'false_alarm', 'T+' : 'correct_target', 'T-' : 'incorrect_target', 'A' : 'action', 'NA' : 'no_action', 'NPT' : 'Hits', 'NPF' : 'false_alarm'}
VAR_PFC = {'F' : 'Foil', 'T' : 'Target', 'T+': 'correct_target', 'T-' : "incorrect_target"}
VAR_DEFAULT = {'F' : 'foil', 'T' : 'target', 'F+' : 'correct_reject', 'F-' : 'false_alarm', 'T+' : 'correct_target', 'T-' : 'incorrect_target', 'A' : 'action', 'NA' : 'no_action', 'C' : 'correct', 'I' : 'incorrect', 'NPT' : 'hits', 'NPF' : 'NPF', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'target', 16 : 'DIO_16384', 32 : 'DIO_32768'}
VAR_DEFAULT_wo_FA =  {'F' : 'foil', 'T' : 'target', 'T+' : 'correct_target', 'T-' : 'incorrect_target', 'NPT' : 'hits'}
VAR_PFC_12032013 = {'T' : 'Target', 'T+' : 'correct_target', 'NPT' : 'Hits', 'F' : 'Foil', 'F+' : 'correct_reject', 'NPF' : 'NPF' }
VAR_DEFAULT_NEW = {'T': 'target', 'F' : 'foils', 'T+' : 'correct_target', 'F+': 'correct_rejects', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'target', 8 : 'DIO_00001', 16 : 'DIO_16384', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_P5 = {'T': 'target', 'F' : 'foils', 'T+' : 'correct_target', 'F+': 'correct_rejects', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'DIO_16384', 8 : 'DIO_00001', 16 : 'target', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_2 = {'T': 'target', 'F' : 'foil', 'T+' : 'correct_target', 'F+': 'correct_rejects', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'target', 8 : 'DIO_00001', 16 : 'DIO_16384', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_2_P5 = {'T': 'target', 'F' : 'foil', 'T+' : 'correct_target', 'F+': 'correct_rejects', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'DIO_16384', 8 : 'DIO_00001', 16 : 'target', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_3 = {'T': 'target', 'F' : 'foils', 'T+' : 'correct_target', 'F+': 'correct_reject', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'target', 8 : 'DIO_00001', 16 : 'DIO_16384', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_3_P5 = {'T': 'target', 'F' : 'foils', 'T+' : 'correct_target', 'F+': 'correct_reject', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'DIO_16384', 8 : 'DIO_00001', 16 : 'target', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_4 = {'T': 'target', 'F' : 'foils', 'T+' : 'correct_target', 'F+': 'correct_reject', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'target', 8 : 'DIO_00001', 16 : 'DIO_16384', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_4_P5 = {'T': 'target', 'F' : 'foils', 'T+' : 'correct_target', 'F+': 'correct_reject', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'DIO_16384', 8 : 'DIO_00001', 16 : 'target', 32 : 'DIO_32768'}
VAR_DEFAULT_NEW_5 = {'T': 'target', 'F' : 'foil', 'T+' : 'correct_target', 'F+': 'correct_reject', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_65439', 1 : 'DIO_63423', 2 : 'DIO_61375', 4 : 'target', 8 : 'DIO_57279', 16 : 'DIO_49087', 32 : 'DIO_32703'}
VAR_DEFAULT_NEW_6 = {'T': 'target', 'F' : 'foil', 'T+' : 'correct_target', 'F+': 'correct_reject', 'NPT': 'npt', 'NPF': 'npf', .5 : 'DIO_00032', 1 : 'DIO_02048', 2 : 'DIO_04096', 4 : 'target', 8 : 'DIO_00001', 16 : 'DIO_16384', 32 : 'DIO_32768'}
