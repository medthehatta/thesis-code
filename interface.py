"""
interface.py

Handles option parsing and general user interaction.
"""
import yaml
import argparse
import data_process as dat
import datafit as fit
import imp
import scipy.optimize as so
from operator import and_

def argument_parse(argv=None):
  """
  Returns the argument namespace from the given values.
  """
  parser_description = "Performs a constrained least-squares fit to"\
                       "tissue data provided in a PK1 stress file and"\
                       "a deformation gradient file aligned with it.  "\
                       "Attempts to produce a fit which is Drucker stable"\
                       "in the strain regime specified."
  parser = argparse.ArgumentParser(description=parser_description)

  parser.add_argument('-P','--stress',
                      default='P.txt', 
                      help='File containing stress data.')

  parser.add_argument('-F','--deformation',
                      default='F.txt',
                      help='File containing deformation data.')

  parser.add_argument('-M','--model',
                      help='File containing functions for computing stress '\
                           'and stiffness of the desired model.')

  if argv is not None:
    parsed =  parser.parse_args(argv)
  else:
    parsed = parser.parse_args()

  # Read in the actual data: stress vs. strain
  DELIMS = ['\n\n','\n',' ']
  stresses = dat.numpy_array_from_file(parsed.stress,DELIMS)
  deformations = dat.numpy_array_from_file(parsed.deformation,DELIMS)

  # Read in the models that will be tried.  Each "model" consists of the actual
  # model itself (file and function), initial parameter guesses, and
  # deformation bounds.
  models = yaml.load(open(parsed.model))

  return (stresses, deformations, models)


