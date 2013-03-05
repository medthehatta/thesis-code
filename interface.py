"""
interface.py

Handles option parsing and general user interaction.
"""
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

  parser.add_argument('-B','--bounds',
                      help='File containing desired stable F rectangle.')

  parser.add_argument('-M','--model',
                      help='File containing functions for computing stress '\
                           'and stiffness of the desired model.')

  parser.add_argument('-I','--initial',
                      help='File containing initial guesses for the '\
                           'paramter vector')

  if argv is not None:
    parsed =  parser.parse_args(argv)
  else:
    parsed = parser.parse_args()

  STD_DELIMITERS = ['\n\n','\n',' ']

  arguments = [parsed.bounds, parsed.deformation, 
               parsed.stress, parsed.initial,
               parsed.model]

  [bounds, deformations, stresses, initial] = \
      [dat.numpy_array_from_file(p,STD_DELIMITERS) for p in arguments[:-1]]

  model = imp.load_source("model", arguments[-1])

  return {'bounds':bounds, 'deformations':deformations, 
          'stresses':stresses, 'model':model, 'initial':initial}

def perform_fit(argv=None):
  """
  Tries to do the requested fit.

  TODO: The fit doesn't converge in general, and also doesn't support the
  penalty method yet.
  """
  setup = argument_parse(argv)

  # Convenient names
  F = setup['deformations']
  P = setup['stresses']
  B = setup['bounds']
  I = setup['initial']
  M = setup['model']

  # We need everything for this to work
  if reduce(and_, [x is not None for x in [F,P,B,I,M]]):
    cost = lambda p: fit.data_leastsqr(F,P,M.model,*p)
    results = [so.fmin(cost,*i,retall=True) for i in I]

  return results
  




