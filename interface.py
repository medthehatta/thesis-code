"""
interface.py

Handles option parsing and general user interaction.
"""
import argparse
import data_process

def argument_parse(argv=None):
  """
  Returns the argument namespace from the given values.
  """
  parser_description = "Performs a constrained least-squares fit to tissue data provided in a PK1 stress file and a deformation gradient file aligned with it.  Attempts to produce a fit which is Drucker stable in the strain regime specified."
  parser = argparse.ArgumentParser(description=parser_description)

  parser.add_argument('-P','--stress',
                      default='P.txt', 
                      help='File containing stress data.')

  parser.add_argument('-F','--deformation',
                      default='F.txt',
                      help='File containing deformation data.')

  parser.add_argument('-B','--bounds',
                      help='File containing desired stable F rectangle.')

  if argv is not None:
    return parser.parse_args(argv)
  else:
    return parser.parse_args()

def load_data(deformation_file, stress_file):
  """
  Returns arrays of deformation gradient and PK1 stress tensors.
  """
  F = data_process.numpy_array_from_file(deformation_file)
  P = data_process.numpy_array_from_file(stress_file)
  return (F,P)

