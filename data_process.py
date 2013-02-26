"""
data_process.py

Utilities for taking messy stress/strain data and transforming it to PK1 stress
vs. Deformation Gradient.
"""
import numpy as np

def relative_to_first(arrays):
  """
  Given a list of arrays, subtracts the first array from all the others --
  effectively "translating" the other arrays to the first -- and pops the first
  array from the list (since now we know its value will be zero relative to
  itself).
  """
  return (arrays - arrays[0])[1:]

def translate_simplices(simplices):
  """
  Given a list of simplices (a list of square matrices whose rows are the
  positions of the vertices), subtracts the position of the first vertex from
  each simplex in the list, leaving three points relative to the last which is
  assumed to lie on the origin.

  I.E., it translates every simplex in the list so one vertex is at the origin.
  """
  # We are given an array with the following indices:
  # [simplex index, vertex index, vertex coordinate index]
  # We need to subtract the first vertex from each of the other vertices, so we
  # transpose the array so the vertex index is first, return the array
  # relative_to_first, and then transpose back.
  transposed = simplices.swapaxes(0,1)

  # Translate
  translated = relative_to_first(transposed)

  # Untranspose and return
  return translated.swapaxes(0,1)

