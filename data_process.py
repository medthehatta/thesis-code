"""
data_process.py

Utilities for taking messy stress/strain data and transforming it to PK1 stress
vs. Deformation Gradient.
"""
import numpy as np

def relative_to_first(arrays,axis=0,keep_first=False):
  """
  Given a list of arrays, subtracts the first array from all the others --
  effectively "translating" the other arrays to the first -- and pops the first
  array from the list (since now we know its value will be zero relative to
  itself).
  The ``axis`` argument specifies the axis to translate.
  """
  # swap the axes so the one that we're translating is the first
  transposed = arrays.swapaxes(0,axis)
  # optionally keep the origin in the list of vertices
  if keep_first:
    translated = (transposed - transposed[0])
  else:
    translated = (transposed - transposed[0])[1:]
  # swap the axes back and return
  return translated.swapaxes(0,axis)

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
  return relative_to_first(simplices,axis=1)

def split_string_with_delimiters(string,delimiters,retype=str):
  """
  Given a list of ``delimiters`` in hierarchical order, splits ``string`` into
  a nested list of strings.
  """
  if len(delimiters)>0:
    return [split_string_with_delimiters(s,delimiters[1:],retype) for 
            s in string.split(delimiters[0])]
  else:
    return retype(string)

def numpy_array_from_string(string,delimiters=STD_DELIMITERS):
  """
  Given a ``string`` and a list of ``delimiters``, reads that string into a
  ``len(delimiters)``-dimensional numpy array.

  This routine assumes the data consists of floating-point numbers.
  """
  nested_list = split_string_with_delimiters(string.strip(),
                                             delimiters,retype=float)
  return np.array(nested_list)

def numpy_array_from_file(filename,delimiters=STD_DELIMITERS):
  """
  Given a ``filename`` and a list of ``delimiters``, reads that file into a
  ``len(delimiters)``-dimensional numpy array.

  This routine assumes the data consists of floating-point numbers.
  """
  data = ''.join(open(filename))
  if data:
    return numpy_array_from_string(data,delimiters)

def join_with_nested_delimiters(arr,delim=STD_DELIMITERS):
  """
  Inverse of split_string_with_delimiters.

  Join a nested array with delimiters for each level.
  """
  if len(delim)>1:
    return delim[0].join([join_with_nested_delimiters(a,delim[1:]) for a in arr])
  else:
    return delim[0].join(map(str,arr))


STD_DELIMITERS = ['\n\n','\n',' ']
