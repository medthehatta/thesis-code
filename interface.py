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

def argument_parse(argv=None):
  """
  Returns the argument namespace from the given values.
  """
  parser_description = 'Performs a constrained least-squares fit to'\
                       'tissue data provided in a PK1 stress file and'\
                       'a deformation gradient file aligned with it.  '\
                       'Attempts to produce a fit which is Drucker stable'\
                       'in the strain regime specified.'
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

  parser.add_argument('-o','--outfile',
                      help='Pickle filename to store data in.')

  if argv is not None:
    parsed =  parser.parse_args(argv)
  else:
    parsed = parser.parse_args()

  # Read in the actual data: stress vs. strain
  DELIMS = ['\n\n','\n',' ']
  stresses = dat.numpy_array_from_file(parsed.stress,DELIMS)
  deformations = dat.numpy_array_from_file(parsed.deformation,DELIMS)

  # Read in the models that will be tried.  Each 'model' consists of the actual
  # model itself (file and function), initial parameter guesses, and
  # deformation bounds.
  models = yaml.load(open(parsed.model))

  return (stresses, deformations, models, parser.outfile)

def run_from_args(stresses, deformations, models, outfile=None):
  """
  Perform a fit for each given model with the given stress and deformation
  data.  Presumably this is called with the return value from
  ``argument_parse``.
  """
  DELIMS = [' ']
  FITS = {}
  # For each model specified, perform the fit and store the data
  for m in models:
    if not m.get('ignore'):
      # Get the tissue model function
      model_module = imp.load_source('model',m['model module'])
      model = getattr(model_module,m['model stress'])
      model_D = getattr(model_module,m['model stiffness'])

      # Get the deformation parameterization
      defmap_module = imp.load_source('defmap',m['strain module'])
      defmap = getattr(defmap_module,m['strain function'])

      # Set the desired stable deformation region
      low = dat.numpy_array_from_string(m['strain low'],DELIMS)
      high = dat.numpy_array_from_string(m['strain high'],DELIMS)

      # Set the monte carlo sample density
      pts_density = float(m['sample density'])

      # Get the lagrange multiplier for the penalty function
      lam = float(m['lagrange multiplier'])

      # Define the cost function for the optimization
      cost = lambda p: fit.data_leastsqr(deformations,stresses,model,*p) +\
                       lam*fit.positive_definite_penalty(model_D,defmap,low,high,*p,
                                         ptsdensity=pts_density)

      # Set the initial guess for the fit
      initial = dat.numpy_array_from_string(m['initial'],DELIMS)

      # Perform the fit and store the data
      print("\n\n")
      print("Performing fit for {0}:".format(m['name']))
      print("{0}\n".format(m))
      fit_data = so.fmin(cost,initial,retall=True)
      FITS[m['name']]={'name':m['name'],
                       'parameters':fit_data,
                       'model':model,
                       'model_D':model_D,
                       'deformations':deformations,
                       'stresses':stresses}
      print("Final parameter values: {0}".format(fit_data[0]))

      # Save data to pickle file
      # TODO: this should be saved in a format matlab can read
      if outfile is not None:
        print("Saving output to: {0}".format(outfile+m['name'].replace(' ','_')))
        pickle.dump(open(outfile,'wb'))

  return FITS




