from xml.dom.minidom import parseString
import re
import numpy as np

position_rex = re.compile(r'translate\(([\-0-9\.]+),([\-0-9\.]+)\)')

def get_trues_falses(path):
    data = open(path).read()
    dom = parseString(data)
    clones = dom.getElementsByTagName("use")

    trues_pts = [c.getAttribute('transform') 
                 for c in clones if c.getAttribute('xlink:href')=='#green']

    falses_pts = [c.getAttribute('transform') 
                  for c in clones if c.getAttribute('xlink:href')=='#red']

    trues = [list(map(float,position_rex.findall(p)[0])) for p in trues_pts]
    falses = [list(map(float,position_rex.findall(p)[0])) for p in falses_pts]

    return (np.array(trues),np.array(falses))

