#!/usr/bin/env python

from __future__ import print_function, division

import shutil
import subprocess
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import re
import os

import StringIO

from .molecule import Molecule
from .nifty import eqcgmx, fqcgmx, bohr2ang, getWorkQueue, queue_up_src_dest

#=============================#
#| Useful TeraChem functions |#
#=============================#

def edit_tcin(fin=None, fout=None, options=None, defaults=None):
    """
    Parse, modify, and/or create a TeraChem input file.

    Parameters
    ----------
    fin : str, optional
        Name of the TeraChem input file to be read
    fout : str, optional
        Name of the TeraChem output file to be written, if desired
    options : dict, optional
        Dictionary of options to overrule TeraChem input file. Pass None as value to delete a key.
    defaults : dict, optional
        Dictionary of options to add to the end

    Returns
    -------
    dictionary
        Keys mapped to values as strings.  Certain keys will be changed to integers (e.g. charge, spinmult).
        Keys are standardized to lowercase.
    """
    if defaults is None:
        defaults = {}
    if options is None:
        options = {}
    intkeys = ['charge', 'spinmult']
    Answer = OrderedDict()
    # Read from the input if provided
    if fin is not None:
        for line in open(fin).readlines():
            line = line.split("#")[0].strip()
            if len(line) == 0: continue
            if line == 'end': break
            s = line.split(' ', 1)
            k = s[0].lower()
            v = s[1].strip()
            if k == 'coordinates':
                if not os.path.exists(v.strip()):
                    raise RuntimeError("TeraChem coordinate file does not exist")
            if k in intkeys:
                v = int(v)
            if k in Answer:
                raise RuntimeError("Found duplicate key in TeraChem input file: %s" % k)
            Answer[k] = v
    # Replace existing keys with ones from options
    for k, v in options.items():
        Answer[k] = v
    # Append defaults to the end
    for k, v in defaults.items():
        if k not in Answer.keys():
            Answer[k] = v
    for k, v in Answer.items():
        if v is None:
            del Answer[k]
    # Print to the output if provided
    havekeys = []
    if fout is not None:
        with open(fout, 'w') as f:
            # If input file is provided, try to preserve the formatting
            if fin is not None:
                for line in open(fin).readlines():
                    # Find if the line contains a key
                    haveKey = False
                    uncomm = line.split("#", 1)[0].strip()
                    # Don't keep anything past the 'end' keyword
                    if uncomm.lower() == 'end': break
                    if len(uncomm) > 0:
                        haveKey = True
                        comm = line.split("#", 1)[1].replace('\n','') if len(line.split("#", 1)) == 2 else ''
                        s = line.split(' ', 1)
                        w = re.findall('[ ]+',uncomm)[0]
                        k = s[0].lower()
                        if k in Answer:
                            line_out = k + w + str(Answer[k]) + comm
                            havekeys.append(k)
                        else:
                            line_out = line.replace('\n', '')
                    else:
                        line_out = line.replace('\n', '')
                    print(line_out, file=f)
            for k, v in Answer.items():
                if k not in havekeys:
                    print("%-15s %s" % (k, str(v)), file=f)
    return Answer

def set_tcenv():
    if 'TeraChem' not in os.environ:
        raise RuntimeError('Please set TeraChem environment variable')
    TCHome = os.environ['TeraChem']
    os.environ['PATH'] = os.path.join(TCHome,'bin')+":"+os.environ['PATH']
    os.environ['LD_LIBRARY_PATH'] = os.path.join(TCHome,'lib')+":"+os.environ['LD_LIBRARY_PATH']

def load_tcin(f_tcin):
    tcdef = OrderedDict()
    tcdef['convthre'] = "3.0e-6"
    tcdef['threall'] = "1.0e-13"
    tcdef['scf'] = "diis+a"
    tcdef['maxit'] = "50"
    # tcdef['dftgrid'] = "1"
    # tcdef['precision'] = "mixed"
    # tcdef['threspdp'] = "1.0e-8"
    tcin = edit_tcin(fin=f_tcin, options={'run':'gradient'}, defaults=tcdef)
    return tcin

#====================================#
#| Classes for external codes used  |#
#| to calculate energy and gradient |#
#====================================#
stored_calcs = OrderedDict()

class Engine(object):
    def __init__(self, molecule):
        if len(molecule) != 1:
            raise RuntimeError('Please pass only length-1 molecule objects to engine creation')
        self.M = deepcopy(molecule)
        # self.stored_calcs = OrderedDict()

    # def __deepcopy__(self, memo):
    #     return copy(self)

    def calc(self, coords, dirname):
        global stored_calcs
        coord_hash = hash(coords.tostring())
        if coord_hash in stored_calcs:
            energy = stored_calcs[coord_hash]['energy']
            gradient = stored_calcs[coord_hash]['gradient']
        else:
            if not os.path.exists(dirname): os.makedirs(dirname)
            energy, gradient = self.calc_new(coords, dirname)
            stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
        return energy, gradient

    def clearCalcs(self):
        global stored_calcs
        stored_calcs = OrderedDict()

    def calc_new(self, coords, dirname):
        raise NotImplementedError("Not implemented for the base class")

    def calc_wq(self, coords, dirname):
        coord_hash = hash(coords.tostring())
        if coord_hash in stored_calcs:
            return
        else:
            self.calc_wq_new(coords, dirname)

    def calc_wq_new(self, coords, dirname):
        raise NotImplementedError("Work Queue is not implemented for this class")

    def read_wq(self, coords, dirname):
        global stored_calcs
        coord_hash = hash(coords.tostring())
        if coord_hash in stored_calcs:
            energy = stored_calcs[coord_hash]['energy']
            gradient = stored_calcs[coord_hash]['gradient']
        else:
            if not os.path.exists(dirname):
                raise RuntimeError("In read_wq, %s doesn't exist" % dirname)
            energy, gradient = self.read_wq_new(coords, dirname)
            stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
        return energy, gradient

    def read_wq_new(self, coords, dirname):
        raise NotImplementedError("Work Queue is not implemented for this class")

    def number_output(self, dirname, calcNum):
        return

class Blank(Engine):
    """
    Always return zero energy and gradient.
    """
    def __init__(self, molecule):
        super(Blank, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        energy = 0.0
        gradient = np.zeros(len(coords), dtype=float)
        return energy, gradient

class TeraChem(Engine):
    """
    Run a TeraChem energy and gradient calculation.
    """
    def __init__(self, molecule, tcin):
        self.tcin = tcin.copy()
        super(TeraChem, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        guesses = []
        have_guess = False
        for f in ['c0', 'ca0', 'cb0']:
            if os.path.exists(os.path.join(dirname, 'scr', f)):
                shutil.copy2(os.path.join(dirname, 'scr', f), os.path.join(dirname, f))
                guesses.append(f)
                have_guess = True
        # This is for when we start geometry optimizations
        # and we have a guess prepped and ready to go.
        if not have_guess and 'guess' in self.tcin:
            for f in self.tcin['guess'].split():
                if os.path.exists(f):
                    shutil.copy2(f, dirname)
                    guesses.append(f)
                    have_guess = True
                else:
                    del self.tcin['guess']
                    have_guess = False
                    break
        self.tcin['coordinates'] = 'start.xyz'
        self.tcin['run'] = 'gradient'
        if have_guess:
            self.tcin['guess'] = ' '.join(guesses)
            self.tcin['purify'] = 'no'
            self.tcin['mixguess'] = "0.0"
        edit_tcin(fout="%s/run.in" % dirname, options=self.tcin)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M[0].write(os.path.join(dirname, 'start.xyz'))
        # Run TeraChem
        subprocess.call('terachem run.in > run.out', cwd=dirname, shell=True)
        # Extract energy and gradient
        subprocess.call("awk '/FINAL ENERGY/ {p=$3} /Correlation Energy/ {p+=$5} END {printf \"%.10f\\n\", p}' run.out > energy.txt", cwd=dirname, shell=True)
        subprocess.call("awk '/Gradient units are Hartree/,/Net gradient/ {if ($1 ~ /^-?[0-9]/) {print}}' run.out > grad.txt", cwd=dirname, shell=True)
        energy = float(open(os.path.join(dirname,'energy.txt')).readlines()[0].strip())
        gradient = np.loadtxt(os.path.join(dirname,'grad.txt')).flatten()
        return energy, gradient

    def calc_wq_new(self, coords, dirname):
        # Run TeraChem
        wq = getWorkQueue()
        if not os.path.exists(dirname): os.makedirs(dirname)
        scrdir = os.path.join(dirname, 'scr')
        if not os.path.exists(scrdir): os.makedirs(scrdir)
        guesses = []
        have_guess = False
        unrestricted = self.tcin['method'][0] == 'u'
        if unrestricted:
            guessfnms = ['ca0', 'cb0']
        else:
            guessfnms = ['c0']
        for f in ['c0', 'ca0', 'cb0']:
            if f not in guessfnms: continue
            if os.path.exists(os.path.join(dirname, 'scr', f)):
                shutil.move(os.path.join(dirname, 'scr', f), os.path.join(dirname, f))
                guesses.append(f)
            if os.path.exists(os.path.join(dirname, f)):
                if f not in guesses: guesses.append(f)
        # Check if all the appropriate guess files have been found
        # and moved to "dirname"
        have_guess = (guesses == guessfnms)
        # This is for when we start geometry optimizations
        # and we have a guess prepped and ready to go.
        if not have_guess and 'guess' in self.tcin:
            for f in self.tcin['guess'].split():
                if os.path.exists(f):
                    shutil.copy2(f, dirname)
                    guesses.append(f)
                    have_guess = True
                else:
                    del self.tcin['guess']
                    have_guess = False
                    break
        self.tcin['coordinates'] = 'start.xyz'
        self.tcin['run'] = 'gradient'
        # For queueing up jobs, delete GPU key and let the worker decide
        self.tcin['gpus'] = None
        if have_guess:
            self.tcin['guess'] = ' '.join(guesses)
            self.tcin['purify'] = 'no'
            self.tcin['mixguess'] = "0.0"
        tcopts = edit_tcin(fout="%s/run.in" % dirname, options=self.tcin)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M[0].write(os.path.join(dirname, 'start.xyz'))
        in_files = [('%s/run.in' % dirname, 'run.in'), ('%s/start.xyz' % dirname, 'start.xyz')]
        out_files = [('%s/run.out' % dirname, 'run.out')]
        if have_guess:
            for g in guesses:
                in_files.append((os.path.join(dirname, g), g))
        for g in guessfnms:
            out_files.append((os.path.join(dirname, 'scr', g), os.path.join('scr', g)))
        queue_up_src_dest(wq, "%s/runtc run.in &> run.out" % rootdir, in_files, out_files, verbose=False)

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,'start.xyz'), os.path.join(dirname,'start_%03i.xyz' % calcNum))
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def read_wq_new(self, coords, dirname):
        # Extract energy and gradient
        subprocess.call("awk '/FINAL ENERGY/ {p=$3} /Correlation Energy/ {p+=$5} END {printf \"%.10f\\n\", p}' run.out > energy.txt", cwd=dirname, shell=True)
        subprocess.call("awk '/Gradient units are Hartree/,/Net gradient/ {if ($1 ~ /^-?[0-9]/) {print}}' run.out > grad.txt", cwd=dirname, shell=True)
        energy = float(open(os.path.join(dirname,'energy.txt')).readlines()[0].strip())
        gradient = np.loadtxt(os.path.join(dirname,'grad.txt')).flatten()
        return energy, gradient

class Psi4(Engine):
    """
    Run a Psi4 energy and gradient calculation.
    """
    def __init__(self, molecule=None):
        # molecule.py can not parse psi4 input yet, so we use self.load_psi4_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]
        super(Psi4, self).__init__(molecule)
        self.threads = None

    def nt(self):
        if self.threads is not None:
            return " -n %i" % self.threads
        else:
            return ""

    def set_nt(self, threads):
        self.threads = threads

    def load_psi4_input(self, psi4in):
        """ Psi4 input file parser, only support xyz coordinates for now """
        coords = []
        elems = []
        found_molecule, found_geo, found_gradient = False, False, False
        psi4_temp = [] # store a template of the input file for generating new ones
        for line in open(psi4in):
            if 'molecule' in line:
                found_molecule = True
                psi4_temp.append(line)
            elif found_molecule is True:
                ls = line.split()
                if len(ls) == 4:
                    if found_geo == False:
                        found_geo = True
                        psi4_temp.append("$!geometry@here")
                    # parse the xyz format
                    elems.append(ls[0])
                    coords.append(ls[1:4])
                else:
                    psi4_temp.append(line)
                    if '}' in line:
                        found_molecule = False
            else:
                psi4_temp.append(line)
            if "gradient(" in line:
                found_gradient = True
        if found_gradient == False:
            raise RuntimeError("Psi4 inputfile %s should have gradient() command." % psi4in)
        self.M = Molecule()
        self.M.elem = elems
        self.M.xyzs = [np.array(coords, dtype=np.float64)]
        self.psi4_temp = psi4_temp

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        # Write Psi4 input.dat
        with open(os.path.join(dirname, 'input.dat'), 'w') as outfile:
            for line in self.psi4_temp:
                if line == '$!geometry@here':
                    for e, c in zip(self.M.elem, self.M.xyzs[0]):
                        outfile.write("%-7s %13.7f %13.7f %13.7f\n" % (e, c[0], c[1], c[2]))
                else:
                    outfile.write(line)
        # Run Psi4
        subprocess.call('psi4%s input.dat' % self.nt(), cwd=dirname, shell=True)
        # Read energy and gradients from Psi4 output
        energy, gradient = self.parse_psi4_output(os.path.join(dirname, 'output.dat'))
        return energy, gradient

    def parse_psi4_output(self, psi4out):
        """ read an output file from Psi4 """
        energy, gradient = None, None
        with open(psi4out) as outfile:
            found_grad = False
            found_num_grad = False
            for line in outfile:
                line_strip = line.strip()
                if line_strip.startswith('*'):
                    # this works for CCSD and CCSD(T) total energy
                    ls = line_strip.split()
                    if len(ls) > 4 and ls[2] == 'total' and ls[3] == 'energy':
                        energy = float(ls[-1])
                elif line_strip.startswith('Total Energy'):
                    # this works for DF-MP2 total energy
                    ls = line_strip.split()
                    if ls[-1] == '[Eh]':
                        energy = float(ls[-2])
                    else:
                        # this works for HF and DFT total energy
                        try:
                            energy = float(ls[-1])
                        except:
                            pass
                elif line_strip == '-Total Gradient:' or line_strip == '-Total gradient:':
                    # this works for most of the analytic gradients
                    found_grad = True
                    gradient = []
                elif found_grad is True:
                    ls = line_strip.split()
                    if len(ls) == 4:
                        if ls[0].isdigit():
                            gradient.append([float(g) for g in ls[1:4]])
                    else:
                        found_grad = False
                        found_num_grad = False
                elif line_strip == 'Gradient written.':
                    # this works for CCSD(T) gradients computed by numerical displacements
                    found_num_grad = True
                    print("found num grad")
                elif found_num_grad is True and line_strip.startswith('------------------------------'):
                    for _ in range(4):
                        line = next(outfile)
                    found_grad = True
                    gradient = []
        if energy is None:
            raise RuntimeError("Psi4 energy is not found in %s, please check." % psi4out)
        if gradient is None:
            raise RuntimeError("Psi4 gradient is not found in %s, please check." % psi4out)
        gradient = np.array(gradient, dtype=np.float64).ravel()
        return energy, gradient



class QChem(Engine):
    def __init__(self, molecule):
        super(QChem, self).__init__(molecule)
        self.qcdir = False
        self.threads = None

    def nt(self):
        if self.threads is not None:
            return " -nt %i" % self.threads
        else:
            return ""

    def set_nt(self, threads):
        self.threads = threads

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M.edit_qcrems({'jobtype':'force'})
        self.M[0].write(os.path.join(dirname, 'run.in'))
        # Run Qchem
        if self.qcdir:
            subprocess.call('qchem%s run.in run.out run.d > run.log 2>&1' % self.nt(), cwd=dirname, shell=True)
        else:
            subprocess.call('qchem%s run.in run.out run.d > run.log 2>&1' % self.nt(), cwd=dirname, shell=True)
            # Assume reading the SCF guess is desirable
            self.qcdir = True
            self.M.edit_qcrems({'scf_guess':'read'})
        M1 = Molecule('%s/run.out' % dirname)
        energy = M1.qm_energies[0]
        gradient = M1.qm_grads[0].flatten()
        return energy, gradient

    def calc_wq_new(self, coords, dirname):
        wq = getWorkQueue()
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file<
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M.edit_qcrems({'jobtype':'force'})
        self.M[0].write(os.path.join(dirname, 'run.in'))
        in_files = [('%s/run.in' % dirname, 'run.in')]
        out_files = [('%s/run.out' % dirname, 'run.out'), ('%s/run.log' % dirname, 'run.log')]
        if self.qcdir:
            raise RuntimeError("--qcdir currently not supported with Work Queue")
        queue_up_src_dest(wq, "qchem%s run.in run.out &> run.log" % self.nt(), in_files, out_files, verbose=False)

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def read_wq_new(self, coords, dirname):
        M1 = Molecule('%s/run.out' % dirname)
        energy = M1.qm_energies[0]
        gradient = M1.qm_grads[0].flatten()
        return energy, gradient

class Gromacs(Engine):
    def __init__(self, molecule):
        super(Gromacs, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        try:
            from forcebalance.gmxio import GMX
        except ImportError:
            raise ImportError("ForceBalance is needed to compute energies and gradients using Gromacs.")
        if not os.path.exists(dirname): os.makedirs(dirname)
        Gro = Molecule("conf.gro")
        Gro.xyzs[0] = coords.reshape(-1,3) * bohr2ang
        cwd = os.getcwd()
        shutil.copy2("topol.top", dirname)
        shutil.copy2("shot.mdp", dirname)
        os.chdir(dirname)
        Gro.write("coords.gro")
        G = GMX(coords="coords.gro", gmx_top="topol.top", gmx_mdp="shot.mdp")
        EF = G.energy_force()
        Energy = EF[0, 0] / eqcgmx
        Gradient = EF[0, 1:] / fqcgmx
        os.chdir(cwd)
        return Energy, Gradient


class Molpro(Engine):
    """
    Run a Molpro energy and gradient calculation.
    """
    def __init__(self, molecule=None):
        # molecule.py can not parse molpro input yet, so we use self.load_molpro_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]
        super(Molpro, self).__init__(molecule)
        self.threads = None
        self.molproExePath = None

    def molproExe(self):
        if self.molproExePath is not None:
            return self.molproExePath
        else:
            return "molpro"

    def set_molproexe(self, molproExePath):
        self.molproExePath = molproExePath

    def nt(self):
        if self.threads is not None:
            return " -n %i" % self.threads
        else:
            return ""

    def set_nt(self, threads):
        self.threads = threads

    def load_molpro_input(self, molproin):
        """ Molpro input file parser, only support xyz coordinates for now """
        coords = []
        elems = []
        labels = []
        found_molecule, found_geo, found_gradient = False, False, False
        molpro_temp = [] # store a template of the input file for generating new ones
        for line in open(molproin):
            if 'geometry' in line:
                found_molecule = True
                molpro_temp.append(line)
            elif found_molecule is True:
                ls = line.split()
                if len(ls) == 4:
                    if found_geo == False:
                        found_geo = True
                        molpro_temp.append("$!geometry@here")
                    # parse the xyz format
                    elem = re.search('[A-Z][a-z]*',ls[0]).group(0)
                    elems.append( elem ) # grabs the element
                    labels.append( ls[0].split(elem)[-1] ) # grabs label after element specification
                    coords.append(ls[1:4]) # grabs the coordinates
                else:
                    molpro_temp.append(line)
                    if '}' in line:
                        found_molecule = False
            else:
                molpro_temp.append(line)
            if "force" in line:
                found_gradient = True
        if found_gradient == False:
            raise RuntimeError("Molpro inputfile %s should have force command." % molproin)
        self.M = Molecule()
        self.M.elem = elems
        self.M.xyzs = [np.array(coords, dtype=np.float64)]
        self.labels = labels
        self.molpro_temp = molpro_temp

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        # Write Molpro run.mol
        with open(os.path.join(dirname, 'run.mol'), 'w') as outfile:
            for line in self.molpro_temp:
                if line == '$!geometry@here':
                    for e, lab, c in zip(self.M.elem, self.labels, self.M.xyzs[0]):
                        outfile.write("%s%-7s %13.7f %13.7f %13.7f\n" % (e, lab, c[0], c[1], c[2]))
                else:
                    outfile.write(line)
        # Run Molpro
        subprocess.call('%s%s run.mol' % (self.molproExe(), self.nt()), cwd=dirname, shell=True)
        # Read energy and gradients from Molpro output
        energy, gradient = self.parse_molpro_output(os.path.join(dirname, 'run.out'))
        return energy, gradient

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def parse_molpro_output(self, molpro_out):
        """ read an output file from Molpro"""
        energy, gradient = None, None
        with open(molpro_out) as outfile:
            found_grad = False
            for line in outfile:
                line_strip = line.strip()
                fields = line_strip.split()
                if line_strip.startswith('!'):
                    # This works for RHF and RKS
                    if len(fields) == 5 and fields[-2] == 'Energy':
                        energy = float(fields[-1])
                    # This works for MP2, CCSD and CCSD(T) total energy
                    elif len(fields) == 4 and fields[1] == 'total' and fields[2] == 'energy:':
                        energy = float(fields[-1])
                elif len(fields) > 4 and fields[-4] == 'GRADIENT' and fields[-3] == 'FOR' and fields[-2] == 'STATE':
                    # this works for most of the analytic gradients
                    found_grad = True
                    gradient = []
                    # Skip three lines of header
                    next(outfile)
                    next(outfile)
                    next(outfile)
                elif found_grad is True:
                    if len(fields) == 4:
                        if fields[0].isdigit():
                            gradient.append([float(g) for g in fields[1:4]])
                    elif "Nuclear force contribution to virial" in line:
                        found_grad = False
                    else:
                        continue
        if energy is None:
            raise RuntimeError("Molpro energy is not found in %s, please check." % molpro_out)
        if gradient is None:
            raise RuntimeError("Molpro gradient is not found in %s, please check." % molpro_out)
        gradient = np.array(gradient, dtype=np.float64).ravel()
        return energy, gradient

class QCEngineAPI(Engine):
    def __init__(self, schema, program):
        try:
            import qcengine
        except ImportError:
            raise ImportError("QCEngine computation object requires the 'qcengine' package. Please pip or conda install 'qcengine'.")

        self.schema = schema
        self.program = program
        self.schema["driver"] = "gradient"

        self.M = Molecule()
        self.M.elem = schema["molecule"]["symbols"]

        # Geometry in (-1, 3) array in angstroms
        geom = np.array(schema["molecule"]["geometry"], dtype=np.float64).reshape(-1, 3) * bohr2ang
        self.M.xyzs = [geom]

        # Use or build connectivity
        if "connectivity" in schema["molecule"]:
            self.M.Data["bonds"] = sorted((x[0], x[1]) for x in schema["molecule"]["connectivity"])
            self.M.built_bonds = True
        else:
            self.M.build_bonds()
        # one additional attribute to store each schema on the opt trajectory
        self.schema_traj = []

    def calc_new(self, coords, dirname):
        import qcengine
        new_schema = deepcopy(self.schema)
        new_schema["molecule"]["geometry"] = coords.tolist()
        ret = qcengine.compute(new_schema, self.program)

        # store the schema_traj for run_json to pick up
        self.schema_traj.append(ret)

        # Unpack the erngies and gradient
        energy = ret["properties"]["return_energy"]
        gradient = np.array(ret["return_result"])
        return energy, gradient

    def calc(self, coords, dirname):
        # overwrites the calc method of base class to skip caching and creating folders
        return self.calc_new(coords, dirname)

def find_parentheses(s, paren_type = ['(', ')'], beg=0, end=None):
    """ Returns the location of the matching parentheses pairs in s.

    Given a string, s, return a dictionary of start: end pairs giving the
    indexes of the matching parentheses in s. Suitable exceptions are
    raised if s contains unbalanced parentheses.

    """

    if end is None:
        end = len(s)

    stack = []
    parentheses_locs = {}
    for i, c in enumerate(s[beg:end], beg):
        if c == paren_type[0]:
            stack.append(i)
        elif c == paren_type[1]:
            try:
                parentheses_locs[stack.pop()] = i
            except IndexError:
                raise IndexError('Too many close parentheses at index {}'
                                                                .format(i))
    if stack:
        raise IndexError('No matching close parenthesis to open parenthesis '
                         'at index {}'.format(stack.pop()))
    return parentheses_locs

def find_command_block(s, command):
    blocks = find_parentheses(s)
    command_start = s.find(command)
    command_block_start = min(filter(lambda x: x > command_start, blocks))
    command_block_end = blocks[command_block_start]
    return [command_block_start+1, command_block_end-1]

def find_option_value(s, option, beg=0, end=None):
    if end is None:
        end = len(s)
    option_start = s.find(option, beg, end)

    option_assignment = s.find("=", option_start)

    value_start = option_assignment+1 + len(s[option_assignment+1:]) - len(s[option_assignment+1:].lstrip())

    if s[value_start] == "'":
        end_quote = s.find("'", value_start+1)
        return [value_start, end_quote]
    elif s[value_start] == "[":
        paren_blocks = find_parentheses(s, ["[", "]"])
        paren_end = paren_blocks[value_start]
        return [value_start, paren_end]
    else:
        next_space = s.find(" ", value_start+1)
        return [value_start, next_space-1]

def get_structure_filename(s):
    block_start, block_end = find_command_block(s, "structure")
    print(block_start, block_end)
    value_start, value_end = find_option_value(s, "file", block_start, block_end)
    print(value_start, value_end)
    filename = s[value_start:value_end+1]
    return filename


def replace_option(s, command, option, new_option, value):
    if not isinstance(command, basestring) and len(command) == 1:
        command = command[0]
    if isinstance(command, basestring):
        block_start, block_end = find_command_block(s, command)

        option_start = s.find(option, block_start, block_end)
        option_end = option_start + len(option)

        value_start, value_end = find_option_value(s, option, block_start, block_end)

        return s.replace(s[option_start:value_end+1], new_option + " = " + value)

    else:
        block_start, block_end = find_command_block(s, command[0])
        return s[0:block_start] + \
               replace_option(s[block_start:block_end],
                              command[1:],
                              option, new_option, value) + \
               s[block_end:]

def get_energy_and_gradient(content):
    """ read an output file from entos"""

    energy, gradient = None, []
    found_grad = False
    buf = StringIO.StringIO(content)
    for line in buf:
        if found_grad is True:
            fields = line.split()
            if len(fields) == 4:
                if fields[0].isdigit():
                    gradient.append(list(map(float, fields[1:4])))
            else:
                found_grad = False
        if "Gradients" in line:
            next(buf)
            found_grad = True
        if "Total energy (hartree)" in line:
            energy = float(line.split()[-1])
    if energy is None:
        print(content)
        raise RuntimeError("entos energy is not found in output, please check.")
    if not gradient:
        print(content)
        raise RuntimeError("entos gradient is not found in output, please check.")
    gradient = np.array(gradient, dtype=np.float64).ravel()
    return energy, gradient

class Entos(Engine):
    """ An Entos energy and gradient engine """

    def __init__(self, filename, threads = None, exe = None):
        # create a fake molecule
        molecule = Molecule()
        molecule.elem = ['H']
        molecule.xyzs = [[[0, 0, 0]]]
        super(Entos, self).__init__(molecule)
        self.threads = threads
        self.exe = exe
        self.init_from_input(filename)
        self.entos_output = None

    @property
    def exe(self):
        """ Executable for entos """
        return self._exe

    @exe.setter
    def exe(self, exe):
        """ Set executable for entos. If None, entos will be used """
        self._exe = "entos"
        if exe is not None:
            self._exe = exe

    def nt(self):
        """ string form of number of threads for entos exe """
        if self.threads is not None:
            return " -n %i" % self.threads
        else:
            return ""

    def init_from_input(self, filename):
        """ Initialise entos engine from a template file """
        self.M = Molecule()
        self.input_file_content = open(filename).read()
        xyz_filename = get_structure_filename(self.input_file_content)
        stuff = Molecule().read_xyz(xyz_filename.replace("'",""))
        self.M.elem = stuff['elem']
        self.M.xyzs = stuff['xyzs']
        self.M.comms = stuff['comms']

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        entos_xyz = [[self.M.elem[i]] + xyz
                     for i, xyz in enumerate(self.M.xyzs[0].tolist())]
        calc_input_content = replace_option(self.input_file_content,
                                            'structure',
                                            'file',
                                            'xyz',
                                            str(entos_xyz))
        
        p = subprocess.Popen([self.exe, self.nt()],
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         cwd=dirname)
        self.entos_output = p.communicate(input=calc_input_content)[0]

        # Read energy and gradients from entos output
        return get_energy_and_gradient(self.entos_output)

    def number_output(self, dirname, calcNum):
        if self.entos_output is None:
            raise RuntimeError('No entos output to write')
        f = open(os.path.join(dirname,'run_%03i.out' % calcNum))
        f.write(self.entos_output)
        f.close()