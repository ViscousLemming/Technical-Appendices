from sympy.logic import simplify_logic
from sympy.logic.boolalg import Equivalent
from sympy.abc import symbols
from sympy.logic.inference import satisfiable
from sympy import sympify
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import time
import queue
cuda_if_available = 'cuda' if torch.cuda.is_available() else 'cpu'
#cuda_if_available = 'cpu'
device = cuda_if_available

#NOTE: To change experiment parameters, go to the end of this file.

def nf_to_parts(nf):
    """
    Splits SymPy statement into parts, stores in list.

    Parameters
    ----------
    nf : a SymPy statement.
    
    Returns
    -------
    parts : a list of SymPy statements.
    """
    nf_str = str(nf)
    parts = []
    current_part = ''
    primed = False
    for character in list(nf_str):
        if character == '(': primed = True
        if primed:
            current_part += character
            if character == ')': 
                parts.append(current_part)
                current_part = ''
                primed = False
    return parts

def parts_to_cnf(parts):
    """
    Reverses nf_to_parts function.

    Parameters
    ----------
    parts : a list of SymPy statements.
    
    Returns
    -------
    statement : a SymPy statement.
    """
    statement = ''
    for ix in range(0, len(parts)-1):
        statement += parts[ix] + ' & '
    statement += parts[len(parts)-1]
    return statement

def matrix_to_dnf(m):
    """
    Converts a PyTorch tensor to a dnf SymPy statement.

    Parameters
    ----------
    m : PyTorch tensor.
    
    Returns
    -------
    dnf : a SymPy statement in dnf.
    """
    dnf = ''
    (rows, cols, value) = m.shape
    for ir in range(0, rows):
        part = '('
        count = 0
        for ic in range(0, cols):
            atom = ''
            if m[ir,ic, 0] == True or m[ir,ic, 1] == True: 
                num = ''
                if count != 0: atom += ' & '
                if m[ir,ic, 0] == True: atom += '~'
                if ic < 10: num += '0'
                num += str(ic)
                atom += 'x' + num
                part += atom
                count += 1
        if len(part) > 1:
            part += ')'
            if ir < rows - 1: part += ' | '
            dnf += part
    return dnf

def matrix_to_cnf(m):
    """
    Converts a PyTorch tensor to a cnf SymPy statement.

    Parameters
    ----------
    m : PyTorch tensor.
    
    Returns
    -------
    cnf : a SymPy statement in cnf.
    """
    cnf = ''
    (rows, cols, value) = m.shape
    for ir in range(0, rows):
        part = '('
        count = 0
        for ic in range(0, cols):
            atom = ''
            if m[ir,ic, 0] == True or m[ir,ic, 1] == True: 
                num = ''
                if count != 0: atom += ' | '
                if m[ir,ic, 0] == True: atom += '~'
                if ic < 10: num += '0'
                num += str(ic)
                atom += 'x' + num
                part += atom
                count += 1
        if len(part) > 1:
            part += ')'
            if ir < rows - 1: part += ' & '
            cnf += part
    return cnf

def nf_to_matrix(nf, l, form = 'cnf', device = cuda_if_available):
    """
    Converts a SymPy statement to a PyTorch tensor.

    Parameters
    ----------
    nf : a SymPy statement
    l  : len(bits)
    form : either `dnf` or `cnf`, whatever nf is
    device : an NVidia GPU if you have one
    
    Returns
    -------
    m : PyTorch tensor.
    """
    if form == 'cnf': delineator = '&'
    else: delineator = '|'
    nf = str(nf)
    rows = 1
    cols = l
    values = 2
    for character in nf: rows += (character == delineator)
    m = torch.zeros((rows, cols, values), dtype=torch.bool, device=device, requires_grad=False)
    row = 0
    idx = 0
    while idx < len(nf):
        if nf[idx] == delineator: row += 1
        else:
            col = None
            val = 1
            if nf[idx] == '~':
                col = int(nf[idx+2] + nf[idx+3])
                val = 0
                idx += 3
            elif nf[idx] == 'x':
                col = int(nf[idx+1] + nf[idx+2])
                idx += 2
            if col != None:
                m[row,col,val] = True
        idx += 1
    return m

def power_set(parts):
    """
    Generates all possible combinations of bit value assignments.

    Parameters
    ----------
    parts : a list of strings
    
    Returns
    -------
    pset : All possible conjunctions of bits in SymPy format.
    """
    set_size = len(parts)
    pow_set_size = (int) (math.pow(2, set_size));
    counter = 0;
    j = 0;
    pset = []
    for counter in range(0, pow_set_size):
        element = ''
        for j in range(0, set_size):
            if((counter & (1 << j)) > 0):
                if len(element) > 0: element += ' & '
                element += str(parts[j]);
        pset.append(element)
    return pset

def generate_all_complete_assignments(bits):
    """
    Generates L, but without any partial assignments (saves time).

    Parameters
    ----------
    bits : a list of bit names (strings)
    
    Returns
    -------
    L : A string listing all possible statements describing complete assignments to the bits as a DNF SymPy format.
    """
    pset = power_set(bits)
    modified_pset = []
    L = ''
    for subs in pset:
        modified_subs = '('
        for character in bits:
            if len(modified_subs) > 1:
                modified_subs += ' & '
            if character not in subs:
                modified_subs += '~' + character
            else:
                modified_subs += character
        modified_pset.append(modified_subs + ')')
    for subs in modified_pset:
        if len(L) > 0:
            L += ' | '
        L += subs
    return L

def generate_L(bits):
    """
    Generates L.

    Parameters
    ----------
    bits : a list of bit names (strings)
    
    Returns
    -------
    L : A string listing all possible statements as a DNF SymPy format.
    """
    pset = power_set(bits)
    L = ''
    for p in pset:
        if len(L) > 0:
            L += ' | '
        L += generate_L(bits)
    return L

def generate_dataset(l, f):
    """
    Generates D_n.

    Parameters
    ----------
    target_bit : String of the format 'x##'
    f : A function of two integers x and y
    l : Integer equal to len(bits)
    
    Returns
    -------
    m : A tensor representation of D_n.
    """
    x_len = max(int(l / 4),1)
    y_len = max(int(l / 4),1)
    z_len = max(int(l / 2),2)
    rows = 2**(x_len + y_len)
    m = torch.zeros((rows, l, 2), dtype=torch.bool, device=device, requires_grad=False)
    row = 0
    for ix in range (0, 2**x_len):
        for iy in range (0, 2**y_len):
            iz = f(ix, iy)
            bx = format(ix, "b")[::-1]
            by = format(iy, "b")[::-1]
            bz = format(iz, "b")[::-1]
            for cx in range (0, x_len):
                try:
                    m[row, cx, int(bx[cx])] = True
                except:
                    m[row, cx, 0] = True
            for cy in range (0, y_len):
                try:
                    m[row, cy + x_len, int(by[cy])] = True
                except: 
                    m[row, cy + x_len, 0] = True
            for cz in range (0, z_len):
                try:
                    m[row, cz + x_len + y_len, int(bz[cz])] = True
                except: 
                    m[row, cz + x_len + y_len, 0] = True    
            row += 1
    return m


def obtain_relevant_statements(target_bit, c, l, device):
    """
    Returns only those statements (in CNF) pertaining to a specific bit we wish to predict.

    Parameters
    ----------
    target_bit : String of the format 'x##'
    D : Tensor
    l : Integer equal to len(X)
    device : 'cpu' or 'gpu'

    Returns
    -------
    c : Matrix where each row is a disjunction.
    """
    m_target = nf_to_matrix('(' + target_bit + '| ~' + target_bit + ')', l,'cnf' , device)
    w = 0
    list_of_statements = []
    for row in range(0, c.shape[0]):
        if torch.sum(m_target[0, :, :] * c[row, :, :]) > 0:
            w += 1
            list_of_statements.append(c[row, :, :])
    c_new = torch.zeros((w, c.shape[1], 2), dtype=torch.bool, device=device, requires_grad=False)
    for row in range(0, len(list_of_statements)):
        c_new[row,:,:] = list_of_statements[row]
    return c_new

def obtain_length(item, c):
    """
    Obtains the description length of a statement which is presently represented as a tensor.

    Parameters
    ----------
    c : a ruleset in tensor form
    item : a list of indices

    Returns
    -------
    l : the length of the cnf statement represented by item (part of c)
    """
    l = 0
    indices = torch.zeros((c.shape[0]), dtype=torch.bool)
    for ix in item:
        indices[ix] = True
    prospect = c[indices,:,:]
    l = len(matrix_to_cnf(prospect))
    return l

def obtain_necessary_disjuncts(c, S, D, L, target_bit):
    """
    A quick check to see which members of c are just by themselves necessary for c to remain a ruleset.

    Parameters
    ----------
    c : a ruleset in tensor form
    S : situations in tensor form
    D : decisions in tensor form
    L : the set of all statements in tensor form

    Returns
    -------
    l : the length of the cnf statement represented by item (part of c)
    """
    indices = torch.zeros((c.shape[0]), dtype=torch.bool)
    necessary = set({})
    if c.shape[0] > 1:
        for ix in range(0,c.shape[0]):
            indices[:] = True
            indices[ix] = False
            prospect = c[indices,:,:]
            Zprospect = obtain_extension(prospect, L)
            reconstructed_D = constrain_by_situation(Zprospect, S, target_bit)
            if not equality(D, reconstructed_D): necessary.add(ix)
    else:
        necessary.add(0)
    return necessary

def obtain_sufficient_cnf(c, S, D, L, target_bit, necessary, optimise_for='w', depth_limit = 100, time_limit = 100):
    """
    Performs an A* search (Hart, Nilsson and Bertram 1968) to find a sufficient ruleset, 
    which is either mdl or weakest possible.

    Parameters
    ----------
    c : a ruleset in tensor form
    S : situations in tensor form
    D : decisions in tensor form
    L : the set of all statements in tensor form
    target_bit : the bit which is to be predicted
    necessary : those subsentential parts of c which are absolutely necessary in all rulesets
    optimise_for : either weakness or length
    depth_limit : an optional limit on how deep the search can go. Use this if you want to rerun the experiment quickly but don't care about it being a perfect rerun (ie, if you don't have a week to wait around)
    time_limit : an optional limit on how long a search can last

    Returns
    -------
    c : old ruleset 
    prospect : new ruleset
    """
    q = queue.PriorityQueue()
    for ix in range(0,c.shape[0]):
        w, _ = w_cnf(c[ix:ix+1,:,:],L)
        item = set({ix})
        item = item.union(necessary)
        if optimise_for == 'w':
            # A* where weakness is heuristic
            q.put((1/w,item))
        else:
            # Length as heuristic
            q.put((obtain_length(item, c),item))
    indices = torch.zeros((c.shape[0]), dtype=torch.bool)
    visited = []
    timeout = time.time() + 60*time_limit #Default timeout after 200, in case something goes wrong
    while not q.empty():
        if time.time() > timeout: 
            raise ValueError('Search aborted to save time...')
        w, item = q.get()
        indices[:] = False
        for ix in item:
            indices[ix] = True
        prospect = c[indices,:,:]
        Zprospect = obtain_extension(prospect, L)
        reconstructed_D = constrain_by_situation(Zprospect, S, target_bit)
        if equality(D, reconstructed_D):
            return prospect
        elif len(item) < depth_limit: #Search depth limit imposed for convenience.
            for ix in range(0,c.shape[0]):
                if ix not in item:
                    new_item = item.copy()
                    new_item.add(ix)
                    if new_item not in visited:
                        visited.append(new_item)
                        
                        indices[:] = False
                        for ix in item:
                            indices[ix] = True
                        prospect = c[indices,:,:]
                        
                        w, _ = w_cnf(prospect,L)
                        if optimise_for == 'w':
                            # A* where weakness is heuristic
                            q.put((1/w,new_item))
                        else: 
                            # Length as heuristic
                            q.put((obtain_length(item, c),item))
    return c

def obtain_true_S(target_bit, D):
    """
    Obtains S by just deleting the parts of D we want to predict.
    
    Parameters
    ----------
    target_bit : the bit which is to be predicted
    D : decisions in tensor form

    Returns
    -------
    S : the set of all situations in tensor form
    """
    target_idx = int(target_bit[1::])
    S = D.clone()
    S[:,target_idx,:] = False
    return S

def hash_sequence(s):
    """
    A simple hash function, ignore this.
    """
    h = ''
    for element in s:
        if element[1]:
            h += '1'
        elif element[0]:
            h += '0'
    return h

def w_row_cnf(row_of_c, m_Z):
    """
    Obtains weakness of a row of c if c is CNF tensor.
    """
    flat_m_Z = torch.flatten(m_Z,  start_dim=1)
    flat_row_of_c = torch.flatten(row_of_c,  start_dim=0)
    repeated_flat_row_of_c = flat_row_of_c.repeat(flat_m_Z.shape[0],1)
    comparison = torch.logical_and(repeated_flat_row_of_c, flat_m_Z)
    comparison = torch.sum(comparison, 1) > 0
    weakness = int(torch.sum(comparison))
    return weakness, comparison

def w_row_dnf(row_of_c, m_Z):
    """
    Obtains weakness of a row of c if c is DNF tensor.
    """
    flat_m_Z = torch.flatten(m_Z,  start_dim=1)
    flat_row_of_c = torch.flatten(row_of_c,  start_dim=0)
    repeated_flat_row_of_c = flat_row_of_c.repeat(flat_m_Z.shape[0],1)
    comparison = torch.logical_and(repeated_flat_row_of_c, flat_m_Z)
    comparison = torch.sum(comparison, 1) == torch.sum(repeated_flat_row_of_c, 1)
    weakness = int(torch.sum(comparison))
    return weakness, comparison

def w_cnf(c, Z):
    """
    Obtains weakness of c if c is CNF tensor.
    """
    _, comparison = w_row_cnf(c[0,:,:], Z)
    for row in range(1,c.shape[0]):
        _, rc = w_row_cnf(c[row,:,:], Z)
        comparison = torch.logical_and(comparison,rc)
    return int(torch.sum(comparison)), comparison

def w_dnf(c, Z):
    """
    Obtains weakness of c if c is DNF tensor.
    """
    _, comparison = w_row_dnf(c[0,:,:], Z)
    for row in range(1,c.shape[0]):
        _, rc = w_row_dnf(c[row,:,:], Z)
        comparison += rc
    return int(torch.sum(comparison)), comparison

def equality(m1, m2):
    """
    Parameters
    ----------
    m1 : Torch tensor in DNF form.
    m2 : Torch tensor in DNF form.
    
    Returns
    -------
    True if they contain the same rows.
    """
    rw, rc = w_dnf(m1, m2)
    if rw == m2.shape[0] and rw == m1.shape[0]:
        return True
    else:
        return False
    
def member(r, m): 
    """
    Checks if row r exists anywhere in tensor m
    """
    r_flat = r.flatten(start_dim=1)
    val = torch.sum(r_flat)
    comparison = torch.logical_and(m, r.repeat((m.shape[0],1,1)))
    comparison_flat = comparison.flatten(start_dim=1)
    baseline_flat = m.flatten(start_dim=1)
    comparison_summed = torch.sum(comparison_flat, dim=1)
    baseline_summed = torch.sum(baseline_flat, dim=1)
    baseline_comparison = comparison_summed == baseline_summed
    val_comparison = val == comparison_summed
    final_comparison = torch.logical_and(baseline_comparison, val_comparison)
    return final_comparison.any()

def intersection(m1, m2):
    """
    Finds all rows that exist in both tensor m1 and tensor m2.
    """
    comparison = torch.zeros(m1.shape[0], dtype=torch.bool)
    for ix in range(0, comparison.shape[0]):
        comparison[ix] = member(m1[ix,:,:],m2)
    return m1[comparison,:,:]

def disjoint(m1, m2):
    #Return all m1 not in m2
    comparison = torch.zeros(m1.shape[0], dtype=torch.bool)
    for ix in range(0, comparison.shape[0]):
        comparison[ix] = member(m1[ix,:,:],m2)
    disjoint_comparison = torch.logical_not(comparison)
    return m1[disjoint_comparison,:,:]
    
def accuracy(m1, m2):
    """
    Parameters
    ----------
    m1 : Torch tensor in DNF form.
    m2 : Torch tensor in DNF form.
    
    Returns
    -------
    Treats m1 and m2 as sets, each member of which is a row.
    Cardinality of the intersection of m1 and m2 divided by the cardinality of m2.
    TP, FP, TN, FN
    """
    TP = intersection(m1,m2).shape[0] / m2.shape[0]
    FP = disjoint(m1,m2).shape[0] / m2.shape[0]
    FN = disjoint(m2,m1).shape[0] / m2.shape[0]
    return TP, FP, FN

def ruleset_accuracy(target_bit, c, S, D, Z):
    Zc = obtain_extension(c, Z)
    D_recon = constrain_by_situation(Zc, S, target_bit)
    return accuracy(D_recon, D)

def obtain_extension(c, Z):
    """Returns the extension of a statement in cnf."""
    _, comparison = w_cnf(c, Z)
    Zc = Z[comparison,:,:]
    return Zc

def obtain_extension_dnf(c, Z):
    """Returns the extension of a statement in dnf."""
    _, comparison = w_dnf(c, Z)
    Zc = Z[comparison,:,:]
    return Zc

def constrain_by_situation(Zc, S, target_bit):
    """Returns the extension of a statement in dnf, constrained by S."""
    w_temp, comp_temp = w_dnf(S, Zc)
    Zc_new = Zc[comp_temp,:,:]
    return Zc_new

def trial(number_of_examples, D_n, L, bits, depth_limit, time_limit):
    """
    Given a value of |D_k| (the number_of_examples), runs 1 trial.
    
    Parameters
    ----------
    number_of_examples : |D_k|
    D_n : the set of all correct decisions for the parent task, in tensor form
    L : set of all statements in tensor form
    bits : the set of bits to which all statements refer
    depth_limit : a search limit to save time. Default 100
    time_limit : a search limit to save time. Default 100 

    Returns
    -------
    For all returned lists, index 0 is c_w, while index 1 is c_mdl
    TP : decisions which are true positives
    FP : decisions which are false positives
    FN : decisions which are false negatives
    """
    target_bit = bits[random.randint(0,len(bits)-1)]
    indices = random.sample(range(0, D_n.shape[0]), number_of_examples)
    D_k = D_n[indices, :, :]
    p_1 = (D_k[:,int(target_bit[1::]),1] > 0).nonzero().shape[0] / D_k.shape[0]
    p_0 = (D_k[:,int(target_bit[1::]),0] > 0).nonzero().shape[0] / D_k.shape[0]
    S_n = obtain_true_S(target_bit, D_n)
    S_k = obtain_true_S(target_bit, D_k)
    smpy_S_k = matrix_to_dnf(S_k)
    smpy_D_k = matrix_to_dnf(D_k)
    smpy_c = simplify_logic(smpy_D_k, form='cnf') 
    c = nf_to_matrix(smpy_c, l, 'cnf', device)
    Zc = obtain_extension(c, L)
    reconstructed_D_k = constrain_by_situation(Zc, S_k, target_bit)
    c = obtain_relevant_statements(target_bit, c, l, device)
    necessary = obtain_necessary_disjuncts(c, S_k, D_k, L, target_bit)
    print("Nec:",len(necessary), '/', c.shape[0])
    TP = [0,0] #True positives in this trial (0 is c_w, 1 is c_mdl)
    FP = [0,0]
    FN = [0,0]
    c_w = obtain_sufficient_cnf(c, S_k, D_k, L, target_bit, necessary, 'w', depth_limit, time_limit)
    TP[0], FP[0], FN[0] = ruleset_accuracy(target_bit, c_w, S_n, D_n, L)
    c_mdl = obtain_sufficient_cnf(c, S_k, D_k, L, target_bit, necessary, 's', depth_limit, time_limit)
    TP[1], FP[1], FN[1] = ruleset_accuracy(target_bit, c_mdl, S_n, D_n, L)
    return TP, FP, FN

def average_x_trials(x, number_of_examples, D_n, L, bits, depth_limit, time_limit):
    """
    Given a value of |D_k| (the number_of_examples), runs a number of trails and then 
    averages the results.
    
    Parameters
    ----------
    x : number of trails to be run and averaged
    number_of_examples : |D_k|
    D_n : the set of all correct decisions for the parent task, in tensor form
    L : set of all statements in tensor form
    bits : the set of bits to which all statements refer
    depth_limit : a search limit to save time. Default 100
    time_limit : a search limit to save time. Default 100 

    Returns
    -------
    bTP : the best recorded portion of true positives in a trial
    generalised : the average portion of trials in which generalisation occurred
    TP : average portion of decisions which are true positives
    FP : average portion of decisions which are false positives
    FN : average portion of decisions which are false negatives
    """
    generalised = [0,0]
    TP = [0,0] #Average True Positives (0 is c_w, 1 is c_mdl)
    FP = [0,0] #Average False Negatives
    FN = [0,0] #Average False Negatives
    bTP = [0,0] #Best True Positive Result
    ix = 0 
    clear = False
    visitedbits = []
    visitedD_k = []
    while ix < x:
        # try:
        nTP, nFP, nFN = trial(number_of_examples, D_n, L, bits, depth_limit, time_limit)
        for k in range(0,len(TP)):
            generalised[k] += int(nTP[k] == 1)
            print(k, generalised[k], nTP[k] == 1, nTP[k], ix, number_of_examples)
            TP[k] += nTP[k]
            FP[k] += nFP[k]
            FN[k] += nFN[k]
            bTP[k] = max(bTP[k],nTP[k])
        ix += 1
    x = max(x,1)
    for k in range(0,len(TP)):
        generalised[k] /= x
        TP[k] /= x
        FP[k] /= x
        FN[k] /= x
    return bTP, generalised, TP, FP, FN

def iterate_trials(D_n, L, bits, trials_per_stage, depth_limit, time_limit):
    """
    Runs trials, iterating different values of D_k.
    Doesn't return anything, but prints to terminal.
    
    Parameters
    ----------
    target_bit : the bit which is to be predicted
    D_n : the set of all correct decisions for the parent task, in tensor form
    L : set of all statements in tensor form
    bits : the set of bits to which all statements refer
    traisl_per_state : the number of trials conducted for each value of |D_k|
    depth_limit : a search limit to save time. Default 100
    time_limit : a search limit to save time. Default 100 
    """
    stage = 1
    stage_limit = 8
    number_of_examples = 0
    counts = torch.sum(D_n, dim=0)
    sample_numbers = []
    bTP_w = []
    bTP_mdl = []
    TP_w = []
    TP_mdl = []
    generalised_w = []
    generalised_mdl = []
    while stage < stage_limit-1:
        stage += 1 
        number_of_examples = int(stage * D_n.shape[0] / (1*stage_limit)) #This increments the cardinality of D_k
        sample_numbers.append(number_of_examples)
        nbTP, ngeneralised, nTP, nFP, nFN = average_x_trials(trials_per_stage, number_of_examples, D_n, L, bits, depth_limit, time_limit)
        bTP_w.append(nbTP[0])
        bTP_mdl.append(nbTP[1])
        TP_w.append(nTP[0])
        TP_mdl.append(nTP[1])
        generalised_w.append(ngeneralised[0])
        generalised_mdl.append(ngeneralised[1])
        print(number_of_examples, ngeneralised, nbTP, nTP, nFP, nFN)
        plt.plot(sample_numbers,TP_w, 'r+', label="extent c_w")
        plt.plot(sample_numbers,TP_mdl, 'g+', label="extent c_mdl")
        plt.plot(sample_numbers,generalised_w, color='blue', label="rate c_w")
        plt.plot(sample_numbers,generalised_mdl, color='orange', label="rate c_mdl")
        plt.legend(loc="upper left")
        plt.show()

#Declare variables.
#For example, lambda will be {x00 ... x07, ~x00 ... ~x07}
x00, x01, x02, x03, x04, x05, x06, x07 = symbols('x00 x01 x02 x03 x04 x05 x06 x07')
bits = ['x00','x01','x02','x03','x04','x05','x06','x07']
l = len(bits)

#For efficiency, generate L without any partial assignments
L_complete_assignments_only = generate_all_complete_assignments(bits)
L = nf_to_matrix(L_complete_assignments_only, l, device) #Represent propositional logic as a tensor in order to save on computing resources

#Parameters
number_of_trials = 128 #Number of trials to run per value of |D_k|. Recommend at least 100.
def operation(x, y): return x * y #Change the operater in this formula to model different functions.
depth_limit = 4 #Aborts searches that go beyond this depth. Default 100 (will never be reached). Set this to 4 if you want to do run experiments quickly
time_limit = 100 #Aborts searches that take more minutes than this to run. Default 100 (shouldn't be reached unless you run this on a slow computer)

#Generate dataset and begin trails.
D_n = generate_dataset(l, operation)
iterate_trials(D_n, L, bits, number_of_trials, depth_limit, time_limit)
