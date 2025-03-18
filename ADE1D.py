#ADEs from van Genuchten and Alves 1984
import numpy as np
from scipy.special import erfc as erfc
from scipy.special import erf as erf

#case A1 - Initial Concentration==================================================================
#upper boundary: type 1, pulse
#lower boundary: semi-infinite
def ADE_1d_A1(x, t, v, R, Ci, C0, t0, al=None, D = None, silent = True):
    '''
    Advection Dispersion Equation Solution for No Production or Decay
    Case A1 - Pulse Injection, Type 1, Semi-Infinite
    From https://www.ars.usda.gov/arsuserfiles/20361500/pdf_pubs/P0753.pdf
    
    Boundary Conditions
    1.  c(x, 0) = Ci
    2a. c(0, t) = C0, when t is less than pulse time
    2b. c(0, t) = 0, when t is greater than pulse time
    3.  dc/dx(inf, t) = 0

    Arguments:
    x  = distance of evalulation
    t  = time of evalulation
    v  = velocity (L/T)
    al = dispersivity (L)
    R  = retardation factor (-)
    Ci = initial concentrtaion at x and t=0
    C0 = pulse concentration
    t0 = pulse duration (T)
    '''

    if (al is not None) & (D is None): 
        D = v*al #dispersion coeff
    elif (D is not None) & (al is None):
        D=D
    elif (D==None) & (al==None):
        print("Supply either D or al values")
    
    def Axt_func(x, t_):
        error_term0 = erfc(((R*x) - (v*t_))/(2*np.sqrt(D*R*t_)))
        error_term1 = erfc(((R*x) + (v*t_))/(2*np.sqrt(D*R*t_)))
        Axt = ((1/2)*error_term0) + (((1/2)*np.exp((v*x)/D)) * error_term1)
        return Axt

    #concentration conditions
    if t == 0: #the initial condition
        C = Ci
        if silent == False:
            print("initial condition: ", t, C)

    elif (0 < t) & (t <= t0): #where the time is less than the pulse time  
        C = Ci + (C0 - Ci) * Axt_func(x, t)
        if silent == False:
            print("pulse occuring: ", t, C)
    
    elif t > t0: #where the time is greater than the pulse -- why does it drop to zero?
        tt0 = t - t0  #difference between time and pulse
        C = Ci + (C0 - Ci)*Axt_func(x, t) - C0*Axt_func(x, tt0)  #second part should be subtracting off (superposition??) 
        
        if silent == False:
            print("after pulse:", t, C)
    
    #for simplicity #avoiding big negative numbers
    if C < 1e-8:
        C = 0.

    return C

#case A2 - Initial Concentration======================================================================================================
#upper boundary: type 3,
#lower boundary: semi-infinite
def ADE_1d_A2(x, t, v, R, Ci, C0, t0, al=None, D = None, silent = True):
    '''
    Advection Dispersion Equation Solution for No Production or Decay
    Case A2 - Type 3, Semi-Infinite
    From https://www.ars.usda.gov/arsuserfiles/20361500/pdf_pubs/P0753.pdf
    
    Boundary Conditions
    1.  c(x, 0) = Ci
    2a. -D dc/dx + vc = vC0, when t is less than pulse time
    2b. -D dc/dx + vc = 0, when t is greater than pulse time
    3.  dc/dx(inf, t) = 0

    Arguments:
    x  = distance of evalulation
    t  = time of evalulation
    v  = velocity (L/T)
    al = dispersivity (L)
    R  = retardation factor (-)
    Ci = initial concentrtaion at x and t=0
    C0 = pulse concentration
    t0 = pulse duration (T)
    '''

    if (al is not None) & (D is None): 
        D = v*al #dispersion coeff
    elif (D is not None) & (al is None):
        D=D
    elif (D==None) & (al==None):
        print("Supply either D or al values")
    
    def Axt_func(x, t_):
        #error functions
        error_term0 = erfc(((R*x) - (v*t_))/(2*np.sqrt(D*R*t_)))
        error_term1 = erfc(((R*x) + (v*t_))/(2*np.sqrt(D*R*t_)))

        #A(x,t)
        Axt = ((1/2)*error_term0) + \
        np.sqrt(((v**2)*t_)/(np.pi*D*R)) * \
        np.exp(-1*((((R*x) - (v*t_))**2)/(4*D*R*t_))) - \
        (1/2)*(1+((v*x)/D) + (((v**2)*t_)/(D*R))) * np.exp((v*x)/D) * error_term1
        
        return Axt

    #concentration conditions
    if t == 0: #the initial condition
        C = Ci
        if silent == False:
            print("initial condition: ", t, C)

    elif (0 < t) & (t <= t0): #where the time is less than the pulse time  
        C = Ci + (C0 - Ci) * Axt_func(x, t)
        if silent == False:
            print("pulse occuring: ", t, C)
    
    elif t > t0: #where the time is greater than the pulse -- why does it drop to zero?
        tt0 = t - t0  #difference between time and pulse
        C = Ci + (C0 - Ci)*Axt_func(x, t) - C0*Axt_func(x, tt0)  #second part should be subtracting off (superposition??) 
        
        if silent == False:
            print("after pulse:", t, C)
    
    #for simplicity #avoiding big negative numbers
    if C < 1e-8:
        C = 0.

    return C

#case A3 - Initial Concentration============================================================================================
#upper boundary: type 1, pulse
#lower boundary: semi-infinite
def ADE_1d_A3(x, L, t, v, R, Ci, C0, t0, al=None, D = None, silent = True):
    '''
    Advection Dispersion Equation Solution for No Production or Decay
    Case A3 - Pulse Injection, Type 1, Finite
    From https://www.ars.usda.gov/arsuserfiles/20361500/pdf_pubs/P0753.pdf
    
    Boundary Conditions
    1.  c(x, 0) = Ci
    2a. c(0, t) = C0, when t is less than pulse time
    2b. c(0, t) = 0, when t is greater than pulse time
    3.  dc/dx(L, t) = 0

    Arguments:
    x  = distance of evalulation
    L  = length of column
    t  = time of evalulation
    v  = velocity (L/T)
    al = dispersivity (L)
    R  = retardation factor (-)
    Ci = initial concentrtaion at x and t=0
    C0 = pulse concentration
    t0 = pulse duration (T)
    '''

    if (al is not None) & (D is None): 
        D = v*al #dispersion coeff
    elif (D is not None) & (al is None):
        D=D
    elif (D==None) & (al==None):
        print("Supply either D or al values")
    
    def Axt_func(x, t_):
        error_term0 = erfc(((R*x) - (v*t_))/(2*np.sqrt(D*R*t_)))
        error_term1 = erfc(((R*x) + (v*t_))/(2*np.sqrt(D*R*t_)))
        error_term2 = erfc(((R*((2*L)-x)) + (v*t_))/(2*np.sqrt(D*R*t_)))
        
        Axt = ((1/2)*error_term0) + ((1/2)*np.exp((v*x)/D)) * error_term1 + \
        (1/2)*(2 + (v*((2*L) - x)/D) + (((v**2)*t_)/(D*R)))*np.exp((v*L)/D) * \
               error_term2 - np.sqrt(((v**2)*t_)/(np.pi*D*R)) * \
               np.exp(((v*L)/D)-(R/(4*D*t_))*((2*L-x)+(v*t_)/R)**2)
        
        return Axt

    #concentration conditions
    if t == 0: #the initial condition
        C = Ci
        if silent == False:
            print("initial condition: ", t, C)

    elif (0 < t) & (t <= t0): #where the time is less than the pulse time  
        C = Ci + (C0 - Ci) * Axt_func(x, t)
        if silent == False:
            print("pulse occuring: ", t, C)
    
    elif t > t0: #where the time is greater than the pulse -- why does it drop to zero?
        tt0 = t - t0  #difference between time and pulse
        C = Ci + (C0 - Ci)*Axt_func(x, t) - C0*Axt_func(x, tt0)  #second part should be subtracting off (superposition??) 
        
        if silent == False:
            print("after pulse:", t, C)
    
    #for simplicity #avoiding big negative numbers
    if C < 1e-8:
        C = 0.

    return C

#case A10 - Initial Concentration, 
#upper boundary: type 1, decay
#lower boundary: semi-infinite
def ADE_1d_A9(x, t, v, R, Ci, Ca, Cb, lamb, al=None, D = None, silent = True):
    '''
    Advection Dispersion Equation Solution for No Production or Decay
    Case A9 - Decaying source, Type 1, semi-infinite
    From https://www.ars.usda.gov/arsuserfiles/20361500/pdf_pubs/P0753.pdf
    
    Boundary Conditions
    1.  c(x, 0) = Ci
    2.  c(0, t) = Ca + Cb * e**-lambda*t
    3.  dc/dx(L, t) = 0

    Arguments:
    x  = distance of evalulation
    t  = time of evalulation
    v  = velocity (L/T)
    al = dispersivity (L)
    R  = retardation factor (-)
    Ci = initial concentrtaion at x and t=0
    Ca = Constant A
    Cb = Constant B
    lamb = lambda (decay rate)
    '''

    if (al is not None) & (D is None): 
        D = v*al #dispersion coeff
    elif (D is not None) & (al is None):
        D=D
    elif (D==None) & (al==None):
        print("Supply either D or al values")
    
    def Axt_func(x, t_):
        error_term0 = erfc(((R*x) - (v*t_))/(2*np.sqrt(D*R*t_)))
        error_term1 = erfc(((R*x) + (v*t_))/(2*np.sqrt(D*R*t_)))
        
        Axt = ((1/2)*error_term0) + ((1/2)*np.exp((v*x)/D)) * error_term1
        
        return Axt

    def Bxt_func(x, t_):
        #note 4*lamb*D*R/v**2 must be < 1
        y = v*(np.sqrt(1-((4*lamb*D*R)/(v**2)))) #run a check to make sure term is greater than 1 in sqrt
        error_term0 = erfc(((R*x) - (y*t_))/(2*np.sqrt(D*R*t_)))
        error_term1 = erfc(((R*x) + (y*t_))/(2*np.sqrt(D*R*t_)))

        Bxt = np.exp(-lamb*t_)*((1/2)*np.exp(((v-y)*x)/(2*D))*error_term0 + \
                                    (1/2)*np.exp(((v+y)*x)/(2*D))*error_term1)
        
        return Bxt

    #concentration conditions
    if t == 0: #C(x,0)
        C = Ci
        if silent == False:
            print("initial condition: ", t, C)

    elif 0 < t: #C(x,t)
        C = Ci + (Ca - Ci) * Axt_func(x, t) + Cb*Bxt_func(x,t)
        if silent == False:
            print("decay functions: ", t, C)

    
    #for simplicity #avoiding big negative numbers
    if C < 1e-8:
        C = 0.

    return C