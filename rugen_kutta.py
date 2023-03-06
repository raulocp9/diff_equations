from sympy import * 
from sympy.abc import x, y
from sympy.parsing.sympy_parser import parse_expr
import math
import numpy as np
from tabulate import *
import os


def rugen_kutta(eq, x, y, h, n, verbose, hide):
    if hide == False:
        print('\n\n\t\tRuge-Kutta method')
        headers = ['i', 'x', 'y', 'k1', 'k2', 'k3', 'k4']
        rows= []
        #print('y(',x,')=',y)

    for k in range(int(n+1)):
        k1 = equation(eq, x, y)
        k2 = equation(eq, x+h/2, y+h/2*k1)
        k3 = equation(eq, x+h/2, y+h/2*k2)
        k4 = equation(eq, x+h, y+h*k3)
        if hide==False:
            rows.append([x, y, k1, k2, k3, k4])
        y_old = y
        y = y + (h/6)*(k1+2*k2+2*k3+k4)
        x= x+h
        #print('y({:.8f})={:.8f}'.format(x,y))
    #print('y({:.8f})={:.8f}'.format(x,y))   
    if hide == False and verbose == True:
        print(tabulate(rows, headers=headers, showindex=True ))
    if verbose == False and hide == False:
        print('The value of x={:.4f}, y={}'.format(x-h,y_old))
    return y

def rugen_kutta_error(y0, y1):
    return 1/30 * y0 - y1

def menu():
    #x, y, z, t= symbols('x y z t', real=True)
    print("\n\nEnter a differential equation y'=f(x,t) in python notation (just right side) in terms of (x, y)")
    eq = parse_expr(input())
    initial_point_x = float(input(('Enter the initial point in x: ')))
    initial_point_y = float(input('Enter the initial point in y: '))
    h = float(input('Enter the step size: '))
    n = int(input('Enter the number of steps: '))
    print('Do you want full results (Y/n)')
    full = input().lower()
    if full == 'y' or full == 'Y':
        verbose = True
    else:
        verbose = False
    return eq, initial_point_x, initial_point_y, h, n, verbose

def equation(eq, x_i, y_i):
    #expression= eq.subs([(x, 2),(y, -1)])
    expression = lambdify((x, y), eq, 'numpy')
    value = expression(x_i, y_i)
    return value

def show(eq, initial_point_x, initial_point_y, h, n):
    print('\n\n\t\tEquation entered: ')
    init_printing()
    pprint(eq, use_unicode = True)  
    print('The step size h= ',h)
    print('x= ', initial_point_x)
    print('y(0)= ',initial_point_y)
    print('Steps= ',n)
 
def rugen_kutta_Merson(eq, x, y, h, n, verbose):
    print('\n\n \t\tRugen-Kutta Merson')
    headers = ['i', 'x', 'y', 'k1', 'k2', 'k3', 'k4']
    rows= []
    #print('y(',x,')=',y)
    for k in range(n+1):
        k1 = h * equation(eq, x, y)
        k2 = h * equation(eq, x + h/3, y + k1/3)
        k3 = h * equation(eq, x + h/3, y + k1/6 + k2/6)
        k4 = h * equation(eq, x + h/2, y + k1/8 + 3/8*k3)
        k5 = h * equation(eq, x + h, y + k1/2 -3/2*k3 + 2*k4)
        rows.append([x, y, k1, k2, k3, k4, k5])
        y_old = y
        y = y + (k1 + 4*k4 + k5)/6
        x= x+h
        #print('y({:.8f})={:.8f}'.format(x,y))
    if verbose == True:
        print(tabulate(rows, headers=headers, showindex=True))
    else:
        print('The value of x={:.4f}, y={}'.format(x-h,y_old))
    error = 1/30 * (2*k1 - 9*k3 + 8*k4 - k5)
    print('\n Error= ', error)

def rugen_kutta_Fehlberg(eq, x, y, h, n):
    #print('y(',x,')=',y)
    for k in range(n):
        k1 = h * equation(eq, x, y)
        k2 = h * equation(eq, x + h/4, y + k1/4)
        k3 = h * equation(eq, x + 3/8*h, y + 3/32*k1 + 9/32*k2)
        k4 = h * equation(eq, x + 12/13*h, y + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)
        k5 = h * equation(eq, x + h, y + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4)
        k6 = h * equation(eq, x + h/2, y - 8/21*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)
        y = y + (16/135*k1 + 6656/12825*k2 +28561/56430*k4 -9/50*k5 + 2/55*k6) 
        x= x+h
        #print('y({:.8f})={:.8f}'.format(x,y))
    error = k1/360 - 128*k3/4275 -2197*k4/75240 + k5/50 + 2*k6/55
    print('\n\nError using Rugen Kutta Fehlberg= ', error)

def main_menu():
    print('''\n\n \n \t\tThis program implement Rugen-Kutta method to solve differential equations
        
        Press Enter to continue
        Press 5 to quit
 
    ''')
    opc= input()
    if opc == '5':
        os._exit(0)
    os.system('clear')
 
if __name__ == '__main__':
    while True:
        main_menu()
        eq, initial_point_x,initial_point_y, h, n, verbose= menu()
        show(eq, initial_point_x, initial_point_y, h, n)
    
        y0 =rugen_kutta(eq, initial_point_x, initial_point_y, h, n, verbose, hide=False, )
        y1 = rugen_kutta(eq, initial_point_x, initial_point_y, h*2, n/2, verbose, hide=True)
        error = rugen_kutta_error(y0, y1)
        print('\n The local truncate error for Rugen-Kutta 4 order with h={} is: {:.10f}'.format(h, error))
        rugen_kutta_Merson(eq, initial_point_x, initial_point_y, h, n, verbose)
        rugen_kutta_Fehlberg(eq, initial_point_x, initial_point_y, h, n) 