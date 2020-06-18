"""
Модуль з тестовими функціями для оптимізації
"""

from math import sqrt, exp, cos, sin, pi

def bukin(x):
    """Функція букіна"""
    return 100*sqrt(abs(x[1]-0.01*x[0]**2)) + 0.01*abs(x[0] + 10)

def ackley(x):
    """Функція Еклі"""
    return -exp(-sqrt(0.5*sum([i**2 for i in x]))) - \
           exp(0.5*sum([cos(i) for i in x])) + 1 + exp(1)

def sphere(x):
    """Функція сфери"""
    return sum([i**2 for i in x])

def matyas(x):
    """Функція Матіаса"""
    return 0.26*sphere(x) - 0.48*x[0]*x[1]

def cross_in_tray(x):
    """Функція <Хрест на підносі>"""
    return round(-0.0001*(abs(sin(x[0])*sin(x[1])*exp(abs(100 -
                            sqrt(sum([i**2 for i in x]))/pi))) + 1)**0.1, 7)

def bohachevsky(x):
    """Функція Богачевського"""
    return x[0]**2 + 2*x[1]**2 - 0.3*cos(3*pi*x[0]) - 0.4*cos(4*pi*x[1]) + 0.7

def rosen(x):
    """Функція Розенброка"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)
