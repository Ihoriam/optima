"""
Модуль реалізує методи безумовної багатовимірної 
мінімізації.
"""
import numpy as np
from numpy.linalg import norm, inv, det


# методи нульового порядку
# метод координатного спуску
def coordinate_descent(f, x0, eps):
    """Метод покоординатного спуску

    Args:
        f (function): функція
        x0 (int, float): початкова координата х
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    # метод дихотомії 
    def dichotomy(f, x, i, eps):
        delta = eps/10
        x_left = x.copy()
        x_right = x.copy()
        a = -10.
        b = 10.
        while np.abs(b-a)>eps:
            x_left[i] = (a + b - delta)/2.
            x_right[i] = (a + b +delta)/2.
            if f(x_left) < f(x_right):
                b = x_right[i]
            else:
                a = x_left[i]
        return (a + b)/2.
    
    n = len(x0)
    x1 = np.zeros(n, dtype = np.float)
    for i in range(0, n):
        x1[i] = dichotomy(f, x0, i, eps)
    k = 1
    while norm(x1 - x0, 1) > eps and k < 5000:
        x0 = x1.copy()
        for i in range(0, n):
            x1[i]=dichotomy(f, x0, i, eps)
        k= k + 1
    return [x1, f(x1), k]

# метод Хука Дживса
def hooke_jeeves(f, x0, eps):
    """Метод Хука Дживса

    Args:
        f (function): функція
        x0 (int, float): початкова координата х
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    n = len(x0)
    x1 = np.zeros(n, dtype = np.float)
    delta = 0.5
    alpha = 2.
    lmbda = 1. 
    k = 1
    while delta > eps and k < 2000:      
        y1 = np.zeros(n, dtype = np.float)
        # крок 2. 
        for i in range(0, n):
            y0 = x0.copy()
            y0[i] = y0[i] + delta
            if f(y0) < f(x0):
                y1[i] = y0[i]
            else:
                y0[i] = y0[i] - 2. * delta
                if f(y0) < f(x0):
                    y1[i] = y0[i]
                else:
                    y1[i] = x0[i]             
        # крок 3. 
        if f(y1) < f(x0):
            # крок 4. 
            x1 = y1.copy()
            y1 = x1 + lmbda*(x1-x0)
            x0 = x1.copy()
        else:
            # крок 5.
            delta = delta/alpha
        x0 = y1.copy()
        x1 = y1.copy()
        k += 1
    
    return [x1, f(x1), k]

# метод Нелдера Міда
def nelder_mead(f, x1, x2, x3, eps):
    """Метод Нелдера Міда

    Args:
        f (function): функція
        x1 (int, float): початкова координата однієї з трьох точок
        x2 (int, float): початкова координата однієї з трьох точок
        x3 (int, float): початкова координата однієї з трьох точок
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    lst = sorted([[f(x1), x1], [f(x2), x2], [f(x3), x3]])
    xl = np.array(lst[0][1])
    xs = np.array(lst[1][1])
    xh = np.array(lst[2][1])
    x4 = (xl + xs) / 2
    sigma = np.sqrt(1./3 * ((f(x1) - f(x4))**2 + (f(x2) - f(x4))**2 + (f(x3) - f(x4))**2))
    k = 0
    while (sigma > eps) & (k <= 250):
        flag = True
        x5 = x4 + alpha * (x4 - xh)
        # крок 6. Перевірити виконання умов
        if f(x5) <= f(xl): # умова а
            x6 = x4 + gamma * (x5 - x4)
            if f(x6) < f(xl):
                xh = x6
            else:
                xh = x5
        elif f(xs) < f(x5) and f(x5) <= f(xh): # умова б
            x7 = x4 + beta * (xh - x4)
            xh = x7
        elif f(xl) < f(x5) and f(x5) <= f(xs): # умова и
            xh = x5
        else: # умова г
            x1 = xl + 0.5 * (x1 - xl)
            x2 = xl + 0.5 * (x2 - xl)
            x3 = xl + 0.5 * (x3 - xl)
            flag = False
        if flag == True:
            x1 = xl
            x2 = xs
            x3 = xh
        lst = sorted([[f(x1), x1], [f(x2), x2], [f(x3), x3]])
        xl = np.array(lst[0][1])
        xs = np.array(lst[1][1])
        xh = np.array(lst[2][1])
        x4 = (xl + xs) / 2
        sigma = np.sqrt(1./3 * ((f(x1) - f(x4))**2 + (f(x2) - f(x4))**2 + (f(x3) - f(x4))**2))
        k += 1
    return [xl, f(xl), k]

# методи першого порядку
# метод градієнтного спуску
def gradient_descent(f, grad, x0, eps):
    """Метод Градієнтноо спуску

    Args:
        f (function): функція
        grad (function): градієнт функції
        x0 (int, float): початкова координата x
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    def dichotomy(f, grad, x0, eps):
        delta = eps/10.
        a = -2.0
        b =  2.5
        while np.abs(b-a) > eps:
            alpha1 = (a + b - delta)/2.
            alpha2 = (a + b + delta)/2.
            f1 = f(x0 - alpha1*grad(x0))
            f2 = f(x0 - alpha2*grad(x0))
            if f1 < f2:
                b = alpha2
            else:
                a = alpha1
        return (a + b)/2.
    
    alpha = dichotomy(f, grad, x0, eps)
    x1 = x0 - alpha * grad(x0)
    k = 1
    while norm((x1-x0), 1) > eps and k < 5000:
        x0 = x1
        alpha = dichotomy(f, grad, x0, eps)
        x1 = x0 - alpha * grad(x0)
        k = k + 1
    return [x1, f(x1), k]

# метод спряжених градієнтів
def conjugate_gradients(f, grad, x0, eps):
    """Метод Градієнтного спуску

    Args:
        f (function): функція
        grad (function): градієнт функції
        x0 (int, float): початкова координата x
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    def dichotomy(f, grad, x0, eps):
        delta = eps/10.
        a = -2.0
        b =  2.5
        while np.abs(b-a) > eps:
            alpha1 = (a + b - delta)/2.
            alpha2 = (a + b + delta)/2.
            f1 = f(x0 - alpha1*grad(x0))
            f2 = f(x0 - alpha2*grad(x0))
            if f1 < f2:
                b = alpha2
            else:
                a = alpha1
        return (a + b)/2.
    
    p = -grad(x0)
    alpha = dichotomy(f, grad, x0, eps)
    x1 = x0 + alpha*p
    k = 1
    while norm(grad(x1), 1) > eps and k < 500:
        b = norm(grad(x1), 1)**2 / norm(grad(x0), 1)**2
        p = -grad(x1) + b * p
        x0 = x1
        alpha = dichotomy(f, grad, x0, eps)
        x1 = x0 + alpha*p
        k = k + 1
    return [x1, f(x1), k]

# методи другого порядку
# метод Нютона
def newton(f, grad, hesse,  x0, eps):
    """Метод Нютона

    Args:
        f (function): функція
        grad (function): градієнт функції
        hesse (function): Гессіан функції
        x0 (int, float): початкова координата x
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    k = 0
    gr = grad(x0)
    while norm(gr, 1) > eps and k < 50:
        hs = inv(hesse(x0))
        dt1 = hs[0][0]
        dt2 = det(hs)
        if dt1 > 0 and dt2 > 0:
            p = np.dot(-hs, gr)
        else:
            p = - gr
        x1 = x0 + p
        k = k + 1
        x0 = x1
        gr = grad(x0)

    return [x1, f(x1), k]

# метод Марквардта
def Marquardt(f, grad, hesse, x0, eps):
    """Метод Марквардта

    Args:
        f (function): функція
        grad (function): градієнт функції
        hesse (function): Гессіан функції
        x0 (int, float): початкова координата x
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    k = 0
    E = np.eye(2)
    my = 10**3
    gr = grad(x0)
    while norm(gr, 1) > eps and k < 50:
        hs = hesse(x0)
        dod = np.dot(inv(hs + np.dot(my, E)), gr)
        x1 = x0 - dod
        if f(x1) < f(x0):
            my = my/2
        else:
            my = 2*my
        x0 = x1
        k = k + 1
        gr = grad(x0)

    return [x1, f(x1), k]