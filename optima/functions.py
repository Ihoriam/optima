
def simple(x):
    return x**2 + x + np.sin(x)

def d_simple(x):
    return 2*x + np.cos(x) + 1

def d2_simple(x):
    return 2 - np.sin(x)


def not_simple(x):
    return 7*x[0]**2 + 2*x[0]*x[1] + 5*x[1]**2 + x[0] - 10*x[1]

def grad_not_simple(x):
    return np.array([ 14*x[0] + 2*x[1] + 1, 2*x[0] + 10*x[1] - 10])

def hesse_not_simple(x):
    return np.array([[14. , 2.], [2. , 10.]])