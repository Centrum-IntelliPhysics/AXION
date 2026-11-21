import ufl

C_W = 2.0


def crack_geo(d):
    return d ** 2


def degradation(d):
    return (1.0 - d) ** 2


def brk_plus(x):
    return 0.5 * (x + abs(x))


def brk_minus(x):
    return 0.5 * (x - abs(x))


def strain(u):
    return ufl.sym(ufl.grad(u))


def psi_split(eps, lmbda, mu):
    e11, e22, e12 = eps[0, 0], eps[1, 1], eps[0, 1]
    tr_e = e11 + e22
    delta = ufl.sqrt((e11 - e22) ** 2 + 4.0 * e12 ** 2 + 1e-25)
    lam1 = 0.5 * (tr_e + delta)
    lam2 = 0.5 * (tr_e - delta)
    psi_p = (0.5 * lmbda * brk_plus(tr_e) ** 2
             + mu * (brk_plus(lam1) ** 2 + brk_plus(lam2) ** 2))
    psi_m = (0.5 * lmbda * brk_minus(tr_e) ** 2
             + mu * (brk_minus(lam1) ** 2 + brk_minus(lam2) ** 2))
    return psi_p, psi_m
