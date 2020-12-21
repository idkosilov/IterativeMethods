import numpy as np
import matplotlib.pyplot as plt
import numba


def init_arrays(file):
    with open(file, 'r') as file:
        n = int(file.readline())
        A = np.array([[file.readline().strip() for i in range(n)] for j in range(n)], dtype=float)
        b = np.array([file.readline().strip() for i in range(n)], dtype=float)
        return A, b


@numba.njit
def relaxation_method(A, b, w):
    x, n = np.zeros(len(b)), len(b)
    k = 0
    while np.sum((np.dot(A, x) - b) ** 2) > 10 ** -16:
        for i in range(n):
            x[i] = (A[i, i] * (1 - w) * x[i] + w * (b[i] - np.dot(np.delete(A[i], i), np.delete(x, i)))) / A[i, i]
        k += 1
    return x, k, w


@numba.njit
def steepest_gradient_descent(A, b):
    x = np.zeros(len(b))
    k = 0
    while np.sum((np.dot(A, x) - b) ** 2) > 10 ** -16:
        r = b - np.dot(A, x)
        x = x + r * np.sum(r ** 2) / np.sum(r * np.dot(A, r))
        k += 1
    return x, k


@numba.njit
def rotation(A, i, j):
    while abs(A[i, j]) > 10 ** (-13):
        T = (A[i, i] - A[j, j]) / (2 * A[i, j])
        t = 1 / (T + np.sign(T) * np.sqrt(1 + T ** 2))
        c = 1 / np.sqrt(t ** 2 + 1)
        s = t * c
        R = np.identity(len(A))
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        A = A @ R
        A = R.transpose() @ A
    return A


@numba.njit
def ND(A):
    s = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                continue
            else:
                s = s + A[i, j] ** 2
    return np.sqrt(s)


@numba.njit
def rotation_algorithm(A):
    k = 0
    while ND(A) > 10 ** (-13):
        for i in range(len(A) - 1):
            for j in range(i + 1, len(A)):
                A = rotation(A, i, j)
        k += 1
    return np.diag(A), k


def test_relaxtion(A, b):
    ws = [i / 30.0 for i in range(1, 60)]
    iterations = [relaxation_method(A, b, w)[1] for w in ws]
    for i in zip(ws, iterations):
        print(f'w = {i[0]}, количество итераций: {i[1]}')
    fig, ax = plt.subplots()
    ax.plot(ws, iterations)
    ax.set_xlabel('Параметр релаксации w')
    ax.set_ylabel('Количество итераций k')
    ax.set_title('Зависимость числа итераций от параметра w')
    ax.grid(True)
    fig.savefig(f'integration.png', dpi=800)
    plt.show()


def main():
    A, b = init_arrays('test3.dat')
    res_sgd = steepest_gradient_descent(A, b)
    x_sgd, k_sgd = res_sgd
    print('*********************************************************************')
    print('\n')
    print(f'Решение уравнения Ax = b методом наискорейшего градиентного спуска:')
    print(f'x = ', *map(lambda x: round(x, 10), x_sgd))
    print(f'Число итераций k = {k_sgd}.')
    print('\n')
    print('*********************************************************************')
    print('\n')
    res_rot = rotation_algorithm(init_arrays('test3.dat')[0])
    x_rot, k_rot = res_rot
    print(f'Собственные значения матрицы A методом вращения:')
    print(f'tr A = ', *map(lambda x: round(x, 10), x_rot))
    print(f'Число итераций k = {k_rot}')
    print('\n')
    print('*********************************************************************')
    print('\n')
    res_rel = relaxation_method(A, b, 0.01)
    x_rel, k_rel, w = res_rel
    print(f'Решение уравнения Ax = b методом релаксации при w = {w}:')
    print(f'x = ', *map(lambda x: round(x, 10), x_rel))
    print(f'Число итераций k = {k_rel}')
    print('\n')
    test_relaxtion(A, b)


if __name__ == '__main__':
    main()

