def format_number(num, width):
    return ' ' * (width - len(str(num))) + str(num)


Cy = 100
Cx = 2 * Cy + 1
Mx = 100

C = [[0] * Mx for _ in range(Mx)]
for i in range(Mx):
    C[i][0] = C[i][i] = 1
for i in range(1, Mx):
    for j in range(1, Mx):
        C[i][j] = C[i - 1][j - 1] + C[i - 1][j]


def calc_a_by_formula(x, y):
    if x < -y or x > y or x % 2 != y % 2:
        return 0, 0
    n, m = (x + y) // 2 + 1, (y - x) // 2
    if m == 0:
        return 0, 1
    a1 = 0
    for k in range(1, min(n, m) + 1):
        a1 += C[n - 1][k - 1] * C[m - 1][k - 1] * (-1) ** (k + 1)
    a2 = 0
    for k in range(1, min(n - 1, m) + 1):
        a2 += C[n - 1][k] * C[m - 1][k - 1] * (-1) ** k
    return a1, a2


a1c = [[0] * Cx for _ in range(Cy)]
a2c = [[0] * Cx for _ in range(Cy)]
for y in range(Cy):
    for x in range(-Cy, Cy + 1):
        a1c[y][x + Cy], a2c[y][x + Cy] = calc_a_by_formula(x, y)

a1 = [[0] * Cx for _ in range(Cy)]
a2 = [[0] * Cx for _ in range(Cy)]
a2[0][Cy] = 1
for y in range(1, Cy):
    for x in range(-y, y + 1):
        a1[y][x + Cy] = a1[y - 1][x + 1 + Cy] + a2[y - 1][x + 1 + Cy]
        a2[y][x + Cy] = a2[y - 1][x - 1 + Cy] - a1[y - 1][x - 1 + Cy]

print(a1 == a1c)
print(a2 == a2c)
