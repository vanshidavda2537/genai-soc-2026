n = int(input("Enter size of square matrix: "))

matrix = []

for i in range(n):
    row = list(map(int, input().split()))
    matrix.append(row)

primary_sum = 0
secondary_sum = 0

for i in range(n):
    primary_sum += matrix[i][i]
    secondary_sum += matrix[i][n - i - 1]

print("Primary Diagonal Sum =", primary_sum)
print("Secondary Diagonal Sum =", secondary_sum)