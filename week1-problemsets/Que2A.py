rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

matrix = []

for i in range(rows):
    row = list(map(int, input().split()))
    matrix.append(row)

print("Transpose:")

for j in range(cols):
    for i in range(rows):
        print(matrix[i][j], end=" ")
    print()