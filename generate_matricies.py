import numpy as np

np.random.seed(0)
# Generate random matrices
matrix_a = np.random.rand(1048, 1048).astype(np.float32)
matrix_b = np.random.rand(1048, 1048).astype(np.float32)
matrix_c = matrix_a @ matrix_b 

# Save matrices to files
matrix_a.tofile('matrix_a.bin')
matrix_b.tofile('matrix_b.bin')
matrix_c.tofile('matrix_c.bin')


print(f"First 5 Elements of Matrix A : {matrix_a[0, :5]}")
print(f"First 5 Elements of Matrix B : {matrix_b[0, :5]}")
print(f"First 5 Elements of Matrix C : {matrix_c[0, :5]}")