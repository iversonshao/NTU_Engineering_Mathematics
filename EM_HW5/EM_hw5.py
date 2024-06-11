import numpy as np
import matplotlib.pyplot as plt

M = 20

# Build data vectors xm
m = np.arange(1, M+1)
xm = np.column_stack((m, np.sin(m * np.pi / (4 * M))))

# Print coordinates of each data point
print("Data point coordinates:")
for x in xm:
    print(f"({x[0]}, {x[1]})")

# Calculate the mean of xm
xm_mean = np.mean(xm, axis=0)
print(f"\nMean of xm: ({xm_mean[0]}, {xm_mean[1]})")

# Center the data
centered_data = xm - xm_mean

# Compute the covariance matrix
cov_matrix = np.cov(centered_data.T)

# Compute the eigenvectors and eigenvalues of the covariance matrix
eigvals, eigvecs = np.linalg.eig(cov_matrix)

# Sort the eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, sorted_indices]

# Take the first eigenvector as the principal component
pca_vector = eigvecs[:, 0]

# Project the data onto the principal component
projected_data = xm.dot(pca_vector)

# Derive the regression line with L = 1
slope = pca_vector[1] / pca_vector[0]
intercept = xm_mean[1] - slope * xm_mean[0]
x_reg = np.linspace(xm[:, 0].min(), xm[:, 0].max(), 100)
y_reg = slope * x_reg + intercept

# Plot data points and regression line
plt.figure(figsize=(8, 6))
plt.scatter(xm[:, 0], xm[:, 1], c='b', label='Data Points')
plt.plot(x_reg, y_reg, c='r', label='Regression Line')
plt.xlabel('m')
plt.ylabel('sin(m*pi/(4*M))')
plt.title('Data Points and Regression Line')
plt.legend()

# Save the figure as a PNG file
plt.savefig('data_points_and_regression_line.png', dpi=300, bbox_inches='tight')