import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print("1D Array:")
print(arr1)
print("\n2D Array:")
print(arr2)
print("\nArray Shape:", arr2.shape)
print("Array Data Type:", arr2.dtype)
print("\nArray Operations:")
print("Sum:", np.sum(arr1))
print("Mean:", np.mean(arr1))
print("Std:", np.std(arr1))
