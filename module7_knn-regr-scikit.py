import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import NotFittedError

def main():
    N = int(input("Enter the number of points (N): "))
    if N <= 0:
        print("N should be a positive integer.")
        return

    k = int(input("Enter the value of k: "))
    if k <= 0:
        print("k should be a positive integer.")
        return
    if k > N:
        print("Error: k cannot be greater than N.")
        return

    X = np.zeros((N, 1))
    Y = np.zeros(N)

    for i in range(N):
        x = float(input(f"Enter x value for point {i + 1}: "))
        y = float(input(f"Enter y value for point {i + 1}: "))
        X[i] = x
        Y[i] = y

    variance_y = np.var(Y)
    print(f"Variance of Y: {variance_y}")

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, Y)

    X_input = float(input("Enter the X value for prediction: "))

    try:
        y_pred = model.predict(np.array([[X_input]]))
        print(f"The predicted Y value for X={X_input} is: {y_pred[0]}")
    except NotFittedError:
        print("Model error")

if __name__ == "__main__":
    main()
