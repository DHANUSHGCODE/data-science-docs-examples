# 📘 Data Science Docs + Examples (Beginner Friendly)

A complete beginner-friendly documentation project with runnable examples in **NumPy, Pandas, Scikit-learn, Matplotlib** and JavaScript equivalents.

Use this as your **README.md** for GitHub.

---

## 🔰 Overview

This project contains easy-to-understand documentation and runnable examples for:

* **NumPy** (Array operations, math, reshaping)
* **Pandas** (DataFrames, CSV, cleaning)
* **Scikit-learn** (Regression, Classification, Training/Testing)
* **Matplotlib** (Basic plots)
* **JavaScript Equivalents**: math.js & TensorFlow.js

Perfect for beginners building a portfolio.

---

# 📂 Folder Structure

```
📁 data-science-docs-examples
│
├── README.md
├── examples/
│   ├── numpy_basics.py
│   ├── pandas_basics.py
│   ├── sklearn_regression.py
│   ├── matplotlib_plot.py
│
├── js/
│   ├── mathjs_example.js
│   ├── tfjs_regression.js
│
└── datasets/
    └── sample.csv
```

---

# 📘 1. NumPy Documentation + Examples

## ✨ What is NumPy?

NumPy is a Python library used for fast numerical computing.

### Example 1: Creating Arrays

```python
import numpy as np
arr = np.array([1, 2, 3])
print(arr)
```

### Example 2: Array Math

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
print(a * b)
```

---

# 📘 2. Pandas Documentation + Examples

## ✨ What is Pandas?

Pandas is used for working with structured data.

### Example 1: Create DataFrame

```python
import pandas as pd
data = {"Name": ["A","B","C"], "Age": [20,25,30]}
df = pd.DataFrame(data)
print(df)
```

### Example 2: Read CSV

```python
df = pd.read_csv("datasets/sample.csv")
print(df.head())
```

---

# 📘 3. Scikit-learn Examples

## ✨ Linear Regression

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)
print(model.predict([[5]]))
```

## ✨ Train-Test Split

```python
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([3,6,9,12,15,18])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Train: ", X_train)
print("Test: ", X_test)
```

---

# 📘 4. Matplotlib Examples

```python
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [2,4,6,8]

plt.plot(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Simple Plot")
plt.show()
```

---

# 📘 5. JavaScript Data Examples

## math.js Example

```javascript
const math = require('mathjs');
let a = math.matrix([1,2,3]);
let b = math.matrix([4,5,6]);
console.log(math.add(a,b));
```

## TensorFlow.js Regression

```javascript
import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.tensor2d([1,2,3,4], [4,1]);
const ys = tf.tensor2d([2,4,6,8], [4,1]);

model.fit(xs, ys).then(() => {
  model.predict(tf.tensor2d([5], [1,1])).print();
});
```

---

# 📊 Sample Dataset (datasets/sample.csv)

```
Name,Age,Score
A,20,85
B,22,90
C,25,88
```

---

# 🚀 How to Run

## Run Python Examples

```
python examples/numpy_basics.py
python examples/pandas_basics.py
python examples/sklearn_regression.py
```

## Run JS Examples

```
node js/mathjs_example.js
node js/tfjs_regression.js
```

---

# ⭐ Good for Resume

This project shows:

* Documentation writing
* Python data skills
* Machine learning basics
* GitHub portfolio
* Understanding of NumPy, Pandas, Scikit-learn
* JavaScript ML skills (extra)

---

# 🎥 Recommended Beginner Data Science Course

Search on YouTube:

* **“Codebasics Data Science Full Course”** (best beginner)
* **“Freecodecamp numpy pandas full course”**
* **“Krish Naik machine learning beginners”**

---

# 🎉 Done!

You can upload this FULL document as your GitHub `README.md`.

If you want:
📌 Full folder with all `.py` and `.js` files written
📌 “About Me” GitHub profile section
📌 Badges, GitHub stats

Just tell me: **"create full code files"**.
