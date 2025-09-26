# 📈 Stock Price Prediction using SVM  

This project applies **Machine Learning (Support Vector Machine - SVM)** to predict stock price movement based on historical data.  
The model predicts whether the **next day’s closing price** will be **higher or lower** compared to today’s price, and evaluates performance using different SVM kernels (Linear, Polynomial, RBF, Sigmoid).  

---

##  Features  
- Data preprocessing using **Pandas & NumPy**  
- Feature engineering (`Open-Close`, `High-Low`)  
- Train-test split (80-20)  
- Model training with **Support Vector Machine (SVC)**  
- Accuracy comparison across multiple kernels:
  - Linear
  - Polynomial
  - RBF
  - Sigmoid  
- Strategy backtesting:
  - Calculates market return vs. strategy return
  - Plots cumulative returns for comparison  

---

##  Results  
- Evaluates **Training Accuracy** and **Testing Accuracy**  
- Shows kernel-specific accuracies  
- Plots **Market Return vs. Strategy Return** for visual comparison  

---

##  Technologies Used  
- Python 3  
- [scikit-learn](https://scikit-learn.org/) (SVC, accuracy_score, train_test_split)  
- [pandas](https://pandas.pydata.org/) (data handling)  
- [numpy](https://numpy.org/) (numerical operations)  
- [matplotlib](https://matplotlib.org/) (visualization)  

---

##  Dataset  
The model is tested on **Reliance stock data** (CSV format).  
You can replace `RELIANCE.csv` with any stock dataset of your choice, making sure it includes at least:  
- Date  
- Open  
- High  
- Low  
- Close  

---

## ▶ How to Run  

1. Clone this repository  
   ```bash
   git clone https://github.com/apoorva9-sudo/Stock_Prediction.git
   cd Stock_Prediction
