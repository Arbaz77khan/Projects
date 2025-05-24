
# 📋 StockTrendPredictor – GitHub Issue Tracker

This document contains a structured list of tasks and features for the StockTrendPredictor project. Use this to create GitHub Issues manually or track project progress.

---

## 🔧 Feature Development

- [ ] **[FEATURE] Create data pipeline to fetch OHLCV stock data from Yahoo Finance**  
  _Use `yfinance` to fetch 1 year of daily stock data and store in `/data/raw`._

- [ ] **[FEATURE] Engineer technical indicators (SMA, EMA, RSI, MACD)**  
  _Generate indicators using TA-Lib or formulas, save in `/data/processed`._

- [ ] **[FEATURE] Create target variable for trend classification**  
  _Label each day as Buy / Sell / Hold based on future price changes._

- [ ] **[FEATURE] Build baseline model for trend prediction**  
  _Train logistic regression / Random Forest classifier to predict trend._

- [ ] **[FEATURE] Predict 7-day future price using regression model**  
  _Build regression model to estimate next 7-day closing prices._

- [ ] **[FEATURE] Build Streamlit dashboard to display predictions**  
  _Simple UI to input stock, view signal, price prediction, and confidence._

---

## 🐞 Bug Tracking

- [ ] **[BUG] Missing values in technical indicator features**  
  _Handle NA rows where indicators can't be computed._

- [ ] **[BUG] Mismatch between prediction outputs and dates**  
  _Fix alignment of target labels and predicted values._

---

## 📈 Enhancements

- [ ] **[ENHANCEMENT] Add sentiment analysis from financial news**  
  _Score headlines using VADER/TextBlob and merge with stock data._

- [ ] **[ENHANCEMENT] Add portfolio simulation module**  
  _Simulate investment performance based on predicted signals._

---

## 📄 Documentation

- [ ] **[DOCS] Finalize README with visuals + project architecture**  
  _Add charts, screenshots, and summary sections._

- [ ] **[DOCS] Write model explanation report for stakeholders**  
  _Generate PDF-style report with methodology, evaluation, and next steps._

---

_Last updated: May 15, 2025
