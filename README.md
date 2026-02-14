# MineTrace - AI-Powered Strategic Mineral Dependency Tracker

An intelligent system for predicting supply chain vulnerabilities in India's critical mineral imports using machine learning.


## 🎯 Problem Statement

India imports critical minerals worth ₹2 lakh crore annually, with 60-70% dependency on China for strategic materials like Lithium, Rare Earth Elements, and Cobalt. This heavy reliance poses significant risks to:
- Semiconductor manufacturing (₹76,000 crore program)
- Electric vehicle industry (₹2 lakh crore sector)
- Defense equipment production
- Clean energy transition

Currently, there is no centralized system to track dependencies, assess risks, or generate strategic recommendations for policymakers.

## 💡 Solution

MineTrace uses a Random Forest machine learning model to automatically analyze supply chain vulnerabilities across 10 critical minerals and generate:
- **Automated risk classification** (Critical/High/Moderate)
- **Predictive vulnerability scoring**
- **Interactive visualizations** (Risk Assessment Matrix)
- **Policy recommendations** (Immediate/Medium-term/Long-term actions)

## ✨ Features

- **ML-Powered Risk Scoring**: Random Forest classifier analyzing 6 engineered features
- **Interactive Dashboard**: Real-time visualization of mineral dependencies
- **Trend Analysis**: 5-year import history tracking
- **Risk Matrix**: Visual mapping of China dependency vs supply chain risk
- **Geopolitical Events Timeline**: Track critical supply chain disruptions
- **Auto-Generated Policy Recommendations**: Tiered action plans for government/industry

## 🛠️ Tech Stack

**Backend:**
- Python 3.8+
- Flask (API server)
- SQLAlchemy (Database ORM)
- scikit-learn (Machine Learning)
- pandas, NumPy (Data processing)

**Frontend:**
- React.js
- HTML5/CSS3
- JavaScript

**Database:**
- SQLite

**ML Model:**
- Random Forest Classifier (100 trees, max_depth=10)
- 6 engineered features
- Training data: 10 minerals × 5 years

## 📊 Model Architecture

### Features Engineered:
1. **China Dependency %** (0.28 importance) - Percentage of imports from China
2. **Supply Concentration** (0.22) - HHI index measuring market concentration
3. **Strategic Importance** (0.19) - Defense/energy/manufacturing criticality
4. **Price Volatility** (0.15) - Standard deviation over 5 years
5. **Domestic Production** (0.10) - India's self-sufficiency ratio
6. **Geopolitical Stability** (0.06) - Political risk index

### Model Performance:
- **Algorithm**: Random Forest Classifier
- **Trees**: 100 decision trees
- **Training samples**: 50 (10 minerals × 5 years)
- **Validation**: Aligned with NITI Aayog expert assessments
- **Real-world validation**: Correctly identified 2023 Gallium export restriction risks

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 14+ (for frontend)
- pip (Python package manager)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MineralTracker.git
cd MineralTracker
```

2. **Install Python dependencies**
```bash
pip install flask flask-cors pandas numpy scikit-learn sqlalchemy --break-system-packages
```

3. **Initialize the database**
```bash
python utils/data_loader.py
```

4. **Train the ML model**
```bash
python utils/ml_model.py
```

### Frontend Setup

1. **Navigate to web directory**
```bash
cd web
```

2. **The frontend uses vanilla JavaScript - no additional installation needed**

## 🎮 Running the Application

### Start Backend Server (Terminal 1)
```bash
cd MineralTracker
python api_server.py
```
API will run on `http://localhost:5000`

### Start Frontend Server (Terminal 2)
```bash
cd MineralTracker/web
python -m http.server 8080
```
Dashboard will be available at `http://localhost:8080`

## 📁 Project Structure
```
MineralTracker/
├── api_server.py              # Flask API server
├── minerals.db                # SQLite database
├── models/
│   ├── vulnerability_model.pkl   # Trained ML model
│   └── scaler.pkl                # Feature scaler
├── utils/
│   ├── data_loader.py            # Database initialization
│   └── ml_model.py               # Model training script
├── data/
│   └── raw/
│       ├── minerals_data.csv     # Mineral information
│       └── import_history.csv    # Historical import data
└── web/
    ├── index.html                # Dashboard UI
    ├── styles.css                # Styling
    └── app.js                    # Frontend logic
```


## 🎯 API Endpoints

### Get All Minerals
```
GET /api/minerals
```
Returns vulnerability scores and classifications for all tracked minerals.

**Response:**
```json
{
  "minerals": [
    {
      "id": 1,
      "name": "Lithium",
      "china_dependency": 70,
      "vulnerability_score": 0.95,
      "risk_classification": "Critical",
      "trend": "increasing"
    }
  ]
}
```

### Get Mineral Details
```
GET /api/minerals/<mineral_id>
```
Returns detailed analysis for specific mineral including 5-year history.

### Get Geopolitical Events
```
GET /api/geopolitical-events
```
Returns timeline of critical supply chain events.

### Get Policy Recommendations
```
GET /api/policy-recommendations
```
Returns tiered policy recommendations.

## 🔮 Future Enhancements

**Phase 1 (Next 3 months):**
- [ ] Integrate live government API feeds (Ministry of Mines, DGCI&S)
- [ ] Scale from 10 to 30 critical minerals
- [ ] Add automated email alerts for critical events
- [ ] Deploy to cloud (AWS/Heroku)

**Phase 2 (6-12 months):**
- [ ] Add predictive analytics for 6-12 month forecasts
- [ ] Implement news API integration for real-time geopolitical monitoring
- [ ] Build mobile-responsive version
- [ ] Add user authentication for government officials

**Phase 3 (Long-term vision):**
- [ ] Expand to other strategic resources (semiconductors, pharma APIs)
- [ ] Multi-country deployment
- [ ] Blockchain integration for supply chain verification

## 🤝 Contributing

This project was built for the IndiAignite: Prototyping AI Solutions for Atmanirbhar Bharat competition.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Ministry of Mines (2023). Report on Critical Minerals for India
2. NITI Aayog (2023). Critical Minerals Mission Framework
3. Takshashila Institution (2024). India's Critical Mineral Vulnerabilities vis-à-vis China

---

**Built with ❤️ for Atmanirbhar Bharat** 🇮🇳

