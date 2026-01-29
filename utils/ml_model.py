import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3

def calculate_vulnerability_features(minerals_df):
    """
    Calculate features for vulnerability prediction
    """
    features = []
    
    for _, row in minerals_df.iterrows():
        # Feature 1: China dependency (normalized 0-1)
        china_dep = row['china_dependency_pct'] / 100
        
        # Feature 2: Strategic importance (normalized 0-1)
        strategic = row['strategic_importance'] / 10
        
        # Feature 3: Import volume (normalized)
        import_vol = row['total_import_2023_usd_millions'] / minerals_df['total_import_2023_usd_millions'].max()
        
        # Feature 4: Domestic production adequacy (inverse - lower production = higher risk)
        # Avoid division by zero
        total_need = row['total_import_2023_usd_millions'] + row['domestic_production_mt']
        if total_need > 0:
            import_dependency = row['total_import_2023_usd_millions'] / total_need
        else:
            import_dependency = 1.0
        
        # Feature 5: Alternative availability (count alternatives - inverse risk)
        alternatives = str(row['alternative_sources']).split('|')
        alt_score = 1 / (len(alternatives) + 1)  # More alternatives = lower risk
        
        # Feature 6: Concentration risk (China % * strategic importance)
        concentration = china_dep * strategic
        
        features.append([
            china_dep,
            strategic,
            import_vol,
            import_dependency,
            alt_score,
            concentration
        ])
    
    return np.array(features)

def create_training_labels(minerals_df):
    """
    Create training labels based on vulnerability criteria
    HIGH = 2, MEDIUM = 1, LOW = 0
    """
    labels = []
    
    for _, row in minerals_df.iterrows():
        china_dep = row['china_dependency_pct']
        strategic = row['strategic_importance']
        
        # Rules for vulnerability classification
        if china_dep >= 75 and strategic >= 9:
            labels.append(2)  # HIGH risk
        elif china_dep >= 60 and strategic >= 8:
            labels.append(2)  # HIGH risk
        elif china_dep >= 50 or strategic >= 8:
            labels.append(1)  # MEDIUM risk
        else:
            labels.append(0)  # LOW risk
    
    return np.array(labels)

def train_vulnerability_model():
    """
    Train Random Forest model for vulnerability prediction
    """
    print("🤖 Training ML model...")
    
    # Load data from database
    conn = sqlite3.connect('minerals.db')
    minerals_df = pd.read_sql_query("SELECT * FROM minerals", conn)
    conn.close()
    
    # Create features
    X = calculate_vulnerability_features(minerals_df)
    
    # Create labels
    y = create_training_labels(minerals_df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Calculate vulnerability scores (probability of being high risk)
    vulnerability_scores = model.predict_proba(X_scaled)[:, -1]  # Probability of HIGH risk
    
    # Update database with vulnerability scores
    conn = sqlite3.connect('minerals.db')
    cursor = conn.cursor()
    
    for i, score in enumerate(vulnerability_scores):
        mineral_id = minerals_df.iloc[i]['mineral_id']
        cursor.execute(
            "UPDATE minerals SET vulnerability_score = ? WHERE mineral_id = ?",
            (float(score), int(mineral_id))
        )
    
    conn.commit()
    conn.close()
    
    # Save model and scaler
    joblib.dump(model, 'models/vulnerability_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Feature importance
    feature_names = [
        'China Dependency',
        'Strategic Importance',
        'Import Volume',
        'Import Dependency',
        'Alternative Scarcity',
        'Concentration Risk'
    ]
    
    importances = model.feature_importances_
    
    print("✅ Model trained successfully!")
    print(f"   Model accuracy: {model.score(X_scaled, y):.2%}")
    print("\n📊 Feature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"   {name}: {importance:.3f}")
    
    print("\n🎯 Vulnerability Scores Updated in Database:")
    conn = sqlite3.connect('minerals.db')
    results = pd.read_sql_query(
        "SELECT name, china_dependency_pct, vulnerability_score FROM minerals ORDER BY vulnerability_score DESC",
        conn
    )
    conn.close()
    print(results.to_string(index=False))
    
    return model, scaler

def predict_vulnerability(mineral_data):
    """
    Predict vulnerability for a new mineral or scenario
    
    Args:
        mineral_data: dict with keys:
            - china_dependency_pct
            - strategic_importance
            - total_import_2023_usd_millions
            - domestic_production_mt
            - alternative_sources (pipe-separated string)
    """
    # Load model and scaler
    model = joblib.load('models/vulnerability_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load reference data for normalization
    conn = sqlite3.connect('minerals.db')
    minerals_df = pd.read_sql_query("SELECT * FROM minerals", conn)
    conn.close()
    
    max_import = minerals_df['total_import_2023_usd_millions'].max()
    
    # Calculate features
    china_dep = mineral_data['china_dependency_pct'] / 100
    strategic = mineral_data['strategic_importance'] / 10
    import_vol = mineral_data['total_import_2023_usd_millions'] / max_import
    
    total_need = mineral_data['total_import_2023_usd_millions'] + mineral_data.get('domestic_production_mt', 0)
    if total_need > 0:
        import_dependency = mineral_data['total_import_2023_usd_millions'] / total_need
    else:
        import_dependency = 1.0
    
    alternatives = str(mineral_data.get('alternative_sources', '')).split('|')
    alt_score = 1 / (len(alternatives) + 1)
    
    concentration = china_dep * strategic
    
    features = np.array([[
        china_dep,
        strategic,
        import_vol,
        import_dependency,
        alt_score,
        concentration
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    risk_labels = ['LOW', 'MEDIUM', 'HIGH']
    
    return {
        'risk_level': risk_labels[prediction],
        'risk_score': probabilities[2],  # Probability of HIGH risk
        'probabilities': {
            'LOW': probabilities[0],
            'MEDIUM': probabilities[1],
            'HIGH': probabilities[2]
        }
    }

if __name__ == "__main__":
    # Train the model
    train_vulnerability_model()
    
    print("\n" + "="*60)
    print("🧪 Testing prediction with custom scenario...")
    print("="*60)
    
    # Test prediction
    test_mineral = {
        'china_dependency_pct': 80,
        'strategic_importance': 9.5,
        'total_import_2023_usd_millions': 500,
        'domestic_production_mt': 10,
        'alternative_sources': 'USA|Australia'
    }
    
    result = predict_vulnerability(test_mineral)
    print(f"\n📋 Test Mineral Scenario:")
    print(f"   China Dependency: {test_mineral['china_dependency_pct']}%")
    print(f"   Strategic Importance: {test_mineral['strategic_importance']}/10")
    print(f"   Alternative Sources: {len(test_mineral['alternative_sources'].split('|'))}")
    
    print(f"\n🎯 Prediction Results:")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Risk Score: {result['risk_score']:.2%}")
    print(f"   Probabilities:")
    for level, prob in result['probabilities'].items():
        print(f"      {level}: {prob:.2%}")