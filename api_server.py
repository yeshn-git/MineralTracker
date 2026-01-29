from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

def get_db_connection():
    conn = sqlite3.connect('minerals.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/minerals', methods=['GET'])
def get_minerals():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM minerals")
    minerals = cursor.fetchall()
    
    cursor.execute("SELECT * FROM import_history ORDER BY mineral_id, year")
    all_history = cursor.fetchall()
    
    result = []
    
    for mineral in minerals:
        mineral_dict = dict(mineral)
        mineral_id = mineral_dict['mineral_id']
        
        mineral_history = [dict(h) for h in all_history if h['mineral_id'] == mineral_id]
        yearly_data = [h['china_share_pct'] for h in mineral_history[-5:]] if mineral_history else [65, 68, 70, 72, 75]
        
        while len(yearly_data) < 5:
            yearly_data.insert(0, yearly_data[0] - 2 if yearly_data else 60)
        
        vuln_score = mineral_dict.get('vulnerability_score') or 0
        if vuln_score > 0.7:
            risk_level = 'critical'
        elif vuln_score > 0.4:
            risk_level = 'high'
        else:
            risk_level = 'moderate'
        
        if len(yearly_data) >= 2:
            if yearly_data[-1] > yearly_data[-2]:
                trend = 'increasing'
            elif yearly_data[-1] < yearly_data[-2]:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        name = mineral_dict['name']
        if name == 'Rare Earth Elements':
            symbol = 'REE'
        else:
            symbol = name[:2].upper() if len(name) >= 2 else name.upper()
        
        applications = mineral_dict.get('key_applications', '').split('|') if mineral_dict.get('key_applications') else ['General Use']
        
        react_mineral = {
            'id': mineral_id,
            'name': name,
            'symbol': symbol,
            'category': mineral_dict.get('category', 'Uncategorized'),
            'chinaImport': round(mineral_dict.get('china_dependency_pct', 0) or 0),
            'totalImport': round((mineral_dict.get('china_dependency_pct', 0) or 0) + 5),
            'domesticProduction': round(100 - (mineral_dict.get('china_dependency_pct', 0) or 0)),
            'strategicUse': applications,
            'riskLevel': risk_level,
            'trend': trend,
            'yearlyData': [round(y) for y in yearly_data],
            'reserves': {'india': 5, 'china': 50, 'world': 100},
            'priceVolatility': round((mineral_dict.get('strategic_importance', 0) or 0) * 10),
            'supplyChainRisk': round((vuln_score or 0) * 100),
            'alternatives': ['Recycling programs', 'Alternative technologies', 'Domestic exploration'],
            'domesticInitiatives': [
                f"Exploration in {name} deposits",
                "KABIL strategic initiatives",
                "Public-private partnerships"
            ]
        }
        
        result.append(react_mineral)
    
    conn.close()
    return jsonify(result)

@app.route('/api/events', methods=['GET'])
def get_events():
    events = [
        {"date": "2023-07", "event": "China restricts Gallium & Germanium exports", "impact": "critical", "minerals": ["Gallium", "Germanium"]},
        {"date": "2023-12", "event": "China tightens Rare Earth export controls", "impact": "high", "minerals": ["Rare Earth Elements"]},
        {"date": "2024-03", "event": "India-Australia Critical Minerals Partnership", "impact": "positive", "minerals": ["Lithium", "Cobalt"]},
        {"date": "2024-06", "event": "KABIL signs lithium exploration MoU with Argentina", "impact": "positive", "minerals": ["Lithium"]},
        {"date": "2024-09", "event": "China announces graphite export permits", "impact": "high", "minerals": ["Graphite"]},
        {"date": "2025-01", "event": "India discovers new deposits in Arunachal Pradesh", "impact": "positive", "minerals": ["Tungsten", "Rare Earth Elements"]}
    ]
    return jsonify(events)

@app.route('/api/policy', methods=['GET'])
def get_policy():
    policy = [
        {
            "category": "Immediate Actions",
            "items": [
                "Establish 6-month strategic mineral stockpile for critical minerals",
                "Fast-track environmental clearances for domestic mining projects",
                "Launch emergency recycling infrastructure development"
            ]
        },
        {
            "category": "Medium-term Strategy",
            "items": [
                "Diversify import sources through bilateral agreements (Australia, Chile, DRC)",
                "Invest ₹50,000 Cr in domestic exploration and mining technology",
                "Establish joint ventures with mineral-rich African nations"
            ]
        },
        {
            "category": "Long-term Vision",
            "items": [
                "Achieve 30% domestic production for critical minerals by 2035",
                "Develop alternative technologies reducing mineral dependency",
                "Create circular economy ecosystem for mineral recycling"
            ]
        }
    ]
    return jsonify(policy)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting Flask API server...")
    print(f"🌐 Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)