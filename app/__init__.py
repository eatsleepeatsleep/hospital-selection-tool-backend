from flask import Flask, request, jsonify
import googlemaps
from datetime import datetime
import numpy as np
from flask_cors import CORS
from scipy.stats import norm
import pytz
import urllib.parse
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Google Maps API 金鑰
api_key = 'AIzaSyCw2T_1-0rF4GgFvzGU6ZwjwLK2t942WW0'

# 初始化 Google Maps Client
gmaps = googlemaps.Client(key=api_key)

# 計算截尾常態分配之公式
def calculate_truncated_normal(lower_bound, mean, variance, threshold):
    sd = np.sqrt(variance)
    z_lower = (lower_bound - mean) / sd
    z_threshold = (threshold - mean) / sd
    prob1 = norm.cdf(z_lower)
    prob2 = norm.cdf(z_threshold)
    prob = (prob2 - prob1) / (1 - prob1)
    return prob

def calculate_google_maps_time(origin, destination):
    try:
        directions = gmaps.directions(origin, destination, mode="driving", departure_time=int(datetime.now().timestamp()))
        if directions and len(directions) > 0:
            route = directions[0]
            if 'legs' in route and len(route['legs']) > 0:
                leg = route['legs'][0]
                if 'duration_in_traffic' in leg:
                    return leg['duration_in_traffic']['value']  # 記錄交通時間
    except Exception as e:
        print(f"Error calculating Google Maps time: {e}")
    return None

def generate_google_map_link(hospital_name):
    encoded_hospital_name = urllib.parse.quote(hospital_name)
    google_map_url = f"https://www.google.com/maps/search/?api=1&query={encoded_hospital_name}"
    return google_map_url

# 輸出計算底線下面積之機率圖
def plot_truncated_normal(hospital_name, prehospital_time, lower_bound, mean, variance, threshold):
    sd = np.sqrt(variance)
    x = np.linspace(mean - 4*sd, mean + 4*sd, 1000)
    y = norm.pdf(x, mean, sd)
    
    # 輸出時將秒轉換成分鐘
    x_minutes = x / 60
    lower_bound_minutes = lower_bound / 60
    mean_minutes = mean / 60
    threshold_minutes = threshold / 60
    prehospital_time_minutes = prehospital_time / 60

    # 輸出圖片中的lower bound寫為prehospital time + lowerbound是為了與論文定義相同(將院內處理時間獨立出來 不與院前時間共同討論) 
    # 同理mean也相同
    
    plt.figure(figsize=(12, 4), dpi=500)  
    plt.fill_between(x_minutes, 0, y, where=(x_minutes < threshold_minutes - prehospital_time_minutes) & (x_minutes >= lower_bound_minutes - prehospital_time_minutes), color='grey', alpha=0.5, label='Truncated Area: The probability of receiving definitive treatment within the threshold')
    plt.plot(x_minutes, y, color='black', label='Probability Density Function')  
    plt.axvline(threshold_minutes, color='#333', linestyle='--', label='Threshold')  
    plt.axvline(lower_bound_minutes, color='#00008B', linestyle='--', label=f'Prehospital Time + Lower Bound: {round(lower_bound_minutes, 3):.3f} minutes')  
    plt.axvline(mean_minutes, color='#007bff', linestyle='--', label=f'Prehospital Time + Mean Time: {round(mean_minutes, 3):.3f} minutes')  
    plt.title(f'Probability Distribution of Time from Symptom Onset to Receiving Definitive Treatment at {hospital_name}', fontsize=10, family='Times New Roman')
    plt.xlabel('The expected time for a patient from symptom onset to receiving definitive treatment (minutes)', fontsize=8, family='Times New Roman')
    plt.ylabel('Probability', fontsize=8, family='Times New Roman') 
    plt.legend(fontsize=8)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()  
    img.seek(0)
    
    # 將圖像轉為 base64 編碼的字符串
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return img_base64

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()

    cpss_score = int(data['cpss-score'])
    onset_time = pytz.timezone('Asia/Taipei').localize(datetime.strptime(data['onset-time'], "%Y-%m-%d %H:%M:%S"))
    
    # 獲得當前時區時間
    current_time = datetime.now(tz=pytz.timezone('Asia/Taipei'))
    pretransport_time = int((current_time - onset_time).total_seconds())

    # 這裡cpss_score就是前端所得到mG-Fast的分數
    if cpss_score == 5:
        p_nLVO = 1
    elif cpss_score == 4:
        p_nLVO = 0.9714
    elif cpss_score == 3:
        p_nLVO = 0.7317
    elif cpss_score == 2:
        p_nLVO = 0.6111
    elif cpss_score == 1:
        p_nLVO = 0.4615
    else:
        p_nLVO = None

    origin = data['location']  # 使用前端傳來的地址作為 origin
    destinations = [
        "Mackay Memorial Hospital",
        "Wanfang Hospital",
        "Tri-Service General Hospital Songshan Branch",
        "Taipei City Hospital Renai Branch",
        "Tri-Service General Hospital",
        "National Taiwan University Hospital Department of Emergency Medicine",
        "Taipei Veterans General Hospital",
        "Taipei Medical University Hospital",
        "Shin Kong Wu Ho-Su Memorial Hospital",
        "Cathay General Hospital emergency room"
    ]
    
    google_maps_times = [calculate_google_maps_time(origin, dest) for dest in destinations]

    # 計算各醫院之機率
    # A1至A4醫院皆轉院到B2

    hospitals = [
        {
            'name': "Mackay Memorial Hospital",
            'prehospital_time':pretransport_time + google_maps_times[0],
            'lower_bound': pretransport_time + google_maps_times[0] + 0 + p_nLVO * 420 + (1 - p_nLVO) * (2790 + 360 + 1680),
            'mean': pretransport_time + google_maps_times[0] + 1020 + p_nLVO * 2340 + (1 - p_nLVO) * (2790 + 360 + 6846),
            'variance': 288**2 + ((p_nLVO)**2) * (1167**2) + ((1 - p_nLVO)**2) * (3287**2)
        },
        {
            'name': "Wanfang Hospital",
            'prehospital_time':pretransport_time + google_maps_times[1],
            'lower_bound': pretransport_time + google_maps_times[1] + 0 + p_nLVO * 1140 + (1 - p_nLVO) * (2790 + 1020 + 1680),
            'mean': pretransport_time + google_maps_times[1] + 1200 + p_nLVO * 2250 + (1 - p_nLVO) * (2790 + 1020 + 6846),
            'variance': 244**2 + ((p_nLVO)**2) * (675**2) + ((1 - p_nLVO)**2) * (3287**2)
        },
        {
            'name': "Tri-Service General Hospital Songshan Branch",
            'prehospital_time':pretransport_time + google_maps_times[2],
            'lower_bound': pretransport_time + google_maps_times[2] + 120 + p_nLVO * 1142 + (1 - p_nLVO) * (2790 + 960 + 1680),
            'mean': pretransport_time + google_maps_times[2] + 1380 + p_nLVO * 2517 + (1 - p_nLVO) * (2790 + 960 + 6846),
            'variance': 459**2 + ((p_nLVO)**2) * (836**2) + ((1 - p_nLVO)**2) * (3287**2)
        },
        {
            'name': "Taipei City Hospital Renai Branch",
            'prehospital_time':pretransport_time + google_maps_times[3],
            'lower_bound': pretransport_time + google_maps_times[3] + 120 + p_nLVO * 1142 + (1 - p_nLVO) * (2790 + 420 + 1680),
            'mean': pretransport_time + google_maps_times[3] + 1380 + p_nLVO * 2517 + (1 - p_nLVO) * (2790 + 420 + 6846),
            'variance': 282**2 + ((p_nLVO)**2) * (836**2) + ((1 - p_nLVO)**2) * (3287**2)  
        },
        {
            'name': "Tri-Service General Hospital",
            'prehospital_time':pretransport_time + google_maps_times[4],
            'lower_bound': pretransport_time + google_maps_times[4] + p_nLVO * (0 + 360) + (1 - p_nLVO) * 5400,
            'mean': pretransport_time + google_maps_times[4] + p_nLVO * (480 + 2940) + (1 - p_nLVO) * 8780,
            'variance': ((p_nLVO)**2) * ( 244**2 + 1568**2) +((1 - p_nLVO) **2) * ( 3183**2 )
        },
        {
            'name': "National Taiwan University Hospital Department of Emergency Medicine",
            'prehospital_time':pretransport_time + google_maps_times[5],
            'lower_bound': pretransport_time + google_maps_times[5] + p_nLVO * (60 + 600) + (1 - p_nLVO) * 1680,
            'mean': pretransport_time + google_maps_times[5] + p_nLVO * (900 + 2070) + (1 - p_nLVO) * 6846,
            'variance': ((p_nLVO)**2) * ( 319**2 + 894**2) +((1 - p_nLVO) **2) * ( 3287**2 ) 
        },
        {
            'name': "Taipei Veterans General Hospital",
            'prehospital_time':pretransport_time + google_maps_times[6],
            'lower_bound': pretransport_time + google_maps_times[6] + p_nLVO * (0 + 120) + (1 - p_nLVO) * 2580,
            'mean': pretransport_time + google_maps_times[6] + p_nLVO * (900 + 1980) + (1 - p_nLVO) * 6284,
            'variance': ((p_nLVO)**2) * ( 399**2 + 1131**2) +((1 - p_nLVO) **2) * ( 2318**2 ) 
        },
        {
            'name': "Taipei Medical University Hospital",
            'prehospital_time':pretransport_time + google_maps_times[7],
            'lower_bound': pretransport_time + google_maps_times[7] + p_nLVO * (0 + 900) + (1 - p_nLVO) * 5520,
            'mean': pretransport_time + google_maps_times[7] + p_nLVO * (1020 + 2160) + (1 - p_nLVO) * 12358,
            'variance': ((p_nLVO)**2) * ( 343**2 + 766**2) +((1 - p_nLVO) **2 )* ( 5290**2 ) 
        },
        {
            'name': "Shin Kong Wu Ho-Su Memorial Hospital",
            'prehospital_time':pretransport_time + google_maps_times[8],
            'lower_bound': pretransport_time + google_maps_times[8] + p_nLVO * (0 + 1140) + (1 - p_nLVO) * 5400,
            'mean': pretransport_time + google_maps_times[8] + p_nLVO * (1140 + 2520) + (1 - p_nLVO) * 8780,
            'variance': ((p_nLVO)**2) * ( 355**2 + 839**2) +((1 - p_nLVO) **2) * ( 3183**2 ) 
        },
        {
            'name': "Cathay General Hospital emergency room",
            'prehospital_time':pretransport_time + google_maps_times[9],
            'lower_bound': pretransport_time + google_maps_times[9] + p_nLVO * (180 + 1080) + (1 - p_nLVO) * 6900,
            'mean': pretransport_time + google_maps_times[9] + p_nLVO * (1500 + 2880) + (1 - p_nLVO) * 10638,
            'variance': ((p_nLVO)**2) * ( 356**2 + 1094**2) +((1 - p_nLVO) **2) * ( 3244**2 ) 
        }
    ]

    # 計算每個醫院的機率
    for hospital in hospitals:
        hospital['probability'] = calculate_truncated_normal(
            hospital['lower_bound'], hospital['mean'], hospital['variance'], 3.75 * 60 * 60
        )
        hospital['google_map_url'] = generate_google_map_link(hospital['name'])
        hospital['plot_base64'] = plot_truncated_normal(
            hospital['name'], hospital['prehospital_time'], hospital['lower_bound'], hospital['mean'], hospital['variance'], 3.75 * 60 * 60
        )

    # 排序醫院機率並取前五高
    top_hospitals = sorted(hospitals, key=lambda x: (x['probability'], -x['mean']), reverse=True)[:5]

    # 準備返回前端的數據
    response_data = {
        'top_hospitals': [
            {
                'priority': f"The {['First', 'Second', 'Third', 'Fourth', 'Fifth'][i]} priority Hospital",
                'name': hospital['name'],
                'probability': hospital['probability'],
                'mean': hospital['mean'],
                'google_map_url': hospital['google_map_url'],
                'plot_base64': hospital['plot_base64']
            } for i, hospital in enumerate(top_hospitals)
        ]
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
