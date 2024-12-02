from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import googlemaps
from datetime import datetime
import numpy as np
#from flask_cors import CORS
from scipy.stats import norm
import pytz
import urllib.parse
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)
# CORS(app, origins="https://eatsleepeatsleep.github.io")
# CORS(app, resources={r"/calculate": {"origins": "https://eatsleepeatsleep.github.io"}})

#@app.route('/calculate', methods=['POST'])
#@cross_origin(origins="https://eatsleepeatsleep.github.io")

# 输入你的 Google Maps API 金钥，最好从环境变量中读取
#api_key = 'AIzaSyCtHUZ8pvBsTEDL35E23a9slI-kpRwS8c8'
api_key = os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyCtHUZ8pvBsTEDL35E23a9slI-kpRwS8c8')

# 初始化 Google Maps Client
gmaps = googlemaps.Client(key=api_key)

# 计算截尾常态分配的概率
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
                    return leg['duration_in_traffic']['value']  # 记录交通时间
    except Exception as e:
        print(f"Error calculating Google Maps time: {e}")
    return None

def generate_google_map_link(hospital_name):
    encoded_hospital_name = urllib.parse.quote(hospital_name)
    google_map_url = f"https://www.google.com/maps/search/?api=1&query={encoded_hospital_name}"
    return google_map_url

def plot_truncated_normal(hospital_name, prehospital_time, lower_bound, mean, variance, threshold):
    sd = np.sqrt(variance)
    x = np.linspace(mean - 4*sd, mean + 4*sd, 500)
    y = norm.pdf(x, mean, sd)
    
    # 将秒转换为分钟
    x_minutes = x / 60
    lower_bound_minutes = lower_bound / 60
    mean_minutes = mean / 60
    threshold_minutes = threshold / 60
    prehospital_time_minutes = prehospital_time / 60
    
    plt.figure(figsize=(12, 4), dpi=300)  # 提高画质，设置更高的 dpi
    plt.fill_between(x_minutes, 0, y, where=(x_minutes < threshold_minutes) & (x_minutes >= lower_bound_minutes), color='grey', alpha=0.5, label='Truncated Area: The probability of receiving definitive treatment within the threshold')
    plt.plot(x_minutes, y, color='black', label='Probability Density Function')  # 将 Density function 改为黑色的线条
    plt.axvline(threshold_minutes, color='#333', linestyle='--', label='Threshold')  
    plt.axvline(lower_bound_minutes, color='#00008B', linestyle='--', label=f'PrehospitalTime + Lower Bound: {round(lower_bound_minutes, 3):.3f} minutes')  
    plt.axvline(mean_minutes, color='#007bff', linestyle='--', label=f'PrehospitalTime + Mean Time: {round(mean_minutes, 3):.3f} minutes')   
    plt.xlim(0, 250)  # 限制 x 軸在 0 到 250 之間
    plt.ylim(0, max(y) * 1.1)  # y 軸設置為比最大值多 10%，以便顯示更清楚
    
    plt.title(f'The continuous probability distribution of a patient receiving definitive treatment at {hospital_name}', fontsize=10)#, family='Times New Roman')  # 设置标题字体
    plt.xlabel('The expected time for a patient from onset to receiving in hospital h', fontsize=8)#, family='Times New Roman')  # 设置X轴标签字体
    plt.ylabel('Probability', fontsize=8)#, family='Times New Roman')  # 设置Y轴标签字体
    plt.legend(fontsize=8)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', pad_inches=0)  # 保存图像时设置 bbox_inches='tight' 以获得更好的图像边界
    # plt.close()  # 关闭图像
    plt.close('all')
    img.seek(0)
    
    # 将图像转换为 base64 编码的字符串
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return img_base64

@app.route('/calculate', methods=['POST'])
@cross_origin(origins="https://eatsleepeatsleep.github.io")
def calculate():
    data = request.get_json()

    cpss_score = int(data['cpss-score'])
    onset_time = pytz.timezone('Asia/Taipei').localize(datetime.strptime(data['onset-time'], "%Y-%m-%d %H:%M:%S"))
    
    # 使用指定的时区建立当前时间
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

    origin = data['location']  # 使用前端传来的地理位置作为 origin
    destinations = [
        "馬偕紀念醫院",
        "萬芳醫院",
        "三軍總醫院松山分院",
        "臺北市立聯合醫院仁愛院區",
        "三軍總醫院",
        "國立台灣大學醫學院附設醫院急診部",
        "臺北榮民總醫院",
        "台北醫學大學附設醫院",
        "新光吳火獅紀念醫院",
        "國泰綜合醫院急診室"
    ]
    
    google_maps_times = [calculate_google_maps_time(origin, dest) for dest in destinations]

# 計算各醫院之機率
    # A1至A4醫院皆轉院到B2

    hospitals = [
        {
            'name': "馬偕紀念醫院",
            'prehospital_time':pretransport_time + google_maps_times[0],
            'lower_bound': pretransport_time + google_maps_times[0] + 0 + p_nLVO * 420 + (1 - p_nLVO) * (2790 + 360 + 1680),
            'mean': pretransport_time + google_maps_times[0] + 1020 + p_nLVO * 2340 + (1 - p_nLVO) * (2790 + 360 + 6846),
            'variance': 288**2 + ((p_nLVO)**2) * (1167**2) + ((1 - p_nLVO)**2) * (3287**2)
        },
        {
            'name': "萬芳醫院",
            'prehospital_time':pretransport_time + google_maps_times[1],
            'lower_bound': pretransport_time + google_maps_times[1] + 0 + p_nLVO * 1140 + (1 - p_nLVO) * (2790 + 1020 + 1680),
            'mean': pretransport_time + google_maps_times[1] + 1200 + p_nLVO * 2250 + (1 - p_nLVO) * (2790 + 1020 + 6846),
            'variance': 244**2 + ((p_nLVO)**2) * (675**2) + ((1 - p_nLVO)**2) * (3287**2)
        },
        {
            'name': "三軍總醫院松山分院",
            'prehospital_time':pretransport_time + google_maps_times[2],
            'lower_bound': pretransport_time + google_maps_times[2] + 120 + p_nLVO * 1142 + (1 - p_nLVO) * (2790 + 960 + 1680),
            'mean': pretransport_time + google_maps_times[2] + 1380 + p_nLVO * 2517 + (1 - p_nLVO) * (2790 + 960 + 6846),
            'variance': 459**2 + ((p_nLVO)**2) * (836**2) + ((1 - p_nLVO)**2) * (3287**2)
        },
        {
            'name': "臺北市立聯合醫院仁愛院區",
            'prehospital_time':pretransport_time + google_maps_times[3],
            'lower_bound': pretransport_time + google_maps_times[3] + 120 + p_nLVO * 1142 + (1 - p_nLVO) * (2790 + 420 + 1680),
            'mean': pretransport_time + google_maps_times[3] + 1380 + p_nLVO * 2517 + (1 - p_nLVO) * (2790 + 420 + 6846),
            'variance': 282**2 + ((p_nLVO)**2) * (836**2) + ((1 - p_nLVO)**2) * (3287**2)  
        },
        {
            'name': "三軍總醫院",
            'prehospital_time':pretransport_time + google_maps_times[4],
            'lower_bound': pretransport_time + google_maps_times[4] + p_nLVO * (0 + 360) + (1 - p_nLVO) * 5400,
            'mean': pretransport_time + google_maps_times[4] + p_nLVO * (480 + 2940) + (1 - p_nLVO) * 8780,
            'variance': ((p_nLVO)**2) * ( 244**2 + 1568**2) +((1 - p_nLVO) **2) * ( 3183**2 )
        },
        {
            'name': "國立台灣大學醫學院附設醫院急診部",
            'prehospital_time':pretransport_time + google_maps_times[5],
            'lower_bound': pretransport_time + google_maps_times[5] + p_nLVO * (60 + 600) + (1 - p_nLVO) * 1680,
            'mean': pretransport_time + google_maps_times[5] + p_nLVO * (900 + 2070) + (1 - p_nLVO) * 6846,
            'variance': ((p_nLVO)**2) * ( 319**2 + 894**2) +((1 - p_nLVO) **2) * ( 3287**2 ) 
        },
        {
            'name': "臺北榮民總醫院",
            'prehospital_time':pretransport_time + google_maps_times[6],
            'lower_bound': pretransport_time + google_maps_times[6] + p_nLVO * (0 + 120) + (1 - p_nLVO) * 2580,
            'mean': pretransport_time + google_maps_times[6] + p_nLVO * (900 + 1980) + (1 - p_nLVO) * 6284,
            'variance': ((p_nLVO)**2) * ( 399**2 + 1131**2) +((1 - p_nLVO) **2) * ( 2318**2 ) 
        },
        {
            'name': "台北醫學大學附設醫院",
            'prehospital_time':pretransport_time + google_maps_times[7],
            'lower_bound': pretransport_time + google_maps_times[7] + p_nLVO * (0 + 900) + (1 - p_nLVO) * 5520,
            'mean': pretransport_time + google_maps_times[7] + p_nLVO * (1020 + 2160) + (1 - p_nLVO) * 12358,
            'variance': ((p_nLVO)**2) * ( 343**2 + 766**2) +((1 - p_nLVO) **2 )* ( 5290**2 ) 
        },
        {
            'name': "新光吳火獅紀念醫院",
            'prehospital_time':pretransport_time + google_maps_times[8],
            'lower_bound': pretransport_time + google_maps_times[8] + p_nLVO * (0 + 1140) + (1 - p_nLVO) * 5400,
            'mean': pretransport_time + google_maps_times[8] + p_nLVO * (1140 + 2520) + (1 - p_nLVO) * 8780,
            'variance': ((p_nLVO)**2) * ( 355**2 + 839**2) +((1 - p_nLVO) **2) * ( 3183**2 ) 
        },
        {
            'name': "國泰綜合醫院急診室",
            'prehospital_time':pretransport_time + google_maps_times[9],
            'lower_bound': pretransport_time + google_maps_times[9] + p_nLVO * (180 + 1080) + (1 - p_nLVO) * 6900,
            'mean': pretransport_time + google_maps_times[9] + p_nLVO * (1500 + 2880) + (1 - p_nLVO) * 10638,
            'variance': ((p_nLVO)**2) * ( 356**2 + 1094**2) +((1 - p_nLVO) **2) * ( 3244**2 ) 
        }
    ]

    # 计算每个医院的概率
    for hospital in hospitals:
        hospital['probability'] = calculate_truncated_normal(
            hospital['lower_bound'], hospital['mean'], hospital['variance'], 3.75 * 60 * 60
        )
        hospital['google_map_url'] = generate_google_map_link(hospital['name'])
        hospital['plot_base64'] = plot_truncated_normal(
            hospital['name'], hospital['prehospital_time'], hospital['lower_bound'], hospital['mean'], hospital['variance'], 3.75 * 60 * 60
        )

    # 排序医院并取前五大高概率的医院
    top_hospitals = sorted(hospitals, key=lambda x: (x['probability'], -x['mean']), reverse=True)[:3]

    # 准备返回的数据
    response_data = {
        'top_hospitals': [
            {
                'priority': f"{['第一', '第二', '第三'][i]}推薦",
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
