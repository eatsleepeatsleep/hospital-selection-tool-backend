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

import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)
CORS(app)

# 输入你的 Google Maps API 金钥，最好从环境变量中读取
# api_key = "AIzaSyCw2T_1-0rF4GgFvzGU6ZwjwLK2t942WW0"   # senior
api_key = "AIzaSyCtHUZ8pvBsTEDL35E23a9slI-kpRwS8c8"  # yuchi

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
        print(f"Google Maps directions from {origin} to {destination}")
        if directions and len(directions) > 0:
            route = directions[0]
            if "legs" in route and len(route["legs"]) > 0:
                leg = route["legs"][0]
                if "duration_in_traffic" in leg:
                    return leg["duration_in_traffic"]["value"]  # 记录交通时间
    except Exception as e:
        print(f"Error calculating Google Maps time: {e}")
    return None


def generate_google_map_link(hospital_name):
    encoded_hospital_name = urllib.parse.quote(hospital_name)
    google_map_url = f"https://www.google.com/maps/search/?api=1&query={encoded_hospital_name}"
    return google_map_url


def plot_truncated_normal(hospital_name, prehospital_time, lower_bound, mean, variance, threshold):
    sd = np.sqrt(variance)
    x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
    y = norm.pdf(x, mean, sd)

    # 将秒转换为分钟
    x_minutes = x / 60
    lower_bound_minutes = lower_bound / 60
    mean_minutes = mean / 60
    threshold_minutes = threshold / 60
    prehospital_time_minutes = prehospital_time / 60

    plt.figure(figsize=(12, 4), dpi=500)  # 提高画质，设置更高的 dpi
    plt.fill_between(
        x_minutes,
        0,
        y,
        where=(x_minutes < threshold_minutes - prehospital_time_minutes) & (x_minutes >= lower_bound_minutes - prehospital_time_minutes),
        color="grey",
        alpha=0.5,
        label="Truncated Area: The probability of receiving definitive treatment within the threshold",
    )
    plt.plot(x_minutes, y, color="black", label="Probability Density Function")  # 将 Density function 改为黑色的线条
    plt.axvline(threshold_minutes - prehospital_time_minutes, color="#333", linestyle="--", label="Threshold")
    plt.axvline(
        lower_bound_minutes - prehospital_time_minutes,
        color="#00008B",
        linestyle="--",
        label=f"Lower Bound: {round(lower_bound_minutes, 3):.3f} minutes",
    )
    plt.axvline(mean_minutes - prehospital_time_minutes, color="#007bff", linestyle="--", label=f"Mean Time: {round(mean_minutes, 3):.3f} minutes")
    plt.title(
        f"The continuous probability distribution of a patient receiving definitive treatment at {hospital_name}",
        fontsize=10)  # 设置标题字体
    plt.xlabel("The mean time from hospital arrival to definitive treatment (minutes)", fontsize=8)  # 设置X轴标签字体
    plt.ylabel("Probability", fontsize=8)  # 设置Y轴标签字体
    plt.legend(fontsize=8)

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")  # 保存图像时设置 bbox_inches='tight' 以获得更好的图像边界
    plt.close()  # 关闭图像
    img.seek(0)

    # 将图像转换为 base64 编码的字符串
    img_base64 = base64.b64encode(img.getvalue()).decode("utf-8")

    return img_base64


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()

    cpss_score = int(data["cpss-score"])
    onset_time = pytz.timezone("Asia/Taipei").localize(datetime.strptime(data["onset-time"], "%Y-%m-%d %H:%M:%S"))
    print(f"Received data: {data}")
    print(f"CPSS Score: {cpss_score}")
    print(f"Onset Time: {onset_time}")

    # 使用指定的时区建立当前时间
    current_time = datetime.now(tz=pytz.timezone("Asia/Taipei"))
    pretransport_time = int((current_time - onset_time).total_seconds())

    if cpss_score == 3:
        p_nLVO = 0.69
    elif cpss_score == 2:
        p_nLVO = 0.871
    elif cpss_score == 1:
        p_nLVO = 0.942
    else:
        p_nLVO = None

    origin = data["location"]  # 使用前端传来的地理位置作为 origin
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
        "Cathay General Hospital emergency room",
    ]

    print(f"Origin: {origin}")
    print(f"Calculating Google Maps times for {destinations}")
    google_maps_times = [calculate_google_maps_time(origin, dest) for dest in destinations]

    print(f"Google Maps times: {google_maps_times}")

    # 计算截尾常态分配的概率
    hospitals = []

    if google_maps_times[0] is not None:
        hospitals.append(
            {
                "name": "Mackay Memorial Hospital",
                "prehospital_time": pretransport_time + google_maps_times[0],
                "lower_bound": pretransport_time + google_maps_times[0] + 540 + p_nLVO * 1860 + (1 - p_nLVO) * (2790 + 7230 - 1530),
                "mean": pretransport_time + google_maps_times[0] + 1320 + p_nLVO * 2100 + (1 - p_nLVO) * (2790 + 7230),
                "variance": 474**2 + ((p_nLVO) ** 2) * (146**2) + ((1 - p_nLVO) ** 2) * (930**2),
            }
        )

    if google_maps_times[1] is not None:
        hospitals.append(
            {
                "name": "Wanfang Hospital",
                "prehospital_time": pretransport_time + google_maps_times[1],
                "lower_bound": pretransport_time + google_maps_times[1] + 300 + p_nLVO * 1080 + (1 - p_nLVO) * (2790 + 7770 - 1530),
                "mean": pretransport_time + google_maps_times[1] + 960 + p_nLVO * 2100 + (1 - p_nLVO) * (2790 + 7770),
                "variance": 401**2 + ((p_nLVO) ** 2) * (620**2) + ((1 - p_nLVO) ** 2) * (930**2),
            }
        )

    if google_maps_times[2] is not None:
        hospitals.append(
            {
                "name": "Tri-Service General Hospital Songshan Branch",
                "prehospital_time": pretransport_time + google_maps_times[2],
                "lower_bound": pretransport_time + google_maps_times[2] + 60 + p_nLVO * 916 + (1 - p_nLVO) * (2790 + 7710 - 1530),
                "mean": pretransport_time + google_maps_times[2] + 1500 + p_nLVO * 1200 + (1 - p_nLVO) * (2790 + 7710),
                "variance": 875**2 + ((p_nLVO) ** 2) * (173**2) + ((1 - p_nLVO) ** 2) * (930**2),
            }
        ),

    if google_maps_times[3] is not None:
        hospitals.append(
            {
                "name": "Taipei City Hospital Renai Branch",
                "prehospital_time": pretransport_time + google_maps_times[3],
                "lower_bound": pretransport_time + google_maps_times[3] + 540 + p_nLVO * 916 + (1 - p_nLVO) * (2790 + 7350 - 1530),
                "mean": pretransport_time + google_maps_times[3] + 1500 + p_nLVO * 1200 + (1 - p_nLVO) * (2790 + 7350),
                "variance": 584**2 + ((p_nLVO) ** 2) * (173**2) + ((1 - p_nLVO) ** 2) * (930**2),
            }
        )

    if google_maps_times[4] is not None:
        hospitals.append(
            {
                "name": "Tri-Service General Hospital",
                "prehospital_time": pretransport_time + google_maps_times[4],
                "lower_bound": pretransport_time + google_maps_times[4] + 60 + p_nLVO * 1140 + (1 - p_nLVO) * 4860,
                "mean": pretransport_time + google_maps_times[4] + 720 + p_nLVO * 2940 + (1 - p_nLVO) * 9720,
                "variance": 401**2 + ((p_nLVO) ** 2) * (1094**2) + ((1 - p_nLVO) ** 2) * (2954**2),
            }
        )

    if google_maps_times[5] is not None:
        hospitals.append(
            {
                "name": "National Taiwan University Hospital Department of Emergency Medicine",
                "prehospital_time": pretransport_time + google_maps_times[5],
                "lower_bound": pretransport_time + google_maps_times[5] + p_nLVO * (240 + 960) + (1 - p_nLVO) * 2100,
                "mean": pretransport_time + google_maps_times[5] + p_nLVO * (1200 + 1980) + (1 - p_nLVO) * 7110,
                "variance": ((p_nLVO) ** 2) * (584**2 + 620**2) + ((1 - p_nLVO) ** 2) * (3046**2),
            }
        )

    if google_maps_times[6] is not None:
        hospitals.append(
            {
                "name": "Taipei Veterans General Hospital",
                "prehospital_time": pretransport_time + google_maps_times[6],
                "lower_bound": pretransport_time + google_maps_times[6] + p_nLVO * (120 + 1320) + (1 - p_nLVO) * 4500,
                "mean": pretransport_time + google_maps_times[6] + p_nLVO * (1200 + 1920) + (1 - p_nLVO) * 6030,
                "variance": ((p_nLVO) ** 2) * (657**2 + 365**2) + ((1 - p_nLVO) ** 2) * (930**2),
            }
        )

    if google_maps_times[7] is not None:
        hospitals.append(
            {
                "name": "Taipei Medical University Hospital",
                "prehospital_time": pretransport_time + google_maps_times[7],
                "lower_bound": pretransport_time + google_maps_times[7] + p_nLVO * (0 + 1140) + (1 - p_nLVO) * 5940,
                "mean": pretransport_time + google_maps_times[7] + p_nLVO * (930 + 2070) + (1 - p_nLVO) * 14610,
                "variance": ((p_nLVO) ** 2) * (565**2 + 565**2) + ((1 - p_nLVO) ** 2) * (5271**2),
            }
        )

    if google_maps_times[8] is not None:
        hospitals.append(
            {
                "name": "Shin Kong Wu Ho-Su Memorial Hospital",
                "prehospital_time": pretransport_time + google_maps_times[8],
                "lower_bound": pretransport_time + google_maps_times[8] + p_nLVO * (120 + 2280) + (1 - p_nLVO) * 5820,
                "mean": pretransport_time + google_maps_times[8] + p_nLVO * (1080 + 2820) + (1 - p_nLVO) * 9960,
                "variance": ((p_nLVO) ** 2) * (584**2 + 328**2) + ((1 - p_nLVO) ** 2) * (2517**2),
            }
        )

    if google_maps_times[9] is not None:
        hospitals.append(
            {
                "name": "Cathay General Hospital emergency room",
                "prehospital_time": pretransport_time + google_maps_times[9],
                "lower_bound": pretransport_time + google_maps_times[9] + p_nLVO * (360 + 1380) + (1 - p_nLVO) * 9120,
                "mean": pretransport_time + google_maps_times[9] + p_nLVO * (1620 + 2580) + (1 - p_nLVO) * 10170,
                "variance": ((p_nLVO) ** 2) * (766**2 + 729**2) + ((1 - p_nLVO) ** 2) * (638**2),
            }
        )

    # 计算每个医院的概率
    for hospital in hospitals:
        hospital["probability"] = calculate_truncated_normal(hospital["lower_bound"], hospital["mean"], hospital["variance"], 5 * 60 * 60)
        hospital["google_map_url"] = generate_google_map_link(hospital["name"])
#        hospital["plot_base64"] = plot_truncated_normal(
#            hospital["name"], hospital["prehospital_time"], hospital["lower_bound"], hospital["mean"], hospital["variance"], 5 * 60 * 60
#        )

    # 排序医院并取前五大高概率的医院
    top_hospitals = sorted(hospitals, key=lambda x: (x["probability"], -x["mean"]), reverse=True)[:5]

    # 准备返回的数据
    response_data = {
        "top_hospitals": [
            {
                "priority": f"The {['First', 'Second', 'Third', 'Fourth', 'Fifth'][i]} priority Hospital",
                "name": hospital["name"],
                "probability": hospital["probability"],
                "mean": hospital["mean"],
                "google_map_url": hospital["google_map_url"],
#                "plot_base64": hospital["plot_base64"],
            }
            for i, hospital in enumerate(top_hospitals)
        ]
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
