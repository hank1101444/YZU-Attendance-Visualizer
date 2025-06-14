from flask import Flask, render_template, request
import os
import pandas as pd
import folium
from io import BytesIO
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_html
import secrets
from flask import session


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SAVE_FOLDER = 'data'
os.makedirs(SAVE_FOLDER, exist_ok=True)


def analyze_attendance(csv_path, campus_coords, expansion_ratio=0.0001, k_clusters=6, html_output_path="data/op.html"):
    # 載入與清理資料
    df = pd.read_csv(csv_path)
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df = df.rename(columns={"姓名": "Name", "經度": "Latitude", "緯度": "Longitude", "簽到時間": "Check-in Time", "點名日期": "Date"})
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Check-in Time"] = pd.to_numeric(df["Check-in Time"], errors="coerce")
    df["Check-in Time"] = pd.to_datetime(df["Check-in Time"], unit="D", origin="1899-12-30")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "Date"]).reset_index(drop=True)

    # 計算地圖中心
    center_lat = sum(p[0] for p in campus_coords) / len(campus_coords)
    center_lon = sum(p[1] for p in campus_coords) / len(campus_coords)

    # 校園多邊形
    campus_polygon = Polygon(campus_coords)

    # 找出第一週簽到作為教室推估依據
    first_day = df["Date"].min()
    first_week_data = df[df["Date"] == first_day].copy()

    # 校內資料分群找出教室簽到點群
    first_week_data["InCampus"] = first_week_data.apply(
        lambda row: campus_polygon.contains(Point(row["Latitude"], row["Longitude"])), axis=1)
    campus_points = first_week_data[first_week_data["InCampus"]].copy()
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    campus_points["SubCluster"] = kmeans.fit_predict(campus_points[["Latitude", "Longitude"]])
    classroom_cluster = campus_points["SubCluster"].value_counts().idxmax()
    classroom_points = campus_points[campus_points["SubCluster"] == classroom_cluster][["Latitude", "Longitude"]].values

    # 建立並擴張教室凸包多邊形
    hull = ConvexHull(classroom_points)
    hull_points = classroom_points[hull.vertices]
    center = classroom_points.mean(axis=0)
    expanded_hull_points = []
    for point in hull_points:
        vector = point - center
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm != 0 else vector
        expanded_point = point + unit_vector * expansion_ratio
        expanded_hull_points.append(tuple(expanded_point))
    classroom_polygon = Polygon(expanded_hull_points)

    # 標記簽到位置類型
    df["IsInClassroom"] = df.apply(lambda row: classroom_polygon.contains(Point(row["Latitude"], row["Longitude"])), axis=1)
    df["LocationType"] = "Outside"
    df.loc[df["IsInClassroom"], "LocationType"] = "InClassroom"
    df.loc[
        ~df["IsInClassroom"] & df.apply(lambda row: campus_polygon.contains(Point(row["Latitude"], row["Longitude"])), axis=1),
        "LocationType"
    ] = "NearCampus"

    # 產出視覺化地圖
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17)
    folium.Polygon(locations=campus_coords, color='blue', weight=2, fill=True, fill_color='blue', fill_opacity=0.2).add_to(m)
    folium.Polygon(locations=expanded_hull_points, color='green', weight=2, fill=True, fill_color='green', fill_opacity=0.1).add_to(m)
    color_map = {"InClassroom": "green", "NearCampus": "yellow", "Outside": "red"}
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=color_map.get(row["LocationType"], "gray"),
            fill=True,
            fill_color=color_map.get(row["LocationType"], "gray"),
            fill_opacity=0.7,
            popup=f"姓名: {row['Name']}<br>簽到時間: {row['Check-in Time']}"
        ).add_to(m)
    # m.save(html_output_path)

    # 統計報表
    # 改為細分類統計
    summary = df.groupby("Name").agg(
        Total_Checkins=("Date", "count"),
        Average_Checkin_Time=("Check-in Time", "mean"),
        InClassroom_Checkins=("LocationType", lambda x: (x == "InClassroom").sum()),
        NearCampus_Checkins=("LocationType", lambda x: (x == "NearCampus").sum()),
        Outside_Checkins=("LocationType", lambda x: (x == "Outside").sum())
    ).reset_index()

    m.location = [24.970199417696264, 121.26651671619625]  # fallback: 台北
              

    csv_output_path = "data/op.csv"
    summary.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

    return summary, m, df

filter_name = None
global_chart_scripts = ""


@app.route('/', methods=['GET', 'POST'])
def upload_csv():
    global filter_name
    global global_chart_scripts
    map_html = None  # 預設地圖為 None

    # 設定校園範圍座標
    campus_coords = [
        (24.970306, 121.263250),
        (24.969944, 121.263333),
        (24.969972, 121.265611),
        (24.968972, 121.265944),
        (24.965083, 121.267139),
        (24.965083, 121.268278),
        (24.966583, 121.269583),
        (24.970972, 121.268778),
        (24.970917, 121.267250),
        (24.970583, 121.266000)
    ]
    if request.method == 'POST':
        filter_name = request.form.get("filter_name")  # ✅ 提前取得查詢姓名
        file = request.files.get('csvFile')
        print("------------------file",file)
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            session["uploaded_csv"] = file.filename

            print(f"✅ 檔案已接收：{file.filename}")
            
            # 執行分析
            summary_df, m, full_df = analyze_attendance(filepath, campus_coords)

            filter_name = request.form.get("filter_name")

            if filter_name:
                print(f"🔍 查詢指定學生：{filter_name}")
                # 只篩選該學生的定位紀錄
                df_filtered = full_df[full_df["Name"] == filter_name]

                # 若有結果則產生新地圖
                if not df_filtered.empty:
                    m = folium.Map(location=[df_filtered["Latitude"].mean(), df_filtered["Longitude"].mean()], zoom_start=17)
                    for _, row in df_filtered.iterrows():
                        folium.CircleMarker(
                            location=[row["Latitude"], row["Longitude"]],
                            radius=6,
                            color="purple",
                            fill=True,
                            fill_color="purple",
                            fill_opacity=0.7,
                            popup=f"{row['Name']}<br>{row['Check-in Time']}"
                        ).add_to(m)
                else:
                    print("⚠️ 查無資料")


            # 將地圖轉為 HTML 字串
            map_html = m._repr_html_()
            print("Summary DataFrame:")
            print(summary_df)

            # 📊 加入圖表
            fig1 = px.bar(summary_df, x="Name", y="Total_Checkins", title="每人點名次數", text="Total_Checkins")
            fig1.update_traces(textposition='outside')
            fig1.update_layout(xaxis_tickangle=-45, height=500)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='InClassroom', x=summary_df["Name"], y=summary_df["InClassroom_Checkins"]))
            fig2.add_trace(go.Bar(name='NearCampus', x=summary_df["Name"], y=summary_df["NearCampus_Checkins"]))
            fig2.add_trace(go.Bar(name='Outside', x=summary_df["Name"], y=summary_df["Outside_Checkins"]))
            fig2.update_layout(barmode='stack', title="各類型點名次數堆疊圖", xaxis_tickangle=-45, height=500)

            # === 📊 額外圖表 ===

            # 簽到時間分佈圖（Histogram）
            summary_df["Hour"] = pd.to_datetime(summary_df["Average_Checkin_Time"]).dt.hour
            fig3 = px.histogram(summary_df, x="Hour", nbins=24, title="平均簽到時間分佈（以時為單位）")
            fig3.update_layout(bargap=0.1, height=400)

            # 每天點名人數變化（折線圖）
            df_raw = pd.read_csv(filepath)
            df_raw = df_raw.iloc[2:].reset_index(drop=True)
            df_raw.columns = df_raw.iloc[0]
            df_raw = df_raw[1:].reset_index(drop=True)
            df_raw["Date"] = pd.to_datetime(df_raw["點名日期"], errors="coerce")
            df_raw = df_raw.dropna(subset=["Date"])
            daily_counts = df_raw.groupby("Date").size().reset_index(name="Checkin_Count")
            fig4 = px.line(daily_counts, x="Date", y="Checkin_Count", markers=True, title="每日點名人數變化")
            fig4.update_layout(height=400)

            # 點名類型比例（Pie Chart）
            total_counts = summary_df[["InClassroom_Checkins", "NearCampus_Checkins", "Outside_Checkins"]].sum()
            fig5 = px.pie(
                names=total_counts.index,
                values=total_counts.values,
                title="點名類型比例（全體總和）",
                hole=0.4  # donut chart
            )
            fig5.update_layout(height=400)

            chart_html_1 = to_html(fig1, include_plotlyjs='cdn', full_html=False)
            chart_html_2 = to_html(fig2, include_plotlyjs=False, full_html=False)
            chart_html_3 = to_html(fig3, include_plotlyjs=False, full_html=False)
            chart_html_4 = to_html(fig4, include_plotlyjs=False, full_html=False)
            chart_html_5 = to_html(fig5, include_plotlyjs=False, full_html=False)

            chart_scripts = (
                chart_html_1 + "<br><br>" +
                chart_html_2 + "<br><br>" +
                chart_html_3 + "<br><br>" +
                chart_html_4 + "<br><br>" +
                chart_html_5
            )

            global_chart_scripts = chart_scripts

            # chart_scripts = chart_html_1 + "<br><br>" + chart_html_2

            return render_template('index.html', map_html=map_html, chart_scripts=chart_scripts)
        
        # ✅ Case 2: 使用者查詢學生（但不重新上傳檔案）
        elif session.get("uploaded_csv"):
            filepath = os.path.join(UPLOAD_FOLDER, session["uploaded_csv"])
            filter_name = request.form.get("filter_name")
            filter_date = request.form.get("filter_date")  # ⬅️ 加入日期欄位

            # === 1. 處理「全部 / all」 ===
            if (filter_name and filter_name.lower() in ["all", "全部"]) and not filter_date:
                summary_df, m, full_df = analyze_attendance(filepath, campus_coords)
                map_html = m._repr_html_()
                return render_template("index.html", map_html=map_html, chart_scripts=global_chart_scripts)

            # === 2. 否則讀取完整資料，準備篩選 ===
            _, _, full_df = analyze_attendance(filepath, campus_coords)
            df_filtered = full_df.copy()

            if filter_name:
                df_filtered = df_filtered[df_filtered["Name"] == filter_name]
            if filter_date:
                df_filtered = df_filtered[df_filtered["Date"] == pd.to_datetime(filter_date)]

            if not df_filtered.empty:
                m = folium.Map(location=[df_filtered["Latitude"].mean(), df_filtered["Longitude"].mean()], zoom_start=17)
                for _, row in df_filtered.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=6,
                        color="purple",
                        fill=True,
                        fill_color="yellow",
                        fill_opacity=0.7,
                        popup=f"{row['Name']}<br>{row['Check-in Time']}"
                    ).add_to(m)
                m.location = [24.970199417696264, 121.26651671619625]  # fallback: 台北
              
                map_html = m._repr_html_()
            else:
                map_html = "<div style='color:red;'>查無此人或日期資料</div>"

            return render_template("index.html", map_html=map_html, chart_scripts=global_chart_scripts)

    return render_template('index.html', map_html=map_html,chart_scripts=None)


if __name__ == '__main__':
    app.run(debug=True)
