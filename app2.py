import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Config halaman utama streamlit
st.set_page_config(page_title="Dashboard Regresi Logistik (Mean)", layout="wide")

# ==========================================
# SIDEBAR: CAPTION INFORMASI PENGEMBANG
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6840/6840410.png", width=80)
st.sidebar.markdown("### 🧑‍💻 Informasi Pengembang")
st.sidebar.markdown("**Dibuat oleh:** \nPrianto Sanema Wau\n**NIM:** \n053286867")
st.sidebar.markdown("**Program Studi:** \nStatistika")
st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini disusun untuk memenuhi tugas analisis inferensi statistik menggunakan pemodelan Regresi Logistik Biner.")

# Konten Utama
st.title("📊 Dashboard Analisis Regresi Logistik (10 Variabel Mean)")
st.markdown("Aplikasi interaktif untuk menganalisis dan memprediksi sifat tumor payudara menggunakan seluruh fitur dengan metrik rata-rata (mean).")

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    return df

try:
    df = load_data()
    # Transformasi variabel dependen biner: M = 1, B = 0
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # 10 VARIABEL DENGAN AKHIRAN 'MEAN'
    fitur = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]
    
    tab1, tab2, tab3 = st.tabs(["📁 Penjelasan Variabel & Eksplorasi", "📉 Ringkasan Model Statistik", "🔮 Kalkulator Prediksi (10 Fitur Mean)"])
    
    # ==========================================
    # TAB 1: PENJELASAN DATASET & EKSPLORASI DATA
    # ==========================================
    with tab1:
        st.header("📋 Deskripsi Dataset dan Penjelasan 10 Variabel 'Mean'")
        
        st.markdown("""
        Dataset **Breast Cancer Wisconsin (Diagnostic)** ini berisi karakteristik visual dari inti sel massa pada jaringan payudara hasil biopsi.
        
        Model ini mengevaluasi jaringan dengan pendekatan **Tendensi Sentral (Rata-rata / Mean)**. Nilai yang digunakan merupakan rata-rata dari seluruh sel yang tertangkap dan terukur dalam satu gambar sampel pasien.
        
        Berikut adalah makna dari kesepuluh variabel rata-rata tersebut:
        
        #### 📏 Kelompok Ukuran Rata-rata (Size)
        * **`radius_mean` (Rata-rata Jari-jari):** Rata-rata jarak dari pusat inti sel ke batas terluarnya untuk seluruh sel dalam sampel.
        * **`perimeter_mean` (Rata-rata Keliling):** Rata-rata panjang garis batas luar inti sel.
        * **`area_mean` (Rata-rata Luas):** Rata-rata luas dimensi inti sel. 
        *(Catatan: Sama halnya dengan metrik 'worst', ketiga ukuran ini sangat rentan menyebabkan multikolinearitas karena dihitung dari objek yang sama).*

        #### 🧩 Kelompok Bentuk dan Kontur Tepi (Shape & Contour)
        * **`smoothness_mean` (Rata-rata Kehalusan):** Mengukur variasi rata-rata lokal dalam panjang jari-jari sel. Semakin tinggi, semakin banyak sel yang pinggirannya bergerigi.
        * **`compactness_mean` (Rata-rata Kepadatan):** Seberapa padat sel-sel tersebut menempati ruang luasnya.
        * **`concavity_mean` (Rata-rata Kecekungan):** Rata-rata tingkat keparahan lekukan yang melesak ke dalam pada permukaan sel.
        * **`concave points_mean` (Rata-rata Titik Cekung):** Rata-rata jumlah lekukan tajam per sel.
        * **`symmetry_mean` (Rata-rata Kesimetrisan):** Mengukur kecenderungan umum simetris tidaknya bentuk sel-sel pasien.
        * **`fractal_dimension_mean` (Rata-rata Dimensi Fraktal):** Kompleksitas rata-rata batas pinggiran sel (menyerupai perhitungan garis pantai).

        #### 🎨 Kelompok Warna dan Tekstur (Texture)
        * **`texture_mean` (Rata-rata Tekstur):** Rata-rata standar deviasi nilai piksel *grayscale*. Semakin tinggi, rata-rata sel pada gambar tersebut semakin kasar dan bervariasi warna kromatinnya.
        """)
        
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Sampel Data Teratas")
            st.dataframe(df[['diagnosis'] + fitur].head(10), use_container_width=True)
        with col2:
            st.subheader("Distribusi Kelas Diagnosis")
            distribusi = df['diagnosis'].value_counts()
            st.bar_chart(distribusi)

    # ==========================================
    # TAB 2: RINGKASAN MODEL STATISTIK
    # ==========================================
    with tab2:
        st.header("Estimasi Parameter Menggunakan Maximum Likelihood (MLE)")
        X = sm.add_constant(df[fitur])
        y = df['target']
        model_sm = sm.Logit(y, X).fit(method='newton', disp=False)
        
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Pseudo R-squared (McFadden)", value=f"{model_sm.prsquared:.4f}")
        m2.metric(label="Log-Likelihood", value=f"{model_sm.llf:.2f}")
        m3.metric(label="LLR p-value", value=f"{model_sm.llr_pvalue:.4e}")
        
        st.subheader("Tabel Summary Regresi Logistik")
        st.text_area("Statsmodels Logit Regression Results", str(model_sm.summary()), height=400)

    # ==========================================
    # TAB 3: KALKULATOR PREDIKSI INTERAKTIF
    # ==========================================
    with tab3:
        st.header("Simulasi Prediksi Real-Time")
        st.write("Silakan geser kesepuluh parameter nilai rata-rata (mean) di bawah ini:")
        
        # SLIDER INPUT DIBUAT OTOMATIS MENJADI 5 KOLOM & 2 BARIS
        slider_vals = {}
        cols = st.columns(5)
        for i, f in enumerate(fitur):
            with cols[i % 5]:
                # Format nama agar lebih rapi (hilangkan underscore, jadikan Title Case)
                nama_label = f.replace('_', ' ').title()
                slider_vals[f] = st.slider(nama_label, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
            
        # HITUNG LOG-ODDS (Z) OTOMATIS
        params = model_sm.params
        z_score = params['const'] + sum(params[f] * slider_vals[f] for f in fitur)
                   
        probabilitas = 1 / (1 + np.exp(-z_score))
        
        st.markdown("---")
        
        col_hasil, col_grafik = st.columns([1, 2])
        
        with col_hasil:
            st.subheader("Hasil Diagnostik")
            st.metric(label="Nilai Log-Odds (Z)", value=f"{z_score:.4f}")
            st.metric(label="Probabilitas Ganas (P)", value=f"{probabilitas * 100:.2f}%")
            
            st.write("---")
            if probabilitas >= 0.5:
                st.error("🚨 **PREDIKSI: MALIGNANT (GANAS)**")
            else:
                st.success("✅ **PREDIKSI: BENIGN (JINAK)**")
                
        with col_grafik:
            # GRAFIK KURVA SIGMOID INTERAKTIF
            z_range = np.linspace(-25, 25, 200)
            p_range = 1 / (1 + np.exp(-z_range))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_range, y=p_range, mode='lines', name='Kurva Sigmoid', line=dict(color='blue', width=3)))
            
            titik_warna = 'red' if probabilitas >= 0.5 else 'green'
            fig.add_trace(go.Scatter(x=[z_score], y=[probabilitas], mode='markers', name='Titik Pasien',
                                     marker=dict(color=titik_warna, size=15, line=dict(color='black', width=2))))
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Batas Keputusan (0.5)")
            
            fig.update_layout(
                title="Visualisasi Posisi Probabilitas pada Fungsi Sigmoid",
                xaxis_title="Log-Odds (Z)",
                yaxis_title="Probabilitas (P)",
                yaxis=dict(range=[-0.05, 1.05]),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("Gagal memuat data. Pastikan file 'data.csv' sudah diunggah di folder kerja yang sama dengan app.py.")
