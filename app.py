import streamlit as st
import pandas as pd
import pickle

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AutoPrice Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Pastikan file model_regresi.pkl ada di direktori yang sama
        with open('model_regresi (2).pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- HEADER DASHBOARD ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🚘 AutoPrice Predictor Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.2rem;'>Sistem Rekomendasi & Prediksi Harga Kendaraan Roda Empat Berbasis Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# --- PEMBAGIAN LAYOUT KOLOM ---
col1, col2 = st.columns([1.2, 1])

# ==========================================
# KOLOM KIRI: PANEL INPUT (INTERAKTIF)
# ==========================================
with col1:
    st.subheader("⚙️ Parameter Spesifikasi Mobil")
    st.markdown("Silakan geser slider di bawah ini untuk menentukan spesifikasi mobil baru yang ingin diprediksi harganya.")
    
    # Menggunakan container agar lebih rapi
    with st.container(border=True):
        horsepower = st.slider("🐎 Tenaga Kuda (Horsepower)", min_value=50.0, max_value=500.0, value=150.0, step=5.0, help="Daya dorong mesin (makin tinggi makin bertenaga)")
        engine_size = st.slider("🛢️ Kapasitas Mesin (Engine Size - Liter)", min_value=1.0, max_value=8.0, value=2.5, step=0.1, help="Ukuran volume silinder mesin")
        power_perf = st.slider("⚡ Power Performance Factor", min_value=20.0, max_value=200.0, value=65.0, step=1.0, help="Indeks rasio performa dan daya tahan")
        fuel_eff = st.slider("🍃 Efisiensi Bahan Bakar (MPG)", min_value=10.0, max_value=50.0, value=25.0, step=1.0, help="Jarak tempuh per galon bahan bakar")
        
    st.write("")
    # Tombol menggunakan lebar penuh (use_container_width)
    hitung = st.button("🔍 Kalkulasi Estimasi Harga", type="primary", use_container_width=True)

# ==========================================
# KOLOM KANAN: PANEL HASIL & IDENTITAS
# ==========================================
with col2:
    st.subheader("📊 Hasil Prediksi")
    
    if hitung:
        if model is not None:
            # Setup input data
            input_data = pd.DataFrame({
                'Horsepower': [horsepower],
                'Engine_size': [engine_size],
                'Power_perf_factor': [power_perf],
                'Fuel_efficiency': [fuel_eff]
            })
            
            # Prediksi dan konversi ke skala aktual (Ribuan USD)
            prediksi = model.predict(input_data)
            harga_aktual = prediksi [0] * 1000
            
            st.success("Analisis spesifikasi berhasil diproses!")
            
            # Menampilkan harga dengan komponen metrik Streamlit yang modern
            st.metric(label="Estimasi Harga Retail Pasar", value=f"${harga_aktual:,.2f}", delta="Akurasi Model Teruji")
            
            # Rincian Data yang diinput dalam bentuk tabel rapi
            st.markdown("**Ringkasan Spesifikasi Input:**")
            st.table(pd.DataFrame({
                "Parameter": ["Horsepower", "Engine Size (L)", "Power Perf Factor", "Fuel Efficiency (MPG)"],
                "Nilai": [f"{horsepower}", f"{engine_size}", f"{power_perf}", f"{fuel_eff}"]
            }))
            
        else:
            st.error("Model Machine Learning belum dimuat. Pastikan 'model_regresi.pkl' tersedia.")
    else:
        st.info("👈 Sistem siap digunakan. Tentukan parameter di panel kiri, lalu tekan tombol 'Kalkulasi Estimasi Harga' untuk melihat hasil.")

    st.write("")
    st.write("")
    
    # --- FOOTER / IDENTITAS PEMBUAT ---
    # Desain kartu profil modern menggantikan kotak biru biasa
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 10px 0; color: #E5E7EB;">Informasi Pengembang</h4>
        <p style="margin: 0; font-size: 16px;"><strong>Nama :</strong> [Dimas Muhammad Surya]</p>
        <p style="margin: 0; font-size: 16px;"><strong>NPM  :</strong> [237006046]</p>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
        <p style="margin: 0; font-size: 12px; color: #D1D5DB;">Final Project - Mata Kuliah Sains Data</p>
    </div>
    """, unsafe_allow_html=True)
