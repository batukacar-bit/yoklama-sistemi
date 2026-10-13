import os
# TensorFlow ve CPU optimizasyonları
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import io
import streamlit as st
import re
import gc
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageOps
from zoneinfo import ZoneInfo
import pillow_heif

pillow_heif.register_heif_opener()

# Klasör ve Dosya Yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEKTOR_DOSYASI = os.path.join(BASE_DIR, "ogrenci_vektorleri.json")
EXCEL_DOSYASI = os.path.join(BASE_DIR, "yoklama.xlsx")
MODEL_NAME = "Facenet"

st.set_page_config(page_title="Akıllı Yoklama Portalı", layout="wide", page_icon="🎓")

# ================= ÖZEL MODERN ARAYÜZ, GÜVENLİK VE MOBİL CSS =================
st.markdown("""
<style>
/* 1. Global Güvenlik: Görsel İndirme / Sürükleme / Toolbar Engelleme */
[data-testid="stImageToolbar"],
[data-testid="stElementToolbar"],
button[title="View fullscreen"],
button[title="Download"] {
    display: none !important;
    visibility: hidden !important;
}
/* 3. Sekmedeki (PDF) önizlemenin net ve büyük görünmesini sağla */
div[data-testid="stTabContent"]:nth-of-type(3) [data-testid="stImage"] img {
    max-height: 850px !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

img {
    pointer-events: none !important;
    -webkit-user-drag: none !important;
    user-select: none !important;
    -webkit-user-select: none !important;
    -webkit-touch-callout: none !important;
}

/* 2. Kameranın Ham Önizlemesini Gizleme */
[data-testid="stCameraInput"] img {
    display: none !important;
    visibility: hidden !important;
}

/* 3. Kompakt ve Ortalı Kamera Kutusu */
[data-testid="stCameraInput"] {
    max-width: 310px !important;
    margin: 10px auto !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
}

[data-testid="stCameraInput"] video {
    max-height: 200px !important;
    object-fit: cover !important;
    border-radius: 12px !important;
}

/* 4. Şık Sekme (Tabs) Tasarımı */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(128, 128, 128, 0.06) !important;
    padding: 6px !important;
    border-radius: 12px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
}

.stTabs [aria-selected="true"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35) !important;
}

/* 5. Metrik Kartları Tasarımı */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(128, 128, 128, 0.05)) !important;
    border: 1px solid rgba(128, 128, 128, 0.15) !important;
    padding: 12px 14px !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
}

/* 6. Butonlar & Form Bileşenleri */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.1s ease, box-shadow 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
}

/* 7. Görsel Çerçeve Boyutlandırması (Kompakt ve Orantılı) */
/* 7. Görsellerin Boyutu: PDF'in ve analizin net görünmesi için sınırı genişlet */
[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 10px auto !important;
}

[data-testid="stImage"] img {
    max-height: 850px !important;
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 10px !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08) !important;
}

/* 8. Mobil Uyumluluk */
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    h1 {
        font-size: 1.45rem !important;
    }
    h2, h3 {
        font-size: 1.15rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Akıllı Öğrenci Yoklama & Kayıt Portalı")
st.caption("🔒 **Biyometrik Gizlilik Korumalı:** Fotoğraflar diske kaydedilmez; anında 128 boyutlu matematiksel vektöre dönüştürülür.")

# ================= YARDIMCI VEKTÖR & VERİTABANI İŞLEMLERİ =================
@st.cache_resource
def model_getir():
    from deepface import DeepFace
    model = DeepFace.build_model(MODEL_NAME)
    try:
        sahte_kare = np.zeros((160, 160, 3), dtype=np.uint8)
        _ = DeepFace.represent(img_path=sahte_kare, model_name=MODEL_NAME, detector_backend="skip", enforce_detection=False)
    except Exception:
        pass
    return model

def veritabanini_oku():
    if not os.path.exists(VEKTOR_DOSYASI):
        batu_yollari = [
            os.path.join(BASE_DIR, f) for f in [
                "batu_kacar.jpg", "batu_kacar.jpeg", "batu_kacar.png",
                "batu.jpg", "batu.jpeg", "batu.png"
            ]
        ]
        for by in batu_yollari:
            if os.path.exists(by):
                try:
                    from deepface import DeepFace
                    with Image.open(by) as img:
                        img_rgb = img.convert("RGB")
                        img_rgb.thumbnail((250, 250))
                        emb = DeepFace.represent(
                            img_path=np.array(img_rgb),
                            model_name=MODEL_NAME,
                            detector_backend="opencv",
                            enforce_detection=False
                        )[0]["embedding"]
                    varsayilan = [{"isim": "Batu Kaçar", "embedding": emb}]
                    veritabanini_yaz(varsayilan)
                    return varsayilan
                except Exception:
                    pass
        return []
    try:
        with open(VEKTOR_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def veritabanini_yaz(veri):
    with open(VEKTOR_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)

def ogrenci_vektor_ekle(etiket, embedding):
    mevcut = veritabanini_oku()
    mevcut = [o for o in mevcut if o["isim"] != etiket]
    mevcut.append({"isim": etiket, "embedding": embedding})
    veritabanini_yaz(mevcut)

def temiz_isim_uret(metin):
    metin = metin.strip().replace("\n", " ")
    metin = re.sub(r'[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ\s]', '', metin).strip()
    metin = re.sub(r'\s+', '_', metin)
    return metin

def metin_listesini_ayikla(ham_metin):
    if not ham_metin or not ham_metin.strip():
        return []
    metin_duzeltilmis = re.sub(r'[\r\n\t,;]+', '\n', ham_metin)
    ham_satirlar = metin_duzeltilmis.split('\n')
    temiz_isimler = []
    for s in ham_satirlar:
        s = s.strip()
        if not s:
            continue
        s_isim = re.sub(r'^[\d\.\-\)\s]+', '', s).strip()
        s_isim = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', s_isim).strip()
        s_isim = re.sub(r'\s+', ' ', s_isim)
        if len(s_isim) >= 2:
            temiz_isimler.append(s_isim)
    return temiz_isimler

def kutulari_hesapla(W, H, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran):
    x_min = int(W * (m_left / 100.0))
    x_max = int(W * (1.0 - m_right / 100.0))
    y_min = int(H * (m_top / 100.0))
    y_max = int(H * (1.0 - m_bottom / 100.0))
    toplam_w = x_max - x_min
    toplam_h = y_max - y_min
    gap_y = int(H * (satirlar_arasi / 100.0))
    gap_x = int(W * (sutunlar_arasi / 100.0))
    hucre_w = (toplam_w - (cols - 1) * gap_x) / cols
    hucre_h = (toplam_h - (rows - 1) * gap_y) / rows

    kutular = []
    for r in range(rows):
        for c in range(cols):
            hx1 = int(x_min + c * (hucre_w + gap_x))
            hy1 = int(y_min + r * (hucre_h + gap_y))
            hx2 = int(hx1 + hucre_w)
            hy2 = int(hy1 + hucre_h)
            kx1 = int(hx1 + hucre_w * 0.05)
            kx2 = int(hx2 - hucre_w * 0.05)
            ky1 = int(hy1 + hucre_h * 0.02)
            ky2 = int(hy1 + hucre_h * (foto_oran / 100.0))
            kutular.append((kx1, ky1, kx2, ky2))
    return kutular

def onizleme_ciz(img_np, kutular):
    onizleme = img_np.copy()
    for idx, (fx1, fy1, fx2, fy2) in enumerate(kutular):
        cv2.rectangle(onizleme, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
        cv2.putText(onizleme, str(idx + 1), (fx1 + 5, fy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return onizleme

def pdf_islem_ve_vektor_kaydet(pdf_bytes, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran, isim_listesi):
    from deepface import DeepFace
    _ = model_getir()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    toplam_eklenen = 0
    isimler = metin_listesini_ayikla(isim_listesi)

    for sayfa_idx, sayfa in enumerate(doc):
        pix = sayfa.get_pixmap(dpi=150)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
        if pix.n == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        H, W, _ = img_np.shape
        kutular = kutulari_hesapla(W, H, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran)

        for k in kutular:
            fx1, fy1, fx2, fy2 = k
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(W, fx2), min(H, fy2)
            foto = img_np[fy1:fy2, fx1:fx2]
            if foto.size == 0 or np.std(foto) < 14:
                continue

            if toplam_eklenen < len(isimler):
                ogrenci_adi = isimler[toplam_eklenen]
            else:
                ogrenci_adi = f"Ogrenci_{toplam_eklenen + 1}"

            foto_kucuk = cv2.resize(foto, (160, 160))
            try:
                emb = DeepFace.represent(
                    img_path=foto_kucuk,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )[0]["embedding"]
                ogrenci_vektor_ekle(ogrenci_adi, emb)
                toplam_eklenen += 1
            except Exception:
                pass

    return toplam_eklenen

def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def yoklama_kaydet(etiket):
    tr_saati = datetime.now(ZoneInfo("Europe/Istanbul"))
    zaman = tr_saati.strftime("%H:%M:%S")
    tarih = tr_saati.strftime("%Y-%m-%d")
    
    parcalar = etiket.split(" - ")
    if len(parcalar) == 2:
        numara, isim = parcalar[0], parcalar[1]
    else:
        numara, isim = "-", etiket

    if not os.path.exists(EXCEL_DOSYASI) or os.path.getsize(EXCEL_DOSYASI) == 0:
        df = pd.DataFrame([{"Öğrenci": isim, "Numara": numara, "Durum": "Var", "Giriş Saati": zaman, "Tarih": tarih}])
    else:
        try:
            df = pd.read_excel(EXCEL_DOSYASI)
            if "Öğrenci" in df.columns and isim in df["Öğrenci"].values:
                df.loc[df["Öğrenci"] == isim, "Durum"] = "Var"
                df.loc[df["Öğrenci"] == isim, "Giriş Saati"] = zaman
                df.loc[df["Öğrenci"] == isim, "Tarih"] = tarih
                if "Numara" in df.columns:
                    df.loc[df["Öğrenci"] == isim, "Numara"] = numara
            else:
                yeni = pd.DataFrame([{"Öğrenci": isim, "Numara": numara, "Durum": "Var", "Giriş Saati": zaman, "Tarih": tarih}])
                df = pd.concat([df, yeni], ignore_index=True)
        except Exception:
            df = pd.DataFrame([{"Öğrenci": isim, "Numara": numara, "Durum": "Var", "Giriş Saati": zaman, "Tarih": tarih}])
            
    df.to_excel(EXCEL_DOSYASI, index=False)
    return zaman

# ================= SEKMELER =================
sekme1, sekme2, sekme3 = st.tabs([
    "📸 Canlı Yoklama",
    "👤 Hızlı Kayıt",
    "📄 PDF Aktarım"
])

# --- 1. SEKME: KAMERA & YOKLAMA ---
with sekme1:
    col_ust1, col_ust2 = st.columns([3, 1])
    with col_ust1:
        st.subheader("📸 Canlı Sınıf Yoklaması")
    with col_ust2:
        if st.button("🔄 Yenile", use_container_width=True):
            st.rerun()

    ogrenci_db = veritabanini_oku()

    gelenler_isimleri = set()
    if os.path.exists(EXCEL_DOSYASI) and os.path.getsize(EXCEL_DOSYASI) > 0:
        try:
            df_temp = pd.read_excel(EXCEL_DOSYASI)
            if "Öğrenci" in df_temp.columns:
                gelenler_isimleri = set(df_temp["Öğrenci"].astype(str).tolist())
        except Exception:
            pass

    toplam_kayitli = len(ogrenci_db)
    toplam_gelen = len(gelenler_isimleri)
    toplam_gelmeyen = max(0, toplam_kayitli - toplam_gelen)
    oran = (toplam_gelen / toplam_kayitli * 100) if toplam_kayitli > 0 else 0.0

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("👥 Kayıtlı", f"{toplam_kayitli}")
        st.metric("❌ Gelmeyen", f"{toplam_gelmeyen}")
    with m_col2:
        st.metric("✅ Derste", f"{toplam_gelen}")
        st.metric("📊 Katılım", f"%{oran:.1f}")

    st.divider()

    esik_degeri = st.slider("Hassasiyet Eşiği", 0.25, 0.65, 0.40, 0.01)

    col_sol, col_sag = st.columns([1.1, 0.9])

    with col_sol:
        if len(ogrenci_db) == 0:
            st.warning("⚠️ Sistemde kayıtlı öğrenci vektörü yok.")
        else:
            secim = st.radio("İşlem Türü:", ["Kameradan Çek", "Fotoğraf Yükle"], horizontal=True)
            gorsel = None

            if secim == "Kameradan Çek":
                # Şık Toggle Butonu
                kamera_aktif = st.toggle("📷 Kamerayı Aç", value=False, key="yoklama_cam_toggle")
                if kamera_aktif:
                    kamera = st.camera_input("Kameraya bakın ve çekin")
                    if kamera:
                        gorsel = Image.open(kamera)
            else:
                dosya = st.file_uploader(
                    "Sınıf Fotoğrafı Seç", 
                    type=['jpg', 'jpeg', 'png', 'heic', 'HEIC', 'heif', 'HEIF', 'webp', 'WEBP'], 
                    key="yoklama_foto"
                )
                if dosya is not None:
                    try:
                        gorsel = Image.open(io.BytesIO(dosya.getvalue()))
                    except Exception as e:
                        st.error(f"Fotoğraf açılamadı: {e}")

            if gorsel is not None:
                gorsel = ImageOps.exif_transpose(gorsel)
                gorsel.thumbnail((800, 800))

                frame_rgb = np.array(gorsel)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                with st.spinner("Yüzler analiz ediliyor..."):
                    from deepface import DeepFace
                    _ = model_getir()
                    try:
                        yuz_tespitleri = DeepFace.extract_faces(
                            img_path=frame_bgr,
                            detector_backend="opencv",
                            enforce_detection=False,
                            align=False
                        )
                    except Exception as e:
                        yuz_tespitleri = []
                        st.error(f"Yüz arama hatası: {e}")

                gecerli_yuzler = [y for y in yuz_tespitleri if y.get('confidence', 0) > 0.40]

                if len(gecerli_yuzler) == 0:
                    st.error("⚠️ Fotoğrafta net bir yüz algılanamadı!")
                else:
                    kullanilan_ogrenciler = set()
                    tespit_sonuclari = []

                    for yuz_bilgisi in gecerli_yuzler:
                        fa = yuz_bilgisi['facial_area']
                        x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']

                        y1, y2 = max(0, y), min(frame_bgr.shape[0], y + h)
                        x1, x2 = max(0, x), min(frame_bgr.shape[1], x + w)
                        yuz_kare = frame_bgr[y1:y2, x1:x2]

                        if yuz_kare.size == 0:
                            continue

                        yuz_kare_kucuk = cv2.resize(yuz_kare, (160, 160))

                        try:
                            kare_emb = DeepFace.represent(
                                img_path=yuz_kare_kucuk,
                                model_name=MODEL_NAME,
                                detector_backend="skip",
                                enforce_detection=False
                            )[0]["embedding"]

                            en_kucuk = 999.0
                            en_yakin = None

                            for kisi in ogrenci_db:
                                dist = cosine_distance(kare_emb, kisi["embedding"])
                                if dist < en_kucuk:
                                    en_kucuk = dist
                                    en_yakin = kisi["isim"]

                            tespit_sonuclari.append({
                                "kutu": (x, y, w, h),
                                "en_yakin": en_yakin,
                                "mesafe": en_kucuk
                            })
                        except Exception:
                            continue

                    tespit_sonuclari.sort(key=lambda item: item["mesafe"])

                    for t in tespit_sonuclari:
                        x, y, w, h = t["kutu"]
                        isim = t["en_yakin"]
                        mesafe = t["mesafe"]

                        if mesafe <= esik_degeri and isim not in kullanilan_ogrenciler:
                            kullanilan_ogrenciler.add(isim)
                            saat = yoklama_kaydet(isim)

                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(frame_rgb, f"{isim} ({mesafe:.2f})", (x, max(25, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                            st.toast(f"✅ {isim} derse katıldı!", icon="🎓")
                            st.success(f"✅ **{isim}** tanındı! (Saat: {saat})")
                        else:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
                            cv2.putText(frame_rgb, f"Bilinmiyor ({mesafe:.2f})", (x, max(25, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

                    # Yalnızca analiz edilmiş görseli göster
                    st.image(frame_rgb, caption="Yoklama Analiz Sonucu", use_container_width=True)

                    del frame_bgr
                    del yuz_tespitleri
                    gc.collect()

    with col_sag:
        st.subheader("📋 Yoklama Paneli")
        tablo_sekme1, tablo_sekme2 = st.tabs(["✅ Katılanlar", "❌ Gelmeyenler"])

        with tablo_sekme1:
            if os.path.exists(EXCEL_DOSYASI) and os.path.getsize(EXCEL_DOSYASI) > 0:
                try:
                    df_guncel = pd.read_excel(EXCEL_DOSYASI)
                    st.dataframe(df_guncel, use_container_width=True, height=280)
                except Exception:
                    st.write("Henüz yoklama kaydı yok.")
            else:
                st.write("Henüz yoklama kaydı yok.")

        with tablo_sekme2:
            gelmeyen_satirlar = []
            for og in ogrenci_db:
                etk = og["isim"]
                parcalar = etk.split(" - ")
                no_str = parcalar[0] if len(parcalar) == 2 else "-"
                isim_str = parcalar[1] if len(parcalar) == 2 else etk

                if isim_str not in gelenler_isimleri:
                    gelmeyen_satirlar.append({"Numara": no_str, "Ad Soyad": isim_str, "Durum": "Yok"})

            if gelmeyen_satirlar:
                st.dataframe(pd.DataFrame(gelmeyen_satirlar), use_container_width=True, height=280)
            else:
                st.success("🎉 Tüm kayıtlı öğrenciler derste!")

        st.markdown("##### ✍️ Manuel Katılım Ekle")
        gelmeyen_etiketler = []
        for og in ogrenci_db:
            etk = og["isim"]
            parcalar = etk.split(" - ")
            isim_str = parcalar[1] if len(parcalar) == 2 else etk
            if isim_str not in gelenler_isimleri:
                gelmeyen_etiketler.append(etk)

        if gelmeyen_etiketler:
            col_m_sec, col_m_btn = st.columns([2, 1])
            with col_m_sec:
                secilen_kisi = st.selectbox("Seç:", gelmeyen_etiketler, label_visibility="collapsed")
            with col_m_btn:
                if st.button("➕ Ekle", use_container_width=True):
                    yoklama_kaydet(secilen_kisi)
                    st.toast(f"✅ {secilen_kisi} eklendi!", icon="🎓")
                    st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Yoklamayı Sıfırla", use_container_width=True):
                if os.path.exists(EXCEL_DOSYASI):
                    os.remove(EXCEL_DOSYASI)
                st.rerun()
        with c2:
            tr_bugun = datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%Y%m%d")
            if os.path.exists(EXCEL_DOSYASI) and os.path.getsize(EXCEL_DOSYASI) > 0:
                with open(EXCEL_DOSYASI, "rb") as f:
                    st.download_button(
                        label="📥 Excel İndir",
                        data=f,
                        file_name=f"yoklama_{tr_bugun}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

# --- 2. SEKME: HIZLI ÖĞRENCİ KAYIT (FOTOĞRAFSIZ) ---
with sekme2:
    st.subheader("👤 Yeni Öğrenci Kayıt Formu")
    st.caption("🔒 Fotoğraf ekranda gösterilmez ve diske kaydedilmez; anında vektöre çevrilir.")

    if "kayit_reset_id" not in st.session_state:
        st.session_state.kayit_reset_id = 0

    reset_id = st.session_state.kayit_reset_id

    # Örnek isim: Mehmet Yılmaz
    ogr_ad = st.text_input("Öğrenci Adı:", placeholder="Örn: Mehmet", key=f"ogr_ad_{reset_id}").strip()
    ogr_soyad = st.text_input("Öğrenci Soyadı:", placeholder="Örn: Yılmaz", key=f"ogr_soyad_{reset_id}").strip()
    ogr_no = st.text_input("Öğrenci Numarası (Opsiyonel):", placeholder="Örn: 150210001", key=f"ogr_no_{reset_id}").strip()

    kayit_yontemi = st.radio("Fotoğraf Kaynağı:", ["Kamera ile Çek", "Fotoğraf Yükle"], horizontal=True, key=f"kayit_yontem_{reset_id}")
    kayit_foto = None

    if kayit_yontemi == "Kamera ile Çek":
        # Şık Toggle Butonu
        kayit_kamera_aktif = st.toggle("📷 Kamerayı Aç", value=False, key=f"ogr_cam_toggle_{reset_id}")
        if kayit_kamera_aktif:
            cam_foto = st.camera_input("Kameraya düz bakın ve çekin", key=f"ogr_cam_{reset_id}")
            if cam_foto:
                kayit_foto = Image.open(cam_foto)
                st.info("✅ Fotoğraf başarıyla alındı (Gizlilik gereği ekranda gösterilmiyor).")
    else:
        # Tüm mobil fotoğraf formatlarını kabul edecek şekilde genişletildi
        yuklenen = st.file_uploader(
            "Vesikalık / Selfie Seç", 
            type=['jpg', 'jpeg', 'png', 'heic', 'HEIC', 'heif', 'HEIF', 'webp', 'WEBP'], 
            key=f"ogr_dosya_{reset_id}"
        )
        if yuklenen is not None:
            try:
                # Mobilde buffer çökmesini önlemek için BytesIO kullanılır
                dosya_baytlari = io.BytesIO(yuklenen.getvalue())
                kayit_foto = Image.open(dosya_baytlari)
                kayit_foto = ImageOps.exif_transpose(kayit_foto)  # Yan/ters çekilen açıyı düzelt
                st.info("✅ Fotoğraf başarıyla yüklendi (Gizlilik gereği gösterilmiyor).")
            except Exception as e:
                st.error(f"Fotoğraf okunurken hata oluştu: {e}")

    kaydet_butonu = st.button("💾 Yüz Vektörünü Sisteme Kaydet", type="primary", use_container_width=True)

    if kaydet_butonu:
        if not ogr_ad or not ogr_soyad:
            st.error("⚠️ Lütfen öğrencinin Adını ve Soyadını girin!")
        elif kayit_foto is None:
            st.error("⚠️ Lütfen bir fotoğraf çekin veya yükleyin!")
        else:
            with st.spinner("Vektör çıkarılıyor..."):
                try:
                    from deepface import DeepFace
                    _ = model_getir()
                    
                    kayit_foto.thumbnail((250, 250))
                    foto_np = np.array(kayit_foto.convert("RGB"))

                    sonuclar = DeepFace.represent(
                        img_path=foto_np,
                        model_name=MODEL_NAME,
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    emb = sonuclar[0]["embedding"]

                    if ogr_no:
                        gorunen_etiket = f"{ogr_no} - {ogr_ad} {ogr_soyad}"
                    else:
                        gorunen_etiket = f"{ogr_ad} {ogr_soyad}"

                    ogrenci_vektor_ekle(gorunen_etiket, emb)

                    del foto_np
                    del kayit_foto
                    gc.collect()

                    st.session_state.kayit_reset_id += 1
                    st.toast(f"⚡ {gorunen_etiket} sisteme kaydedildi!", icon="✅")
                    st.success(f"🎉 **{gorunen_etiket}** başarıyla kaydedildi!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Kayıt sırasında hata oluştu: {e}")

    st.markdown("---")
    
    col_liste_baslik, col_liste_buton = st.columns([2, 1])
    with col_liste_baslik:
        st.subheader("📚 Kayıtlı Öğrenci Listesi")
    with col_liste_buton:
        if st.button("🗑️ Sıfırla (Batu Kaçar kalsın)", type="secondary", use_container_width=True):
            gc.collect()
            
            yeni_veritabani = []
            
            if os.path.exists(VEKTOR_DOSYASI):
                try:
                    with open(VEKTOR_DOSYASI, "r", encoding="utf-8") as f:
                        eski_veri = json.load(f)
                    for kisi in eski_veri:
                        isim_l = str(kisi.get("isim", "")).lower().replace("ç", "c")
                        if "batu" in isim_l and "kacar" in isim_l:
                            kisi["isim"] = "Batu Kaçar"
                            yeni_veritabani.append(kisi)
                            break
                    if not yeni_veritabani:
                        for kisi in eski_veri:
                            if "batu" in str(kisi.get("isim", "")).lower():
                                kisi["isim"] = "Batu Kaçar"
                                yeni_veritabani.append(kisi)
                                break
                except Exception:
                    pass

            if not yeni_veritabani:
                batu_dosyalari = [
                    "batu_kacar.jpg", "batu_kacar.jpeg", "batu_kacar.png",
                    "batu.jpg", "batu.jpeg", "batu.png", "Batu.jpg"
                ]
                for b_isim in batu_dosyalari:
                    b_yol = os.path.join(BASE_DIR, b_isim)
                    if not os.path.exists(b_yol):
                        b_yol = os.path.join(BASE_DIR, "ogrenciler", b_isim)
                    
                    if os.path.exists(b_yol):
                        try:
                            from deepface import DeepFace
                            with Image.open(b_yol) as img:
                                img_rgb = img.convert("RGB")
                                img_rgb.thumbnail((250, 250))
                                emb = DeepFace.represent(
                                    img_path=np.array(img_rgb),
                                    model_name=MODEL_NAME,
                                    detector_backend="opencv",
                                    enforce_detection=False
                                )[0]["embedding"]
                            yeni_veritabani.append({"isim": "Batu Kaçar", "embedding": emb})
                            break
                        except Exception:
                            pass

            if os.path.exists(VEKTOR_DOSYASI):
                try:
                    os.remove(VEKTOR_DOSYASI)
                except Exception:
                    pass

            with open(VEKTOR_DOSYASI, "w", encoding="utf-8") as f:
                json.dump(yeni_veritabani, f, ensure_ascii=False, indent=2)

            st.cache_resource.clear()
            gc.collect()
            
            st.toast("Veritabanı sıfırlandı!", icon="🗑️")
            st.success("✅ Yalnızca 'Batu Kaçar' korundu.")
            st.rerun()

    mevcut_ogrenciler = veritabanini_oku()

    if mevcut_ogrenciler:
        st.info(f"Sistemde toplam **{len(mevcut_ogrenciler)}** öğrenci vektörü kayıtlı.")
        
        tablo_verisi = []
        for o in mevcut_ogrenciler:
            etiket = o["isim"]
            parcalar = etiket.split(" - ")
            if len(parcalar) == 2:
                numara, isim = parcalar[0], parcalar[1]
            else:
                numara, isim = "-", etiket
            tablo_verisi.append({"Öğrenci No": numara, "Ad Soyad": isim, "Vektör": f"{len(o['embedding'])}D"})

        df_ogrenciler = pd.DataFrame(tablo_verisi)
        st.dataframe(df_ogrenciler, use_container_width=True)
    else:
        st.write("Henüz sisteme kayıtlı öğrenci bulunmuyor.")

# --- 3. SEKME: PDF İLE TOPLU YÜKLEME (FOTOĞRAFSIZ) ---
with sekme3:
    st.subheader("Görsel Kalibrasyon & Esnek İsim Eşleme (PDF)")
    st.caption("PDF'teki yüzler kırpılır, doğrudan sayı dizisine çevrilir ve fotoğraflar diske kaydedilmeden silinir.")
    yuklenen_dosya = st.file_uploader("Sınıf Listesi PDF Seçin", type=["pdf"])

    if yuklenen_dosya is not None:
        pdf_bytes = yuklenen_dosya.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        sayfa1 = doc[0]
        pix = sayfa1.get_pixmap(dpi=150)
        ilk_sayfa_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
        if pix.n == 4:
            ilk_sayfa_np = cv2.cvtColor(ilk_sayfa_np, cv2.COLOR_RGBA2RGB)

        H, W, _ = ilk_sayfa_np.shape
        # Eski hali:
        # col_onizleme, col_ayarlar = st.columns([1.2, 1])

        # Yeni hali (Görsele 2 kat daha fazla alan verir):
        col_onizleme, col_ayarlar = st.columns([2, 1])
        with col_ayarlar:
            st.markdown("### 📝 Öğrenci İsim Listesi")
            isimler_text = st.text_area("İsimleri yapıştırın:", height=140, placeholder="Mehmet Yılmaz\nKemal Demir...")
            ayiklanan = metin_listesini_ayikla(isimler_text)
            if ayiklanan:
                st.success(f"✅ Toplam **{len(ayiklanan)}** isim algılandı.")

            cols = st.number_input("Sütun Sayısı", value=7, min_value=1, max_value=15)
            rows = st.number_input("Satır Sayısı", value=5, min_value=1, max_value=15)
            satirlar_arasi = st.slider("Satırlar Arası Boşluk Payı (%)", -25.0, 25.0, -6.20, 0.05)
            sutunlar_arasi = st.slider("Sütunlar Arası Boşluk Payı (%)", -25.0, 25.0, 0.50, 0.05)
            m_left = st.slider("Sol Boşluk", 0.0, 35.0, 12.00, 0.5)
            m_right = st.slider("Sağ Boşluk", 0.0, 35.0, 13.00, 0.5)
            m_top = st.slider("Üst Boşluk", 0.0, 40.0, 13.50, 0.5)
            m_bottom = st.slider("Alt Boşluk", 0.0, 60.0, 31.00, 0.5)
            foto_oran = st.slider("Fotoğraf Yüksekliği (%)", 20, 90, 44, 1)

            if st.button("✂️ Vektörleri Çıkar ve Kaydet", type="primary", use_container_width=True):
                with st.spinner("Yüzler taranıyor ve vektöre dönüştürülüyor..."):
                    toplam = pdf_islem_ve_vektor_kaydet(pdf_bytes, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran, isimler_text)
                    gc.collect()
                    st.toast(f"🎉 {toplam} öğrenci vektörü çıkarıldı!", icon="✅")
                    st.success(f"🎉 {toplam} öğrencinin yüz vektörü başarıyla kaydedildi! Fotoğraflar silindi.")
                    st.rerun()

        with col_onizleme:
            kutular = kutulari_hesapla(W, H, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran)
            onizleme_img = onizleme_ciz(ilk_sayfa_np, kutular)
            st.image(onizleme_img, caption="Izgara Kalibrasyonu", use_container_width=True)
