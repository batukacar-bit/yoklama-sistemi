import streamlit as st
import os
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageOps
from deepface import DeepFace

# Klasör ve Dosya Yolları
KLASOR_ADI = "ogrenciler"
if os.path.exists(KLASOR_ADI):
    VERI_YOLU = os.path.abspath(KLASOR_ADI)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VERI_YOLU = os.path.join(BASE_DIR, KLASOR_ADI)

os.makedirs(VERI_YOLU, exist_ok=True)
EXCEL_DOSYASI = os.path.join(os.path.dirname(VERI_YOLU), "yoklama.xlsx")
MODEL_NAME = "Facenet512"

st.set_page_config(page_title="Akıllı Yoklama Portalı", layout="wide", page_icon="🎓")
st.title("🎓 Akıllı Öğrenci Yoklama & Kayıt Portalı")

# ================= SEKME YARDIMCI FONKSİYONLARI =================
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

def pdf_kes_ve_kaydet_sirali(pdf_bytes, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran, isim_listesi):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    toplam_kesilen = 0
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

            if toplam_kesilen < len(isimler):
                ogrenci_adi = isimler[toplam_kesilen]
                dosya_adi = temiz_isim_uret(ogrenci_adi)
            else:
                dosya_adi = f"Ogrenci_{toplam_kesilen + 1}"

            Image.fromarray(foto).save(os.path.join(VERI_YOLU, f"{dosya_adi}.jpg"), "JPEG")
            toplam_kesilen += 1

    return toplam_kesilen

# ================= ORTAK MOTOR & VERİTABANI =================
def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def yoklama_kaydet(etiket):
    zaman = datetime.now().strftime("%H:%M:%S")
    tarih = datetime.now().strftime("%Y-%m-%d")
    
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

@st.cache_resource
def ogrenci_veritabanini_yukle():
    veritabani = []
    dosyalar = [f for f in os.listdir(VERI_YOLU) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for dosya in dosyalar:
        tam_yol = os.path.join(VERI_YOLU, dosya)
        dosya_kok = os.path.splitext(dosya)[0]
        
        parcalar = dosya_kok.split('_')
        if len(parcalar) >= 2 and parcalar[0].isdigit():
            numara = parcalar[0]
            ad_soyad = " ".join(parcalar[1:])
            gorunen_etiket = f"{numara} - {ad_soyad}"
        else:
            gorunen_etiket = dosya_kok.replace('_', ' ')

        try:
            img_pil = Image.open(tam_yol).convert("RGB")
            img_pil = ImageOps.exif_transpose(img_pil)
            img_np = np.array(img_pil)

            emb = DeepFace.represent(
                img_path=img_np,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False
            )[0]["embedding"]
            veritabani.append({"isim": gorunen_etiket, "embedding": emb})
        except Exception as e:
            st.sidebar.warning(f"Okunamadı: {dosya} -> {e}")
            
    return veritabani

# ================= SEKMELER =================
sekme1, sekme2, sekme3 = st.tabs([
    "📸 Kamera & Canlı Yoklama",
    "👤 Hızlı Öğrenci Kayıt",
    "📄 PDF Kalibrasyon & Toplu Aktarım"
])

# --- 1. SEKME: KAMERA & YOKLAMA ---
with sekme1:
    col_ust1, col_ust2 = st.columns([3, 1])
    with col_ust1:
        st.subheader("📸 Canlı Sınıf Yoklaması")
    with col_ust2:
        if st.button("🔄 Veritabanını Yenile"):
            st.cache_resource.clear()
            st.rerun()

    ogrenci_db = ogrenci_veritabanini_yukle()
    esik_degeri = st.slider("Hassasiyet Eşiği", 0.25, 0.60, 0.38, 0.01)

    col_sol, col_sag = st.columns([1.1, 0.9])

    with col_sol:
        if len(ogrenci_db) == 0:
            st.warning("⚠️ Sistemde kayıtlı öğrenci yok. '👤 Hızlı Öğrenci Kayıt' sekmesinden öğrenci ekleyebilirsiniz.")
        else:
            secim = st.radio("İşlem Türü:", ["Kameradan Çek", "Fotoğraf Yükle"], horizontal=True)
            gorsel = None

            if secim == "Kameradan Çek":
                kamera = st.camera_input("Kameraya bakın ve fotoğraf çekin")
                if kamera:
                    gorsel = Image.open(kamera)
            else:
                dosya = st.file_uploader("Sınıf Fotoğrafı Yükle", type=['jpg', 'jpeg', 'png'], key="yoklama_foto")
                if dosya:
                    gorsel = Image.open(dosya)

            if gorsel is not None:
                gorsel = ImageOps.exif_transpose(gorsel)
                gorsel.thumbnail((1280, 1280))

                frame_rgb = np.array(gorsel)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                with st.spinner("Yüzler taranıyor ve eşleştiriliyor..."):
                    try:
                        yuz_tespitleri = DeepFace.extract_faces(
                            img_path=frame_bgr,
                            detector_backend="opencv",
                            enforce_detection=False,
                            align=True
                        )
                    except Exception as e:
                        yuz_tespitleri = []
                        st.error(f"Yüz arama hatası: {e}")

                gecerli_yuzler = [y for y in yuz_tespitleri if y.get('confidence', 0) > 0.40]

                if len(gecerli_yuzler) == 0:
                    st.error("⚠️ Fotoğrafta net bir yüz tespit edilemedi!")
                    st.image(frame_rgb, use_container_width=True)
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

                        try:
                            kare_emb = DeepFace.represent(
                                img_path=yuz_kare,
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
                            st.success(f"✅ **{isim}** tanındı! (Saat: {saat})")
                        else:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
                            cv2.putText(frame_rgb, f"Bilinmiyor ({mesafe:.2f})", (x, max(25, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

                    st.image(frame_rgb, caption="Yoklama Analiz Sonucu", use_container_width=True)

    with col_sag:
        st.subheader("📋 Güncel Yoklama Tablosu")
        if os.path.exists(EXCEL_DOSYASI):
            try:
                df_guncel = pd.read_excel(EXCEL_DOSYASI)
                st.dataframe(df_guncel, use_container_width=True, height=430)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🗑️ Yoklamayı Sıfırla"):
                        os.remove(EXCEL_DOSYASI)
                        st.rerun()
                with c2:
                    with open(EXCEL_DOSYASI, "rb") as f:
                        st.download_button(
                            label="📥 Excel'i İndir",
                            data=f,
                            file_name=f"yoklama_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            except Exception:
                st.write("Henüz yoklama kaydı yok.")
        else:
            st.write("Henüz yoklama kaydı yok.")

# --- 2. SEKME: HIZLI ÖĞRENCİ KAYIT ---
with sekme2:
    st.subheader("👤 Yeni Öğrenci Kayıt Formu")
    st.markdown("Öğrenci bilgilerini girip net bir yüz fotoğrafı yükleyin veya kameradan çekin.")

    c_form1, c_form2 = st.columns([1, 1])

    with c_form1:
        ogr_ad = st.text_input("Öğrenci Adı:", placeholder="Örn: Ahmet").strip()
        ogr_soyad = st.text_input("Öğrenci Soyadı:", placeholder="Örn: Yılmaz").strip()
        ogr_no = st.text_input("Öğrenci Numarası (Opsiyonel):", placeholder="Örn: 150210001").strip()

        kayit_yontemi = st.radio("Fotoğraf Kaynağı:", ["Kamera ile Çek", "Fotoğraf Yükle"], horizontal=True, key="kayit_yontem")
        kayit_foto = None

        if kayit_yontemi == "Kamera ile Çek":
            cam_foto = st.camera_input("Kameraya düz bakın ve çekin", key="ogr_cam")
            if cam_foto:
                kayit_foto = Image.open(cam_foto)
        else:
            yuklenen = st.file_uploader("Net bir vesikalık / selfie seçin", type=['jpg', 'jpeg', 'png'], key="ogr_dosya")
            if yuklenen:
                kayit_foto = Image.open(yuklenen)

        kaydet_butonu = st.button("💾 Öğrenciyi Sisteme Kaydet", type="primary", use_container_width=True)

    with c_form2:
        if kayit_foto is not None:
            kayit_foto = ImageOps.exif_transpose(kayit_foto)
            st.image(kayit_foto, caption="Yüklenecek Fotoğraf Önizlemesi", width=280)

        if kaydet_butonu:
            if not ogr_ad or not ogr_soyad:
                st.error("⚠️ Lütfen öğrencinin Adını ve Soyadını eksiksiz girin!")
            elif kayit_foto is None:
                st.error("⚠️ Lütfen bir fotoğraf çekin veya yükleyin!")
            else:
                with st.spinner("Yüz taranıyor ve kaydediliyor..."):
                    try:
                        foto_np = np.array(kayit_foto.convert("RGB"))
                        yuz_kontrol = DeepFace.extract_faces(
                            img_path=foto_np,
                            detector_backend="opencv",
                            enforce_detection=False
                        )
                        guvenli_yuzler = [y for y in yuz_kontrol if y.get('confidence', 0) > 0.40]

                        if len(guvenli_yuzler) == 0:
                            st.warning("⚠️ Fotoğrafta net bir yüz algılanamadı, ancak yine de kaydediliyor...")

                        ad_soyad_temiz = temiz_isim_uret(f"{ogr_ad} {ogr_soyad}")
                        if ogr_no:
                            no_temiz = re.sub(r'[^a-zA-Z0-9]', '', ogr_no)
                            dosya_adi = f"{no_temiz}_{ad_soyad_temiz}.jpg"
                        else:
                            dosya_adi = f"{ad_soyad_temiz}.jpg"

                        kayit_yolu = os.path.join(VERI_YOLU, dosya_adi)
                        kayit_foto.convert("RGB").save(kayit_yolu, "JPEG", quality=95)

                        st.cache_resource.clear()
                        st.success(f"🎉 **{ogr_ad} {ogr_soyad}** başarıyla kaydedildi!")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Kayıt sırasında hata oluştu: {e}")

    st.markdown("---")
    st.subheader("📚 Kayıtlı Öğrenci Listesi")
    mevcut_ogrenciler = [f for f in os.listdir(VERI_YOLU) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if mevcut_ogrenciler:
        st.info(f"Sistemde şu an toplam **{len(mevcut_ogrenciler)}** öğrenci kayıtlı.")
        cols_liste = st.columns(6)
        for i, dosya in enumerate(mevcut_ogrenciler):
            dosya_koku = os.path.splitext(dosya)[0].replace('_', ' ')
            with cols_liste[i % 6]:
                img_path = os.path.join(VERI_YOLU, dosya)
                st.image(img_path, caption=dosya_koku, use_container_width=True)
    else:
        st.write("Henüz sisteme kayıtlı öğrenci bulunmuyor.")

# --- 3. SEKME: PDF İLE TOPLU YÜKLEME ---
with sekme3:
    st.subheader("Görsel Kalibrasyon & Esnek İsim Listesi Eşleme (PDF)")
    yuklenen_dosya = st.file_uploader("Sınıf Listesi PDF Dosyası Seçin", type=["pdf"])

    if yuklenen_dosya is not None:
        pdf_bytes = yuklenen_dosya.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        sayfa1 = doc[0]
        pix = sayfa1.get_pixmap(dpi=150)
        ilk_sayfa_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
        if pix.n == 4:
            ilk_sayfa_np = cv2.cvtColor(ilk_sayfa_np, cv2.COLOR_RGBA2RGB)

        H, W, _ = ilk_sayfa_np.shape
        col_onizleme, col_ayarlar = st.columns([1.5, 1.1])

        with col_ayarlar:
            st.markdown("### 📝 Öğrenci İsim Listesi")
            isimler_text = st.text_area("İsimleri buraya yapıştırın:", height=180, placeholder="Ahmet Yılmaz\nMehmet Demir...")
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

            kutular = kutulari_hesapla(W, H, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran)

            if st.button("✂️ Fotoğrafları İsimlerle Kaydet", type="primary"):
                with st.spinner("Kaydediliyor..."):
                    toplam = pdf_kes_ve_kaydet_sirali(pdf_bytes, cols, rows, m_left, m_right, m_top, m_bottom, satirlar_arasi, sutunlar_arasi, foto_oran, isimler_text)
                    st.cache_resource.clear()
                    st.success(f"🎉 {toplam} öğrenci kaydedildi!")
                    st.rerun()

        with col_onizleme:
            onizleme_img = onizleme_ciz(ilk_sayfa_np, kutular)
            st.image(onizleme_img, use_container_width=True)
