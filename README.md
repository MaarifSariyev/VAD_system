# Voice Activity Detection (VAD)

## 📌 Layihənin Məqsədi
Bu modelin əsas məqsədi real zamanlı və ya fayl üzərindən səsli siqnallarda danışıq hissələrini müəyyən etməkdir. VAD modeli, səsin danışıq və ya səssizlik olduğunu təyin edərək növbəti səs emal prosesləri üçün faydalı ön iş kimi çıxış edir.

---

## 🧠 İstifadə Edilən Alqoritm
- Enerji əsaslı sadə VAD yanaşması (MFCC və ya RMS əsasında)
- Gələcəkdə WebRTC VAD və ya neural VAD-lə əvəz oluna bilər

---

## 🛠️ Texnologiyalar və Kitabxanalar
- `numpy`
- `librosa`
- `sounddevice` (real-time üçün)
- `matplotlib` (vizualizasiya üçün)

---

## 📁 Fayl Quruluşu
VAD/
├── vad_model.py # VAD funksiyası
├── real_time_vad.py # Mikrofon ilə real zamanlı test
├── test_with_file.py # Audio faylı ilə test
├── utils.py # Yardımçı funksiyalar
└── README_VAD.md

## 📈 İmkanlar və Gələcək Planlar
- VAD nəticələrinin spektrogram ilə birlikdə vizuallaşdırılması
- Gələcəkdə neural VAD tətbiqi
- WebRTC VAD inteqrasiyası
