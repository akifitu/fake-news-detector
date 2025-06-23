# Fake News Detector 🕵️‍♂️

Bu proje, makine öğrenimi algoritmalarını kullanarak sahte haber tespiti yapan bir Python uygulamasıdır.

## 📋 İçindekiler
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Performansı](#model-performansı)
- [Veri Seti](#veri-seti)
- [Teknolojiler](#teknolojiler)
- [Sonuçlar](#sonuçlar)

## ✨ Özellikler

- **Çoklu Model Karşılaştırması**: Logistic Regression, Naive Bayes, SVM ve Random Forest algoritmalarını karşılaştırır
- **Otomatik Model Seçimi**: En iyi performansı gösteren modeli otomatik olarak seçer
- **Metin Ön İşleme**: NLTK ile gelişmiş metin temizleme ve işleme
- **Görselleştirme**: Confusion matrix ve performans metrikleri
- **Gerçek Zamanlı Tahmin**: Yeni metinler için anında sahte haber tespiti

## 🚀 Kurulum

### Gereksinimler
```bash
python >= 3.7
```

### Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### Gerekli NLTK Verileri
Program ilk çalıştırıldığında gerekli NLTK veri setlerini otomatik olarak indirir.

## 💻 Kullanım

### Temel Kullanım
```bash
python3 fake_news_detector.py
```

### Programatik Kullanım
```python
from fake_news_detector import FakeNewsDetector

# Model oluştur
detector = FakeNewsDetector()

# Veri yükle ve eğit
data = detector.load_data()
data = detector.prepare_data(data)
detector.train_model(data)

# Tahmin yap
result = detector.predict("Your news text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📊 Model Performansı

### Test Sonuçları
- **Random Forest**: 99.75% accuracy (En iyi model)
- **SVM**: 99.18% accuracy
- **Logistic Regression**: 98.26% accuracy
- **Naive Bayes**: 92.24% accuracy

### Confusion Matrix
Program çalıştırıldığında `confusion_matrix.png` dosyası oluşturulur.

## 📁 Veri Seti

Proje iki ana veri seti kullanır:
- `archive/Fake.csv`: Sahte haber örnekleri (23,481 kayıt)
- `archive/True.csv`: Gerçek haber örnekleri (21,417 kayıt)

**Toplam**: 44,898 haber metni

### Veri Seti Özellikleri
- **Sütunlar**: title, text, subject, date
- **Dil**: İngilizce
- **Kaynak**: Çeşitli haber siteleri ve fact-checking platformları

## 🛠 Teknolojiler

- **Python 3.12**
- **Pandas**: Veri manipülasyonu
- **Scikit-learn**: Makine öğrenimi algoritmaları
- **NLTK**: Doğal dil işleme
- **Matplotlib/Seaborn**: Görselleştirme
- **NumPy**: Sayısal hesaplamalar

## 🔬 Algoritmalar

### 1. Logistic Regression
- Doğrusal sınıflandırma algoritması
- Hızlı eğitim ve tahmin
- Yorumlanabilir sonuçlar

### 2. Naive Bayes (MultinomialNB)
- Metin sınıflandırma için optimize edilmiş
- Bağımsızlık varsayımı
- Küçük veri setlerinde etkili

### 3. Support Vector Machine (SVM)
- Yüksek boyutlu veriler için ideal
- Kernel trick kullanımı
- Robust sınır belirleme

### 4. Random Forest
- Birden fazla karar ağacının kombinasyonu
- Overfitting'e karşı dirençli
- En yüksek accuracy (99.75%)

## 📈 Metin Ön İşleme

1. **Küçük harfe dönüştürme**
2. **Özel karakter ve sayıları temizleme**
3. **Stopwords kaldırma** (manuel ve NLTK)
4. **Stemming** (PorterStemmer)
5. **TF-IDF Vektörizasyonu** (5000 özellik)

## 🎯 Sonuçlar

### Başarı Metrikleri
- **Test Accuracy**: %99.75
- **Precision**: %98 (Fake), %98 (True)
- **Recall**: %98 (Fake), %98 (True)
- **F1-Score**: %98 (Fake), %98 (True)

### Örnek Tahminler
```
"Scientists have discovered a new planet that could support life."
→ Prediction: Fake (96.0% confidence)

"Breaking: Aliens have landed in İstanbul according to unreliable sources."
→ Prediction: Fake (89.0% confidence)
```

## 📝 Dosya Yapısı

```
OptimisationProject/
├── fake_news_detector.py      # Ana model dosyası
├── requirements.txt           # Python bağımlılıkları
├── confusion_matrix.png       # Model performans görseli
├── archive/
│   ├── Fake.csv              # Sahte haber veri seti
│   └── True.csv              # Gerçek haber veri seti
└── README.md                 # Bu dosya
```

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👤 Geliştirici

**Akif Kaya** - [akifitu](https://github.com/akifitu)

## 📞 İletişim

Herhangi bir soru veya öneri için GitHub Issues kullanabilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! 