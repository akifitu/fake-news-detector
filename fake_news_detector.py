"""
Fake News Detector
=================

Bu modül, makine öğrenimi algoritmalarını kullanarak sahte haber tespiti yapar.
Çeşitli algoritmaları karşılaştırır ve en iyi performansı gösteren modeli seçer.

Desteklenen Algoritmalar:
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Random Forest

Geliştirici: Akif Kaya
GitHub: akifitu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')  # Uyarıları gizle

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError:
    pass

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK veri setleri indirilemedi, manuel stopwords kullanılacak")

ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

class FakeNewsDetector:
    """
    Sahte Haber Tespit Sınıfı
    =========================
    
    Bu sınıf, metin tabanlı haber verilerini analiz ederek sahte haber tespiti yapar.
    TF-IDF vektörizasyonu ve çeşitli makine öğrenimi algoritmaları kullanır.
    
    Attributes:
        vectorizer (TfidfVectorizer): Metin özellik çıkarma için TF-IDF vektörizer
        model (sklearn.base.BaseEstimator): Seçilen makine öğrenimi modeli
        stemmer (PorterStemmer): Kelime kökleri için stemmer (opsiyonel)
    
    Methods:
        load_data(): CSV dosyalarından veri yükler
        preprocess_text(text): Metni ön işlemden geçirir
        prepare_data(data): Veri setini model için hazırlar
        train_model(data): Modeli eğitir ve değerlendirir
        predict(text): Tek bir metin için tahmin yapar
        compare_models(data): Farklı algoritmaları karşılaştırır
    """
    
    def __init__(self):
        """
        FakeNewsDetector sınıfını başlatır.
        
        Varsayılan olarak TF-IDF vektörizer (5000 özellik) ve 
        Logistic Regression modeli kullanır.
        """
        # TF-IDF vektörizer: En önemli 5000 kelimeyi özellik olarak kullan
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Varsayılan model: Logistic Regression (hızlı ve etkili)
        self.model = LogisticRegression()
        
        # Stemmer: Kelime köklerini bulmak için (opsiyonel)
        self.stemmer = None
        try:
            self.stemmer = PorterStemmer()
        except:
            pass  # NLTK yüklü değilse manuel stemming kullanılacak
        
    def load_data(self):
        """Veri setlerini yükle ve birleştir"""
        print("Veri setleri yükleniyor...")
        
        # Fake news datasını yükle
        try:
            fake_data = pd.read_csv('archive/Fake.csv')
            fake_data['label'] = 0  # Fake news için 0
            print(f"Fake news sayısı: {len(fake_data)}")
        except Exception as e:
            print(f"Fake.csv dosyası yüklenirken hata: {e}")
            return None
            
        # True news datasını yükle  
        try:
            true_data = pd.read_csv('archive/True.csv')
            true_data['label'] = 1  # True news için 1
            print(f"True news sayısı: {len(true_data)}")
        except Exception as e:
            print(f"True.csv dosyası yüklenirken hata: {e}")
            return None
            
        # Veri setlerini birleştir
        data = pd.concat([fake_data, true_data], ignore_index=True)
        
        # Sütun isimlerini kontrol et ve text sütununu belirle
        print("Sütunlar:", data.columns.tolist())
        
        # Olası text sütun isimleri
        text_columns = ['text', 'content', 'article', 'news', 'body', 'description']
        text_column = None
        
        for col in text_columns:
            if col in data.columns:
                text_column = col
                break
                
        if text_column is None:
            # İlk string sütunu bul
            for col in data.columns:
                if data[col].dtype == 'object' and col != 'label':
                    text_column = col
                    break
        
        if text_column is None:
            print("Text sütunu bulunamadı!")
            return None
            
        print(f"Text sütunu olarak '{text_column}' kullanılacak")
        
        # Gerekli sütunları seç
        data = data[[text_column, 'label']].copy()
        data = data.rename(columns={text_column: 'text'})
        
        # Eksik değerleri temizle
        data = data.dropna()
        
        # Veri setini karıştır
        data = data.sample(frac=1).reset_index(drop=True)
        
        print(f"Toplam veri sayısı: {len(data)}")
        print(f"Label dağılımı:\n{data['label'].value_counts()}")
        
        return data
    
    def simple_stem(self, word):
        """Basit stemming fonksiyonu"""
        # Basit suffix removal
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def preprocess_text(self, text):
        """Metin ön işleme"""
        if pd.isna(text):
            return ""
            
        # Küçük harfe çevir
        text = text.lower()
        
        # Özel karakterleri ve sayıları temizle
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Stopwords'leri kaldır
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = ENGLISH_STOPWORDS
            
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Stemming uygula
        if self.stemmer:
            words = [self.stemmer.stem(word) for word in words]
        else:
            words = [self.simple_stem(word) for word in words]
        
        return ' '.join(words)
    
    def prepare_data(self, data):
        """Veriyi model için hazırla"""
        print("Veri ön işleme başlıyor...")
        
        # Text preprocessing
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        # Boş metinleri filtrele
        data = data[data['processed_text'] != ''].reset_index(drop=True)
        
        print(f"Ön işleme sonrası veri sayısı: {len(data)}")
        
        return data
    
    def train_model(self, data):
        """Modeli eğit"""
        print("Model eğitimi başlıyor...")
        
        # Features ve target'ı ayır
        X = data['processed_text']
        y = data['label']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set boyutu: {len(X_train)}")
        print(f"Test set boyutu: {len(X_test)}")
        
        # TF-IDF vectorization
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Model eğitimi
        self.model.fit(X_train_tfidf, y_train)
        
        # Tahminler
        y_train_pred = self.model.predict(X_train_tfidf)
        y_test_pred = self.model.predict(X_test_tfidf)
        
        # Performans değerlendirme
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTrain Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred, 
                                    target_names=['Fake', 'True']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fake', 'True'], 
                    yticklabels=['Fake', 'True'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix 'confusion_matrix.png' olarak kaydedildi.")
        
        return X_test, y_test, y_test_pred
    
    def predict(self, text):
        """Tek bir metin için tahmin yap"""
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return {
            'prediction': 'True' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'probabilities': {
                'fake': probability[0],
                'true': probability[1]
            }
        }
    
    def compare_models(self, data):
        """Farklı modelleri karşılaştır"""
        print("Farklı modeller karşılaştırılıyor...")
        
        X = data['processed_text']
        y = data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name} eğitiliyor...")
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # En iyi modeli seç
        best_model_name = max(results, key=results.get)
        self.model = models[best_model_name]
        self.model.fit(X_train_tfidf, y_train)
        
        print(f"\nEn iyi model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
        
        return results

def main():
    """Ana fonksiyon"""
    print("=== FAKE NEWS DETECTOR ===\n")
    
    # Detector'ı başlat 
    detector = FakeNewsDetector()
    
    # Veriyi yükle detector modeline yükleme
    data = detector.load_data()
    if data is None:
        print("Veri yükleme hatası!")
        return
    
    # Veriyi hazırladık
    data = detector.prepare_data(data)
    
    # Modeli eğittik
    X_test, y_test, y_test_pred = detector.train_model(data)
    
    # Farklı modelleri karşılaştırma
    model_results = detector.compare_models(data)
    
    # Örnek tahminleri yaparak sonuçları görüntüleme
    print("\n=== ÖRNEK TAHMİNLER ===")
    
    sample_texts = [
        "Scientists have discovered a new planet that could support life.",
        "Breaking: Aliens have landed in İstanbul according to unreliable sources.",
        "The stock market closed higher today following positive economic news.",
        "Shocking: This one weird trick doctors don't want you to know!"
    ]
    
    for text in sample_texts:
        result = detector.predict(text)
        print(f"\nText: {text[:100]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    print("\nModel eğitimi tamamlandı!")

if __name__ == "__main__":
    main() 