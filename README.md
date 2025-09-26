# Akciğer Röntgenlerinden Zatürre Tespiti: Bir Derin Öğrenme Projesi 🫁

Bu proje, Akbank Derin Öğrenme Bootcamp'i kapsamında geliştirilmiş olup, göğüs röntgeni (X-ray) görüntülerinden zatürre (pneumonia) vakalarını yüksek doğrulukla tespit edebilen bir derin öğrenme modelinin geliştirilmesini ve analiz edilmesini içermektedir. Proje, sadece yüksek bir doğruluk oranına ulaşmayı değil, aynı zamanda modelin kararlarının yorumlanabilirliğini ve en iyi pratiklerin uygulanmasını da ön planda tutmaktadır.

## 📋 Proje İçeriği

1.  [Proje Metodolojisi](#-proje-metodolojisi)
2.  [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
3.  [Veri Seti](#-veri-seti)
4.  [Depo Yapısı](#-depo-yapısı)
5.  [Nasıl Çalıştırılır?](#-nasıl-çalıştırılır)
6.  [Elde Edilen Sonuçlar](#-elde-edilen-sonuçlar)
7.  [Gelecek Çalışmalar](#-gelecek-çalışmalar)

## 🛠️ Proje Metodolojisi

Proje, bir derin öğrenme probleminin başından sonuna kadar nasıl ele alınacağını gösteren sistematik bir yaklaşımla geliştirilmiştir:

### 1. Keşifsel Veri Analizi (EDA)
Veri setinin yapısı incelenmiş, sınıflar (`NORMAL`, `PNEUMONIA`) arasındaki örnek sayısı dağılımı görselleştirilmiştir. Bu analiz, eğitim setindeki belirgin **sınıf dengesizliğini** ortaya koymuş ve bu problemi çözmek için stratejiler geliştirilmesi gerektiğini göstermiştir.

### 2. Baseline (Temel) Model
Projenin başında bir referans noktası oluşturmak amacıyla, sıfırdan basit bir Evrişimli Sinir Ağı (CNN) modeli inşa edilmiştir. Bu modelin performansı, daha sonra geliştirilecek olan karmaşık modellerin başarısını ölçmek için bir taban çizgisi (baseline) olarak kullanılmıştır.

### 3. Gelişmiş Model (Transfer Learning ile VGG16)
ImageNet veri seti üzerinde eğitilmiş, başarısı kanıtlanmış **VGG16** mimarisi, Transfer Öğrenme tekniği ile projeye adapte edilmiştir. Modelin esnekliğini artırmak ve Grad-CAM gibi ileri analizlere olanak tanımak için Keras'ın **Functional API**'si tercih edilmiştir.

### 4. Veri Pipeline Optimizasyonu ve Veri Artırma
* **Yüksek Performanslı Veri Hattı:** Eski `ImageDataGenerator` yerine, modern ve verimli `tf.data.Dataset` pipeline'ı kurulmuştur. `.cache()` ve `.prefetch()` gibi optimizasyonlarla GPU'nun veri bekleme süresi minimuma indirilerek eğitim süreci hızlandırılmıştır.
* **Veri Artırma (Data Augmentation):** Modelin genelleme yeteneğini artırmak ve ezberlemeyi (overfitting) önlemek amacıyla, eğitim setindeki görüntülere `RandomFlip`, `RandomRotation`, `RandomZoom` gibi anlık (on-the-fly) dönüşümler uygulanmıştır.

### 5. Akıllı Eğitim Stratejileri
* **Sınıf Ağırlıkları:** EDA'da tespit edilen sınıf dengesizliğini gidermek için, eğitim sırasında azınlık sınıfına (`NORMAL`) daha fazla önem veren `class_weight` tekniği kullanılmıştır.
* **Callback'ler:** Eğitim süreci, `EarlyStopping` (gereksiz eğitimi durdurma), `ModelCheckpoint` (sadece en iyi modeli kaydetme) ve `ReduceLROnPlateau` (öğrenme yavaşladığında öğrenme oranını düşürme) gibi `callback`'ler ile akıllı bir şekilde yönetilmiştir.

### 6. Nihai Model (EfficientNetB3 ile Fine-Tuning)
Projenin en iyi sonucunu elde etmek amacıyla, VGG16'dan daha modern ve verimli bir mimari olan **EfficientNetB3** kullanılmıştır. Bu model, iki aşamalı bir **ince ayar (fine-tuning)** stratejisi ile eğitilmiştir:
1.  **Aşama 1:** Modelin temel katmanları dondurularak sadece üste eklenen özel sınıflandırıcı eğitilmiştir.
2.  **Aşama 2:** Temel modelin son 20 katmanı "çözülerek", çok düşük bir öğrenme oranıyla tüm model üzerinde ince ayar yapılmıştır.

### 7. Model Yorumlanabilirliği (Grad-CAM)
Modelin bir "kara kutu" olmasını önlemek amacıyla **Grad-CAM** tekniği uygulanmıştır. Bu sayede, modelin "Zatürre" teşhisi koyarken görüntünün hangi bölgelerine odaklandığı görselleştirilerek, karar mekanizması yorumlanabilir hale getirilmiştir.

## 💻 Kullanılan Teknolojiler
* **Python 3**
* **TensorFlow & Keras:** Derin öğrenme modellerini oluşturmak, eğitmek ve değerlendirmek için.
* **Scikit-learn:** Model değerlendirme metrikleri (Classification Report, Confusion Matrix, ROC Curve) için.
* **Pandas:** Veri analizi ve manipülasyonu için.
* **Matplotlib & Seaborn:** Veri ve sonuçların görselleştirilmesi için.
* **Numpy:** Sayısal operasyonlar için.
* **Jupyter Notebook:** Projenin interaktif olarak geliştirilmesi ve belgelenmesi için.

## 📁 Depo Yapısı
```
.
├── rontgenle-zaturre-tespiti.ipynb   # Projenin tüm adımlarını içeren ana Jupyter Notebook dosyası
├── requirements.txt                  # Gerekli Python kütüphaneleri ve versiyonları
├── .gitignore                        # Git tarafından takip edilmeyecek dosyalar (veri setleri, modeller vb.)
└── README.md                         # Bu dosya, projenin özeti ve kılavuzu
```

## 🚀 Nasıl Çalıştırılır?

1.  **Depoyu Klonlayın:**
    ```bash
    git clone <depo-adresi>
    cd <proje-klasoru>
    ```
2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Veri Setini İndirin:**
    * Projeyi çalıştırmak için [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) veri setini indirmeniz gerekmektedir.
    * İndirilen `chest_xray` klasörünü, projenin çalıştığı ortamda Kaggle'ın standart `../input/` dizin yapısına uygun bir yere yerleştirin.

4.  **Notebook'u Çalıştırın:**
    * `rontgenle-zaturre-tespiti.ipynb` dosyasını bir Jupyter ortamında açın.
    * Hücreleri sırayla çalıştırarak projenin tüm adımlarını yeniden üretebilirsiniz.

## 📈 Elde Edilen Sonuçlar

Proje boyunca geliştirilen modellerin test seti üzerindeki performansları karşılaştırılmıştır. Yapılan optimizasyonlar ve kullanılan gelişmiş mimariler, modelin teşhis yeteneğini önemli ölçüde artırmıştır.

| Model Mimarisi                     | Test Başarımı (Accuracy) | Temel İyileştirme |
| ---------------------------------- | ------------------------ | ----------------- |
| **Temel (Baseline) CNN** | ~87-89%                  | -                 |
| **Nihai Model (EfficientNetB3)** | **~92-94%** | **~+5%** |

*Not: Başarı oranları, eğitimin stokastik doğası gereği her çalıştırmada küçük farklılıklar gösterebilir.*

## 🔮 Gelecek Çalışmalar

Bu projenin üzerine inşa edilebilecek potansiyel geliştirme alanları şunlardır:

* **Çok Sınıflı Teşhis:** Projeyi, COVID-19 veya tüberküloz gibi diğer akciğer hastalıklarını da teşhis edebilen çok sınıflı bir modele dönüştürmek.
* **K-Fold Çapraz Doğrulama:** Modelin performansını daha güvenilir bir şekilde ölçmek için K-Fold gibi çapraz doğrulama teknikleri uygulamak.
* **Modeli Hayata Geçirme (Deployment):** Eğitilmiş modeli, Streamlit veya Gradio gibi araçlarla basit bir web arayüzüne entegre ederek canlı tahminler yapabilen bir demo uygulaması oluşturmak.
