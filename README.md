# Dijital Görüntü İşleme ve Bilgisayarlı Görü Yol Haritası

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green?style=flat-square)
![NumPy](https://img.shields.io/badge/Library-NumPy-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Computer%20Vision%20Roadmap-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Curated%20Curriculum-success?style=flat-square)

Bu repository; sıfırdan bilgisayarlı görü (computer vision) ve dijital görüntü işleme (DIP) alanına adım atmak isteyen mühendisler, araştırmacılar ve öğrenciler için hazırlanmış kapsamlı bir **eğitim şablonu ve uygulama yol haritasıdır**. Temel piksel matrislerinden uzamsal konvolüsyon filtrelerine, geometrik dönüşümlerden morfolojik operatörlere kadar olan çekirdek algoritmaları hem matematiksel arka planı hem de pratik Jupyter notebook uygulamalarıyla sunar.

---

## Müfredat ve Bilgisayarlı Görü Pipeline Akışı

```mermaid
graph TD
    subgraph Ön İşleme ve Görüntü İyileştirme
        A[01 Piksel Matrisi & Histogram / CLAHE] --> B[02 Geometrik Dönüşümler & Afin]
        B --> C[09 Renk Uzayları: RGB / HSV / LAB Maskeleme]
    end
    subgraph Çekirdek Filtreleme ve Kenar Analizi
        C --> D[03 Uzamsal Konvolüsyon & Sobel Filtresi]
        D --> E[06 Canny Kenar Algılama & Gradyan Yönü]
    end
    subgraph İkili Dönüşüm ve Morfoloji
        E --> F[04 Eşikleme & İkili Maske]
        F --> G[05 Morfolojik Operatörler: Açma / Kapama]
    end
    subgraph Yüksek Seviye Çıkarım ve Tespit
        G --> H[07 Kontur Analizi & Geometrik Şekil Tanıma]
        H --> I[08 Hough Dönüşümü: Çizgi & Daire Tespiti]
        I --> J[10 Haar Cascade ile Yüz & Göz Tespiti]
    end
```

---

## Müfredat ve Modül Haritası

| No | Modül Adı | Temel Algoritma / Yöntem | Matematiksel Odak | Durum | Dizin |
|:---|:---|:---|:---|:---|:---|
| **01** | Temel Piksel ve Histogram | Histogram Eşitleme, CLAHE | Kümülatif Dağılım Fonksiyonu (CDF), Yerel Kontrast | Tamamlandı | `01-temel-piksel-ve-histogram-islemleri/` |
| **02** | Geometrik Dönüşümler | Afin Dönüşümü, Bilinear Interpolation | 3x3 Homojen Dönüşüm Matrisi, 4-Komşu Enterpolasyon | Tamamlandı | `02-geometrik-donusumler-ve-enterpolasyon/` |
| **03** | Uzamsal Filtreleme | Sobel Kenar Tespiti | 2B Ayrık Konvolüsyon, Gradyan Büyüklüğü ve Açısı | Tamamlandı | `03-uzamsal-filtreleme-ve-kenar-belirleme/` |
| **04** | Eşikleme ve İkili Görüntü | Dinamik / Uyarlamalı Eşikleme | Yerel Ortalama ve Gauss Pencereleme, Eşikleme | Tamamlandı | `04-esikleme-ve-ikili-goruntu-islemleri/` |
| **05** | Morfolojik Operatörler | Genişletme, Aşındırma, Açma, Kapama | Yapılandırıcı Eleman, Minkowski Küme İşlemleri | Tamamlandı | `05-morfolojik-goruntu-islemleri/` |
| **06** | Canny Kenar Algılama | Non-Maximum Suppression, Histerezis | Çift Eşikli Kenar İzleme | Tamamlandı | `06-canny-kenar-tespiti-ve-gradyan-yonu/` |
| **07** | Kontur Analizi | `findContours`, Geometrik Momentler | Yeşil Alan, Çevre, Sınırlayıcı Dikdörtgen | Tamamlandı | `07-kontur-analizi-ve-sekil-tanima/` |
| **08** | Hough Dönüşümü | Hough Çizgi ve Çember Tespiti | Parametre Uzayı Akümülatörü, Polar Koordinatlar | Tamamlandı | `08-hough-donusumu-cizgi-ve-daire-tespiti/` |
| **09** | Renk Uzayları | RGB, HSV, LAB ve Renk Maskeleme | Renk Özü (Hue), Doygunluk (Saturation), Parlaklık | Tamamlandı | `09-renk-uzaylari-ve-segmentasyon/` |
| **10** | Haar Cascade Sınıflandırma | AdaBoost Zayıf Öğreniciler, İntegral Görüntü | Haar Benzeri Öznitelikler ile Yüz Algılama | Tamamlandı | `10-haar-cascade-ile-yuz-ve-goz-tespiti/` |

---

## 1. Temel Piksel ve Histogram İşlemleri (`01-temel-piksel-ve-histogram-islemleri`)

Piksel değerlerinin histogram dağılımı üzerinden kontrast analizi ve dinamik aralık iyileştirmesi:
- **Histogram Eşitleme:** Görüntüdeki gri seviye dağılımını tüm dinamik aralığa yaymak için kümülatif olasılık dağılım fonksiyonu (CDF) kullanılır:
  $$s_k = T(r_k) = (L-1) \sum_{j=0}^k p_r(r_j) = \frac{L-1}{MN} \sum_{j=0}^k n_j$$
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Global eşitlemenin neden olduğu aşırı parlama ve gürültü amplifikasyonunu önlemek amacıyla görüntüyü yerel alt ızgaralara (tiles) böler ve kontrastı bir kırpma limiti (clip limit) ile sınırlar.

---

## 2. Geometrik Dönüşümler ve Enterpolasyon (`02-geometrik-donusumler-ve-enterpolasyon`)

Görüntü koordinat uzayının matris manipülasyonları:
- **Afin Dönüşüm Matrisi:** Doğruların paralelliğini koruyan doğrusal koordinat dönüşümü:
  $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- **Çift Doğrusal Enterpolasyon (Bilinear Interpolation):** Dönüşüm sonrası tamsayı olmayan koordinatlardaki piksel değerini belirlemek için en yakın 4 komşu pikselin ağırlıklı ortalaması alınır:
  $$f(x,y) \approx \frac{(x_2-x)(y_2-y)}{(x_2-x_1)(y_2-y_1)}f(Q_{11}) + \frac{(x-x_1)(y_2-y)}{(x_2-x_1)(y_2-y_1)}f(Q_{21}) + \frac{(x_2-x)(y-y_1)}{(x_2-x_1)(y_2-y_1)}f(Q_{12}) + \frac{(x-x_1)(y-y_1)}{(x_2-x_1)(y_2-y_1)}f(Q_{22})$$

---

## 3. Uzamsal Filtreleme ve Sobel Kenar Tespiti (`03-uzamsal-filtreleme-ve-kenar-belirleme`)

Piksel yoğunluğundaki ani değişimlerin (gradyanların) birinci türev çekirdekleri ile çıkarımı:
- **Sobel Konvolüsyon Çekirdekleri:**
  $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$
- **Toplam Kenar Gradyanı ve Yönü:**
  $$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$

---

## 4. Dinamik ve Uyarlamalı Eşikleme (`04-esikleme-ve-ikili-goruntu-islemleri`)

Düzensiz aydınlatma koşullarında sabit küresel eşik yerine, her pikselin etrafındaki yerel $N \times N$ komşuluk penceresinin ortalaması veya Gauss ağırlıklı ortalaması hesaplanarak dinamik eşik değeri belirlenir:
$$T(x,y) = \text{mean}_{k \in N(x,y)} I(k) - C$$

---

## 5. Morfolojik Operatörler (`05-morfolojik-goruntu-islemleri`)

İkili (binary) görüntüler üzerinde bir yapılandırıcı eleman (kernel) $B$ ile uygulanan küme teorisi işlemleri:
- **Aşındırma (Erosion):** $A \ominus B = \{ z \mid (B)_z \subseteq A \}$ (Nesne sınırlarını daraltır, küçük parazitleri yok eder)
- **Genişletme (Dilation):** $A \oplus B = \{ z \mid (\hat{B})_z \cap A \neq \emptyset \}$ (Delikleri kapatır, nesne sınırlarını genişletir)
- **Açma (Opening):** $(A \ominus B) \oplus B$ (Dış gürültüyü temizler)
- **Kapama (Closing):** $(A \oplus B) \ominus B$ (İç boşlukları ve çatlakları onarır)

---

## Yol Haritası Gelişim Durumu (10/10 Modül Tamamlandı)

Tüm temel ve ileri düzey klasik görüntü işleme modülleri kodlanmış ve doğrulanmıştır:
- [x] **Modül 01:** Temel Piksel ve Histogram (Histogram Eşitleme, CLAHE)
- [x] **Modül 02:** Geometrik Dönüşümler (Afin Dönüşümü, Bilinear Interpolation)
- [x] **Modül 03:** Uzamsal Filtreleme ve Sobel Kenar Tespiti
- [x] **Modül 04:** Dinamik ve Uyarlamalı Eşikleme
- [x] **Modül 05:** Morfolojik Operatörler (Genişletme, Aşındırma, Açma, Kapama)
- [x] **Modül 06:** Canny Kenar Algılama ve Çift Eşikli Histerezis
- [x] **Modül 07:** Kontur Çıkarımı, Nesne Sınırları ve Geometrik Şekil Sınıflandırma
- [x] **Modül 08:** Hough Çizgi ve Çember Dönüşümleri ile Şerit / Daire Tespiti
- [x] **Modül 09:** HSV / LAB Renk Uzaylarında Renk Filtreleme ve Segmentasyon
- [x] **Modül 10:** Haar Cascade Sınıflandırıcılar ile Gerçek Zamanlı Yüz ve Göz Algılama

---

## Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için:

```bash
git clone https://github.com/SuleymanToklu/image-processing.git
cd image-processing

python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib pillow jupyter

jupyter notebook
```