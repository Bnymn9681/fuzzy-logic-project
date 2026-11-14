# Meta-Sezgisel Optimize Edilmiş ANFIS ve YSA Modelleri Kullanarak BİST 100 Endeksi Tahmini

## Öğrenci Bilgileri
* **Ad Soyad:** Bünyamin Öztürk
* **Ders:** Bulanık Mantık Dersi Dönem Projesi

## Referans Makale
Kazak, H., Kumar, S., Gündüz, M. A., & Akcan, A. T. (2025). *Metaheuristic-optimized ANFIS and ANN models for stock price forecasting: evidence from the Borsa Istanbul 100 index*. Discover Artificial Intelligence, 5(1), 272.

## Problem Tanımı
Bu proje, finansal piyasaların (özellikle Borsa İstanbul 100 endeksinin) doğrusal olmayan ve yüksek volatilite içeren yapısını tahmin etme problemini ele almaktadır. Geleneksel zaman serisi yöntemlerinin yetersiz kaldığı bu alanda, makroekonomik değişkenlerin borsa üzerindeki etkileri yapay zeka ile modellenmektedir.

## Yöntem
BİST 100 endeksinin fiyat tahminlemesini gerçekleştirmek için referans makalede sunulan ANFIS (Adaptif Ağ Tabanlı Bulanık Çıkarım Sistemi) mimarisi temel alınmış ve PyTorch kullanılarak yeniden gerçekleştirilmiştir (reproduction). Karşılaştırma amacıyla Çoklu Doğrusal Regresyon ve Yapay Sinir Ağı (YSA) temel (baseline) modelleri de geliştirilmiştir.

## Bulanık Sistem Mimarisi
* **Girdiler:** Altın/TL Kuru (GOLD_TL), VIX Endeksi, 2 Yıllık Devlet Tahvili Faizi (BOND_2Y)
* **Çıktı:** BİST 100 Endeksi
* **Temel (Baseline) Mimari:** 3 Girdi, Takagi-Sugeno Çıkarımı, 8 Kural (Gaussian Üyelik Fonksiyonu)
* **Özgün Katkı (Proposed):** 3 Girdi, Takagi-Sugeno Çıkarımı, 16 Kural (Generalized Bell - G-Bell Üyelik Fonksiyonu)

## Veri Seti
Deneylerde, referans makaleye sadık kalınarak `01.01.2010 - 31.12.2023` tarih aralığında toplanmış **haftalık** finansal veriler kullanılmıştır (Toplam 720 gözlem). Veriler otokorelasyonu önlemek için logaritmik birinci farklara dönüştürülmüş ve 0-1 aralığında min-max normalizasyonu ile ölçeklendirilmiştir. Veri seti 540 hafta eğitim (train) ve 180 hafta test (test) olarak bölünmüştür.

## Kurulum ve Gereksinimler
Projeyi yerel makinenizde çalıştırmak için Python 3.10+ kurulu olmalıdır. Gerekli kütüphaneleri yüklemek için:

```bash
git clone https://github.com/Bnymn9681/fuzzy-logic-project
cd fuzzy-logic-project
pip install -r requirements.txt
```

## Nasıl Çalıştırılır
1. `src/1_data_preprocessing.ipynb`: Ham verileri işler ve `/data` klasörüne kaydeder.
2. `src/2_baseline_models.ipynb`: Regresyon ve YSA modellerini eğiterek test eder.
3. `src/3_anfis_reproduction.ipynb`: Makaledeki referans (Gaussian) ANFIS modelini çalıştırır.
4. `src/4_proposed_extension.ipynb`: Özgün katkı olan G-Bell üyelik fonksiyonlu geliştirilmiş ANFIS modelini test eder.

## Yeniden Gerçekleştirme (Reproduction) Deneyleri
Makaledeki referans yöntemin yeniden gerçekleştirilmesi aşamasında, PyTorch üzerinde kurulan Takagi-Sugeno tabanlı ANFIS modeli 1500 epok boyunca Adam optimizasyonu ile eğitilmiş ve test seti üzerinde çalıştırılmıştır.

## Önerilen Özgün Katkı (Proposed Extension)
Sistemin finansal şoklara karşı esnekliğini artırmak amacıyla, referans çalışmadaki Gaussian üyelik fonksiyonları yerine 3 parametreli (genişlik, eğim, merkez) **Generalized Bell (G-Bell)** üyelik fonksiyonları sisteme entegre edilmiştir. Kural sayısı 8'den 16'ya çıkarılarak öğrenme kapasitesi artırılmıştır.

## Deneysel Sonuçlar
| Model | MSE (Hata) | MAE (Hata) |
| :--- | :--- | :--- |
| Çoklu Doğrusal Regresyon | 0.00441 | 0.03268 |
| YSA (Baseline) | 0.00659 | 0.04974 |
| ANFIS (Yeniden Gerçekleştirme) | 0.01024 | 0.05684 |
| **G-Bell ANFIS (Özgün Katkı)** | **0.00758** | **0.04838** |

Sonuçlar, G-Bell mimarisinin temel ANFIS modeline göre hata oranını (MSE) yaklaşık %26 oranında düşürdüğünü göstermektedir.

## Sınırlılıklar
Gelişmekte olan piyasalardaki ani jeopolitik ve ekonomik şoklar (örneğin pandemi veya ani faiz kararları), sadece üç makroekonomik girdiyle tam olarak modellenememektedir. Sistem, uzun vadeli trendleri yakalamada başarılı olsa da ani kırılımlarda hata payı artmaktadır.

## Kaynaklar
1. Kazak, H. et al. (2025). Discover Artificial Intelligence, 5(1), 272.