# BİST 100 Endeksi Tahmininde Sugeno Tipi ANFIS: Gaussian ve G-Bell Üyelik Fonksiyonlarının Karşılaştırmalı İncelenmesi

Bulanık Mantık dersi dönem projesi — makale yeniden gerçekleştirme, özgün katkı ve kontrollü deneyler.

## Öğrenci Bilgileri

* **Ad Soyad:** Bünyamin Öztürk
* **Ders:** Bulanık Mantık

## Referans Makale

H. Kazak, S. Kumar, M. A. Gündüz, A. T. Akcan, and H. H. Bilgiç, "Metaheuristic-optimized ANFIS and ANN models for stock price forecasting: Evidence from the Borsa Istanbul 100 index," *Discover Artificial Intelligence*, vol. 5, Art. no. 272, Oct. 2025, doi: 10.1007/s44163-025-00395-6.

## Problem Tanımı

Proje, BİST 100 endeksinin haftalık hareketlerinin üç makroekonomik değişken (Altın/TL kuru, VIX endeksi, 2 yıllık devlet tahvili faizi) ile tahmin edilmesi problemini ele alır. Referans makalede bu problem için meta-sezgisel algoritmalarla optimize edilmiş ANFIS ve YSA modelleri önerilmektedir.

## Yöntem ve MRR Kapsamı

Referans makaledeki Takagi–Sugeno tipi ANFIS mimarisi PyTorch ile sıfırdan yeniden gerçekleştirilmiştir. Karşılaştırma için çoklu doğrusal regresyon, YSA (15 nöron, tanh) ve bir **naif tahminci** (eğitim kümesi ortalamasını sabit tahmin eden model) kullanılmıştır.

**Kapsam dışı:** Makalenin meta-sezgisel eğitim bileşeni (PSO/ACO) minimum yeniden gerçekleştirme gereksinimi (MRR) kapsamı dışında bırakılmış, parametre optimizasyonu Adam algoritmasıyla yapılmıştır. Bu nedenle sonuçlar makalenin ANFIS-PSO/ACO değerleriyle birebir kıyaslanamaz.

**Makalede belirtilmeyen kararlar:** Makale üyelik fonksiyonu türünü ve kural sayısını belirtmemektedir. Gaussian üyelik fonksiyonu ve 8 kural seçimleri bu çalışmaya aittir.

## Bulanık Sistem Mimarisi

* **Girdiler:** Altın/TL kuru (GOLD_TL), VIX endeksi, 2 yıllık devlet tahvili faizi (BOND_2Y)
* **Çıktı:** BİST 100 endeksi
* **Baseline (A0):** 3 girdi, Takagi–Sugeno çıkarımı, 8 kural, Gaussian üyelik fonksiyonu
* **Önerilen (A4):** 3 girdi, Takagi–Sugeno çıkarımı, 16 kural, Generalized Bell (G-Bell) üyelik fonksiyonu, lr = 0,005 / 2500 epok
* Kural ateşleme güçleri logaritma uzayında toplanıp softmax ile normalize edilir (sayısal kararlılık için).
* Gaussian ve G-Bell modelleri adil karşılaştırma için aynı rastgele başlatma şemasını kullanır.

## Veri Seti

Investing.com üzerinden 01.01.2010–31.12.2023 aralığında haftalık (Cuma kapanışı) veriler toplanmıştır; her seri için 731 gözlem. Seriler W-FRI frekansına hizalanmış, yalnızca tatil boşlukları için en fazla iki haftalık ileri doldurma uygulanmıştır. Otokorelasyon ve değişen varyansı azaltmak için logaritmik birinci fark alınmış (730 gözlem), ardından MinMaxScaler ile 0–1 aralığına ölçeklenmiştir.

* **Ölçekleme:** yalnızca eğitim kümesi üzerinde `fit`, test kümesine yalnızca `transform` (veri sızıntısı önlenir)
* **Bölme:** kronolojik — eğitim 550 gözlem (15.01.2010–24.07.2020), test 180 gözlem (31.07.2020–05.01.2024)
* Makale veri setini rastgele karıştırarak böler; bu çalışma zaman sırasını korur (gerekçe: Tablo 4 ve raporun Bölüm 9.2'si).

## Kurulum

```bash
git clone https://github.com/Bnymn9681/fuzzy-logic-project
cd fuzzy-logic-project
pip install -r requirements.txt
```

Python 3.10+ gereklidir. Sürümler `requirements.txt` içinde sabitlenmiştir (torch 2.15.0, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.4.6, matplotlib 3.11.2). Tüm eğitimler CPU üzerinde çalışır.

## Nasıl Çalıştırılır

1. `src/1_data_preprocessing.ipynb` — ham CSV'leri işler, `data/processed_train.csv` ve `data/processed_test.csv` dosyalarını üretir.
2. `src/5_final_experiments.ipynb` — baseline modeller, ANFIS ablation (A0–A4, 5 tohum), protokol karşılaştırması ve tüm şekiller. Rapordaki her tablo ve şekil bu defterin tek çalıştırmasıyla `results/tables` ve `results/figures` altına üretilir.

`src/2_baseline_models.ipynb`, `src/3_anfis_reproduction.ipynb` ve `src/4_proposed_extension.ipynb` ilk aşama defterleridir; içerikleri 5 numaralı defterde birleştirilmiş ve düzeltilmiştir (adil başlatma, kararlı normalizasyon, RMSE ve süre kaydı). Rapordaki değerler 5 numaralı defterden alınmıştır.

## Deney Tasarımı (Ablation)

| Varyant | Üyelik fonksiyonu | Kural | Öğrenme oranı / epok |
| :--- | :--- | :--- | :--- |
| A0 (baseline) | Gaussian | 8 | 0,01 / 1500 |
| A1 | G-Bell | 8 | 0,01 / 1500 |
| A2 | Gaussian | 16 | 0,01 / 1500 |
| A3 | Gaussian | 8 | 0,005 / 2500 |
| A4 (önerilen) | G-Bell | 16 | 0,005 / 2500 |

Her varyant 5 farklı rastgele tohum (0–4) ile eğitilmiş, ortalama ± standart sapma raporlanmıştır.

## Deneysel Sonuçlar (Test Seti — 180 hafta)

| Model | MSE | RMSE | MAE | R² | Süre (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naif (eğitim ortalaması)** | **0,03444** | **0,18557** | **0,14648** | **−0,047** | 0,00 |
| Çoklu doğrusal regresyon | 0,04773 | 0,21847 | 0,15055 | −0,451 | 0,00 |
| YSA (15 nöron, tanh) | 0,05069 ± 0,01979 | 0,22211 ± 0,04124 | 0,16918 ± 0,03204 | −0,541 | 0,03 |
| A0: ANFIS (Gaussian, 8 kural) | 0,04905 ± 0,00442 | 0,22130 ± 0,00989 | 0,15615 ± 0,00489 | −0,491 | 1,85 |
| A1: yalnız G-Bell | 0,04807 ± 0,00495 | 0,21901 ± 0,01123 | 0,15693 ± 0,00562 | −0,461 | 3,02 |
| A2: yalnız 16 kural | 0,05102 ± 0,00595 | 0,22556 ± 0,01309 | 0,15939 ± 0,00564 | −0,551 | 2,64 |
| A3: yalnız lr 0,005 / 2500 epok | 0,04756 ± 0,00393 | 0,21794 ± 0,00907 | 0,15732 ± 0,00378 | −0,446 | 3,16 |
| A4: G-Bell + 16 kural + eğitim ayarı | 0,06545 ± 0,01289 | 0,25484 ± 0,02527 | 0,17248 ± 0,00953 | −0,989 | 7,40 |

Önerilen model (A4), baseline ANFIS'e göre test MSE değerini **%33,4 artırmıştır** (kötüleşme).

### Protokol karşılaştırması (aynı ham veri, aynı modeller)

| Protokol | Regresyon R² | YSA R² | ANFIS (A0) R² |
| :--- | :--- | :--- | :--- |
| P1: log-seviye, rastgele bölme (makale benzeri) | 0,882 | 0,788 | 0,971 |
| P2: log-seviye, kronolojik bölme | −0,204 | −0,850 | −1,367 |
| P3: log-fark, kronolojik (bu çalışma) | −0,451 | −0,541 | −0,491 |

Makalenin raporladığı değerler: regresyon 0,885, ANN 0,925, ANFIS-PSO 0,923.

## Bulgular

1. **Önerilen katkı başarımı artırmamıştır.** Üyelik fonksiyonu değişikliğinin tek başına etkisi (A1) %2 iyileşmedir ve tohumlar arası standart sapmanın altındadır. Üç değişiklik birleştiğinde (A4) belirgin kötüleşme görülür. Eğitim kayıp eğrilerinde A4 en düşük **eğitim** hatasına ulaşırken en yüksek **test** hatasını üretir — aşırı öğrenme. Öğrenilen G-Bell eğim parametresi ortalama b = 3,02'ye (en büyük 11,46) çıkarak üyelik fonksiyonlarını kare dalgaya yaklaştırmış, bulanık geçişi ortadan kaldırmıştır.
2. **Hiçbir model naif tahminciyi geçememiştir.** Tüm R² değerleri negatiftir. Regresyonun eğitim R² değeri yalnızca 0,272'dir ve girdilerin hedefle korelasyonu eğitim ile test dönemi arasında işaret değiştirmektedir (ör. Altın/TL: −0,185 → +0,098).
3. **Makaledeki yüksek R² değerleri büyük ölçüde değerlendirme protokolünden kaynaklanmaktadır.** Aynı ham veride makale benzeri protokol (P1) makaledekine yakın değerler üretirken, yalnızca bölme yöntemi kronolojiğe çevrildiğinde (P2) R² negatife düşmektedir.

## Sınırlılıklar

* Yalnızca üç makroekonomik girdi kullanılmıştır; enflasyon, merkez bankası faiz kararları ve kredi risk primi gibi değişkenler dışarıda kalmıştır.
* Meta-sezgisel eğitim (PSO/ACO) uygulanmamıştır.
* Doğrulama kümesi, erken durdurma veya ağırlık cezası gibi düzenlileştirme önlemleri kullanılmamıştır; A4'ün aşırı öğrenmesi bununla doğrudan ilişkilidir.
* Tek kronolojik bölme kullanılmış, kayan pencere (walk-forward) doğrulaması yapılmamıştır. Test dönemi (2020–2024) eğitim dönemine kıyasla farklı bir ekonomik rejimi kapsar.
* Varyantlar arası farklar için biçimsel hipotez testi uygulanmamıştır.
* Haftalık veri gün içi dinamikleri yakalayamaz.

## Üretken Yapay Zekâ Kullanımı

Projede Google Gemini (literatür özetleme, veri ön işleme, hata ayıklama, ANFIS'in tensör formülasyonu) ve Anthropic Claude (rapor incelemesi, teknik doğrulama, kodlardaki sayısal kararlılık ve deney tasarımı hatalarının tespiti, ablation ve protokol deneylerinin kurgulanması) yardımcı araç olarak kullanılmıştır. Tüm kod ve sonuçlar öğrenci tarafından çalıştırılarak doğrulanmıştır.

## Kaynaklar

[1]	H. Kazak, S. Kumar, M. A. Gündüz, A. T. Akcan, and H. H. Bilgiç, “Metaheuristic-optimized ANFIS and ANN models for stock price forecasting: Evidence from the Borsa Istanbul 100 index,” Discov. Artif. Intell., vol. 5, Art. no. 272, Oct. 2025, doi: 10.1007/s44163-025-00395-6.
[2]	J.-S. R. Jang, “ANFIS: Adaptive-network-based fuzzy inference system,” IEEE Trans. Syst., Man, Cybern., vol. 23, no. 3, pp. 665–685, May/Jun. 1993, doi: 10.1109/21.256541.
[3]	E. F. Fama, “Efficient capital markets: A review of theory and empirical work,” J. Finance, vol. 25, no. 2, pp. 383–417, 1970.
[4]	A. W. Lo, “The adaptive markets hypothesis: Market efficiency from an evolutionary perspective,” J. Portfolio Manag., vol. 30, pp. 15–29, 2004.
[5]	H. Gunduz, Y. Yaslan, and Z. Cataltepe, “Intraday prediction of Borsa Istanbul using convolutional neural networks and feature correlations,” Knowl.-Based Syst., vol. 137, pp. 138–148, 2017, doi: 10.1016/j.knosys.2017.09.023.
[6]	I. U. Armagan, “Price prediction of the Borsa Istanbul banks index with traditional methods and artificial neural networks,” Borsa Istanbul Rev., vol. 23, pp. S30–S39, 2023, doi: 10.1016/j.bir.2023.10.005.
[7]	E. Kaya, A. Kaya, and K. C. Baştemur, “Adaptive network-based fuzzy inference system training using nine different metaheuristic optimization algorithms for time-series analysis of Brent oil price and detailed performance analysis,” Symmetry, vol. 17, no. 5, Art. no. 786, 2025, doi: 10.3390/sym17050786.
[8]	R. M. Adnan Ikram, A. Jaafari, S. G. Milan, O. Kisi, S. Heddam, and M. Zounemat-Kermani, “Hybridized adaptive neuro-fuzzy inference system with metaheuristic algorithms for modeling monthly pan evaporation,” Water, vol. 14, no. 21, Art. no. 3549, 2022, doi: 10.3390/w14213549.
[9]	I. Ahmadianfar, S. Shirvani-Hosseini, J. He, A. Samadi-Koucheksaraee, and Z. M. Yaseen, “An improved adaptive neuro fuzzy inference system model using conjoined metaheuristic algorithms for electrical conductivity prediction,” Sci. Rep., vol. 12, Art. no. 4934, 2022, doi: 10.1038/s41598-022-08875-w.
[10]	R. Eberhart and J. Kennedy, “A new optimizer using particle swarm theory,” in Proc. 6th Int. Symp. Micro Machine and Human Science (MHS), 1995, pp. 39–43, doi: 10.1109/MHS.1995.494215.
[11]	M. Dorigo, M. Birattari, and T. Stützle, “Ant colony optimization: Artificial ants as a computational intelligence technique,” IEEE Comput. Intell. Mag., vol. 1, no. 4, pp. 28–39, Nov. 2006, doi: 10.1109/MCI.2006.329691.
[12]	T. Takagi and M. Sugeno, “Fuzzy identification of systems and its applications to modeling and control,” IEEE Trans. Syst., Man, Cybern., vol. SMC-15, no. 1, pp. 116–132, Jan./Feb. 1985, doi: 10.1109/TSMC.1985.6313399.

