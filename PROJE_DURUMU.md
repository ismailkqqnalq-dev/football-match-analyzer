# Proje Durumu (2026-11-26)

Bu belge, projenin teslim anındaki durumunu özetler: neyin çalıştığı, nasıl ölçüldüğü
ve bilinen sınırlar. Ölçümlerin hepsi gerçek maç görüntüsünde, elle etiketlenmiş
noktalara karşı yapıldı.

## Ne yapıyor
Maç videosundan (dosya ya da link) oyuncu, kaleci, hakem ve topu bulur; oyuncuları
kareler boyunca takip eder, takımlara ayırır, sahayı otomatik tanıyıp her şeyi saha
metresine çevirir. Bunun üzerinden top sahipliği, paslar, top kayıpları, takım
biçimi (genişlik, derinlik, savunma hattı) ve ısı haritaları üretir; CSV/SQLite ve
JSON olarak dışa aktarır.

## Faz A2: Otomatik saha tanıma (kalibrasyon)
Saha çizgilerini ve önemli noktaları (köşe, ceza sahası, orta yuvarlak) bir yapay
zekâ modeli bulur; tablo saha çizgilerine oturtulup doğrulanır, kamera hareketi
optik akışla izlenir. Yanlış tabloyu kabul etmek, tablo bulamamaktan daha zararlı
olduğu için birkaç güvenlik kontrolü var: çizgilere uymayan, tek bir çizgiye yığılan
ya da doğrulanamayan tablo reddedilir ve önceki tablo korunur.

| Ölçüt | Hedef | Geniş / taktik kamera (10 dk) | TV yayını (7,7 dk) |
|---|---|---|---|
| Etiketli noktalarda ortanca hata | geniş < 0,5 m, TV < 1 m | **0,28 m** (%90: 0,65 m) — geçti | 4,6 m — geçmedi (düzeltmeler öncesi 7,5 m) |
| Ana kamera karelerinde geçerli tablo | ≥ %90 | **%100** — geçti | %79 — geçmedi (öncesi %92; düşüş yanlış tabloların artık reddedilmesinden) |
| Duran nesne hızı (ortanca) | < 1 km/sa | **0,64 km/sa** — geçti (1 sn'lik parçaların %62'si < 1 km/sa) | 1,26 km/sa — geçmedi |

Ölçüm yöntemi: `utils/evaluate_homography.py` (etiketli noktalar), ana kamera kare
oranı ve duran nesne hızı için oturum betikleri (sahadaki sabit çim/çizgi noktaları
1 sn boyunca izlenip hızları ölçüldü).

**Bilinen sınır (TV):** Geniş kamerada hedefler tutuyor, TV yayınında tutmuyor. İki
sebep var: (1) model TV yakın çekimlerinde doğru tabloyu seyrek buluyor, aradaki uzun
kamera hareketlerinde optik akış zamanla kayıyor; (2) model bazen ceza sahasını yanlış
yere koyuyor. Kontroller bu yanlışların bir kısmını yakalıyor (ör. tüm çizgileri tek
çizgiye yığan tablolar), bir kısmını yakalayamıyor. TV ölçümü yalnızca 5 etiketli
kareye dayanıyor; sayılar kaba bir göstergedir. Kök sebep, modelin TV görüntüsüyle
yeterince eğitilmemiş olması; çare, başka maçlardan daha çok etiketli TV karesiyle
ince ayar. Geniş/sabit kamerada bu sorun görülmedi.

## Faz V: Pas ve top sahipliği doğrulaması
Araçlar hazır: tarayıcıda çalışan etiketleme aracı (`utils/label_events.html`) ve
karşılaştırma aracı (`utils/validate_events.py`, ±1 sn tolerans; kesinlik,
duyarlılık, F1 ve saniye bazında sahiplik uyumu). Kullanımı README'de.

**Ölçüm (geniş/taktik kamera, ilk 4 dakika):** Etiketler insan değil Claude
tarafından, sistem çıktısına bakılmadan, saniyede 2 kare (pas anlarında 0,1 sn) izlenerek
çıkarıldı. Kaba bir göstergedir. İlk 120 sn sorunu bulmak ve düzeltmek için kullanıldı;
120-240 sn düzeltme sırasında hiç bakılmamış **kontrol aralığıdır** (dürüst sonuç budur).

| Ölçüt (±1 sn) | 0-120 sn: ilk → #15 → #16 → #17 | 120-240 sn kontrol: ilk → #15 → #16 → #17 |
|---|---|---|
| Pas F1 | 0,28 → 0,53 → 0,56 → 0,65 (gerçek 22; yakalanan 4 → 9 → 11 → 13) | 0,30 → 0,36 → 0,39 → 0,68 (gerçek 34; yakalanan 6 → 8 → 9 → 19) |
| Pas kesinliği (#17) | %72 | %86 |
| Top kaybı F1 | 0,14 → 0,12 → 0,17 → 0,24 (gerçek 5) | 0 → 0 → 0,17 → 0,13 (gerçek 3, sistem 13) |
| Sahiplik uyumu | %62 → %62 → %81 → %82 | %72 → %72 → %81 → %86 |

Bulunan ve düzeltilen hatalar:
- **#15:** Top pas sırasında görünmez olunca Kalman tahmini pasörün yanında donuyordu;
  top yeniden görününce bu "imkânsız sıçrama" sayılıp pas eleniyordu. Ayrıca top
  satırlarının %21'i kale ağının arkasındaki iki sabit noktada sahte tespitti.
- **#16:** Takipçi aynı kimliği zamanla farklı oyunculara veriyor (ör. bir kimlik 10
  dakikanın yarısında A, yarısında B). Sahiplik takımı izin ömür boyu çoğunluğundan
  alındığı için City pasları "Tottenham'a top kaybı" sayılıyordu; artık her kontrol
  periyodunun kendi karelerinin çoğunluk takımı kullanılıyor. #16'da kontrol aralığının
  yalnız sayısal özetine bakıldı (kaçırma sınıfları), düzeltme 0-120 sn'den türetildi.

- **#17:** Top dedektörü gerçek topla 10-25 m ötedeki sahte bir nokta arasında gidip
  geliyordu (ör. 0,22 sn'de 21 m = 96 m/sn); takipçi de aradaki boşluğu ara değerle
  dolduruyordu. Artık iki uyumlu top izi parçası arasına 8 m'den büyük sıçramayla
  girilen, en fazla 1 sn süren parça sahte sayılıp atılıyor; imkânsız sıçramayı örten
  ara değer kareleri de atılıyor. Bu karar sadece 0-120 sn'ye bakılarak verildi.

Kontrol aralığında artık sistemin dediği pasların %86'sı doğru ve gerçek pasların
yarısından fazlası (%56) yakalanıyor; top sahipliği %86 uyumlu. **Top kaybı hâlâ
güvenilir değil**: sistem 4 dakikada 3 gerçek kayba karşı 13 kayıp yazıyor. Bu sayılar
Claude etiketiyle 4 dakikalık tek bir klipte ölçüldü; insan etiketiyle ve başka bir
maçta doğrulanmadan kesin kabul edilmemeli. Ölçüm README'deki komutla tekrarlanabilir.

## Dar C: Takım metrikleri
Takım biçimi metrikleri artık bir **kapsama etiketi** taşıyor: o takımdan karede
ortanca kaç oyuncu göründüğü ve buna göre high / medium / low düzeyi. TV yakın
çekiminde 3-4 oyuncuyla hesaplanan "takım genişliği" tüm takımı temsil etmez; etiket
bunu açıkça söyler. Gerçek veride geniş kamera "high" (ortanca 11 oyuncu), TV yayını
"medium" (ortanca 7-8) çıktı.

## Sırada ne var
1. TV için daha çok eğitim verisi ve model ince ayarı (A2'nin TV hedefi için).
2. Top kaybı: fazladan yazılan kayıpların sebebini incelemek (aday: markajdaki rakibin
   topa en yakın oyuncu sayılması). ML analizöründe (`ml_analyzer.py`) #15-#17
   düzeltmeleri yok. Sonra Faz V'yi insan etiketiyle ve bir TV klibinde tekrarlamak.
3. Takip iyileştirmesi: geniş kamerada 10 dakikada 439 iz, ortanca iz süresi 1,5 sn;
   oyuncu kimliği (Faz B: takım + forma numarası) bunun üzerine kurulacak.
4. Ofsayt sonrası iptal edilen şutların istatistikten düşülmesi (tasarımı hazır).

## Lisans notu
Tespit ve saha modelleri Ultralytics YOLO (AGPL-3.0) üzerine kurulu. Kapalı kaynak
ticari kullanım için ya Ultralytics ticari lisansı ya da Apache-2.0 lisanslı bir
alternatife geçiş gerekir; eğitim veri setlerinin lisansları da ayrıca kontrol
edilmeli.
