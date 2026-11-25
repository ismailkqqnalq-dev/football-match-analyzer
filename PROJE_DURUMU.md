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

**İlk ölçüm (geniş/taktik kamera, ilk 120 sn):** Etiketler insan değil Claude
tarafından, sistem çıktısına bakılmadan, saniyede 2 kare (pas anlarında 0,1 sn) izlenerek
çıkarıldı: 22 pas, 5 top kaybı, 115 sn sahiplik. Kaba bir göstergedir.

| Ölçüt | Sonuç |
|---|---|
| Pas (±1 sn) | sistem 7, gerçek 22, eşleşen 4 → kesinlik %57, duyarlılık %18, F1 0,28 |
| Top kaybı (±1 sn) | sistem 9, gerçek 5, eşleşen 1 → kesinlik %11, duyarlılık %20, F1 0,14 |
| Sahiplik (saniye bazında) | sistemin takım verdiği saniyelerde uyum %62; etiketli saniyelerin %70'inde sistem bir takım veriyor |

Toleransı 2 sn'ye çıkarmak sonucu az değiştiriyor (pas F1 0,34), yani sorun zamanlama
değil: sistem pasların çoğunu kaçırıyor ve fazladan top kaybı yazıyor. Geniş kamerada
top 2-3 piksel; top tespiti kopunca pas zinciri kırılıp sahiplik el değiştirmiş
görünüyor. **Pas ve top kaybı sayıları şu an güvenilir değil**; top sahipliği kabaca
yön gösteriyor. Ölçüm README'deki komutla tekrarlanabilir.

## Dar C: Takım metrikleri
Takım biçimi metrikleri artık bir **kapsama etiketi** taşıyor: o takımdan karede
ortanca kaç oyuncu göründüğü ve buna göre high / medium / low düzeyi. TV yakın
çekiminde 3-4 oyuncuyla hesaplanan "takım genişliği" tüm takımı temsil etmez; etiket
bunu açıkça söyler. Gerçek veride geniş kamera "high" (ortanca 11 oyuncu), TV yayını
"medium" (ortanca 7-8) çıktı.

## Sırada ne var
1. TV için daha çok eğitim verisi ve model ince ayarı (A2'nin TV hedefi için).
2. Pas/top kaybı tespiti: geniş kamerada top takibini güçlendirmek (kısa kopmalarda
   topu ara değerle sürdürmek), sonra Faz V'yi insan etiketiyle ve bir TV klibinde
   tekrarlamak.
3. Takip iyileştirmesi: geniş kamerada 10 dakikada 439 iz, ortanca iz süresi 1,5 sn;
   oyuncu kimliği (Faz B: takım + forma numarası) bunun üzerine kurulacak.
4. Ofsayt sonrası iptal edilen şutların istatistikten düşülmesi (tasarımı hazır).

## Lisans notu
Tespit ve saha modelleri Ultralytics YOLO (AGPL-3.0) üzerine kurulu. Kapalı kaynak
ticari kullanım için ya Ultralytics ticari lisansı ya da Apache-2.0 lisanslı bir
alternatife geçiş gerekir; eğitim veri setlerinin lisansları da ayrıca kontrol
edilmeli.
