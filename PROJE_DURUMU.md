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

**Ölçüm (geniş/taktik kamera, ilk 4 dakika):** Etiketler insan değil bir yapay zekâ
tarafından, sistem çıktısına bakılmadan, saniyede 2 kare (pas anlarında 0,1 sn) izlenerek
çıkarıldı. Kaba bir göstergedir. İlk 120 sn sorunu bulmak ve düzeltmek için kullanıldı;
120-240 sn düzeltme sırasında hiç bakılmamış **kontrol aralığıdır** (dürüst sonuç budur).

| Ölçüt (±1 sn) | 0-120 sn: ilk → #15 → #16 → #17 → #18 | 120-240 sn: ilk → #15 → #16 → #17 → #18 |
|---|---|---|
| Pas F1 | 0,28 → 0,53 → 0,56 → 0,65 → 0,71 (gerçek 22; yakalanan 4 → 9 → 11 → 13 → 16) | 0,30 → 0,36 → 0,39 → 0,68 → 0,69 (gerçek 34; yakalanan 6 → 8 → 9 → 19 → 20) |
| Pas kesinliği (son) | %70 | %83 |
| Top kaybı F1 | 0,14 → 0,12 → 0,17 → 0,24 → 0,40 (gerçek 5, sistem 5) | 0 → 0 → 0,17 → 0,13 → 0,25 (gerçek 3, sistem 5) |
| Sahiplik uyumu | %62 → %62 → %81 → %82 → %82 | %72 → %72 → %81 → %86 → %86 |

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
- **#18:** Aynı takımın iki kontrol periyodu arasına sıkışmış, en fazla 0,6 sn süren tek
  rakip periyodu (top pas sırasında rakibin dibinden geçiyor) artık "kayıp + geri
  kazanım" diye iki sahte top kaybı üretmiyor. **Bu desen 120-240 sn'deki olaylara da
  bakılarak bulundu; #18 için 120-240 sn temiz kontrol aralığı değildir.**
- **#19:** #18'in kuralı, sonraki periyot saniyeler sonra gelse de rakip periyodunu
  siliyordu (gerçek bir kayıp + geri kazanımı yutuyordu). Artık yalnızca önceki ve sonraki
  periyot arası bir pas süresini (`PASS_MAX_GAP_SEC`, 4 sn) aşmıyorsa uygulanıyor.
  120-240 sn'de pas F1 0,69 → 0,71, top kaybı F1 0,25 → 0,20 (sistem 5 → 7 kayıp, doğru
  1); 0-120 sn değişmedi. Aynı PR'da düzeltmeler ML analizörüne (`ml_analyzer.py`) de
  taşındı: saha dışı top, sahte top sıçraması ve kısa rakip periyodu. #15 ortak
  fonksiyonda olduğu için zaten geçerliydi; #16'ya gerek yok, çünkü ML takımı izin ömür
  boyu çoğunluğundan değil, her karenin kendisinden alıyor. ML'nin vekil pas olaylarında
  pas F1 0-120 sn'de 0,41 → 0,71, 120-240 sn'de 0,33 → 0,64.

Son durumda (iki aralık birlikte, 4 dk) sistemin dediği pasların ~%77'si doğru, gerçek
pasların ~%64'ü yakalanıyor; top sahipliği ~%84 uyumlu; top kaybında sistem 8 gerçek
kayba karşı 10 kayıp yazıyor, 3'ü doğru (hâlâ zayıf). Temiz kontrol sonucu #17'ye kadar
olan sütundur. Bu sayılar yapay zekâ etiketiyle tek bir klipte ölçüldü; insan etiketiyle ve
başka bir maçta doğrulanmadan kesin kabul edilmemeli. Ölçüm README'deki komutla
tekrarlanabilir.

## Dar C: Takım metrikleri
Takım biçimi metrikleri artık bir **kapsama etiketi** taşıyor: o takımdan karede
ortanca kaç oyuncu göründüğü ve buna göre high / medium / low düzeyi. TV yakın
çekiminde 3-4 oyuncuyla hesaplanan "takım genişliği" tüm takımı temsil etmez; etiket
bunu açıkça söyler. Gerçek veride geniş kamera "high" (ortanca 11 oyuncu), TV yayını
"medium" (ortanca 7-8) çıktı.

## Sırada ne var
1. TV için daha çok eğitim verisi ve model ince ayarı (A2'nin TV hedefi için).
2. Faz V'yi insan etiketiyle, bu klibin görülmemiş bir bölümünde ve bir TV klibinde
   tekrarlamak (#18'in gerçek etkisi için).
3. Takip iyileştirmesi: geniş kamerada 10 dakikada 439 iz, ortanca iz süresi 1,5 sn;
   oyuncu kimliği (Faz B: takım + forma numarası) bunun üzerine kurulacak.
4. Ofsayt sonrası iptal edilen şutların istatistikten düşülmesi (tasarımı hazır).

## Lisans notu
Tespit ve saha modelleri Ultralytics YOLO (AGPL-3.0) üzerine kurulu. Kapalı kaynak
ticari kullanım için ya Ultralytics ticari lisansı ya da Apache-2.0 lisanslı bir
alternatife geçiş gerekir. Repoda henüz `LICENSE` dosyası yok.

Ultralytics dışındaki bağımlılıklar ticari kullanıma izin veriyor (2026-11-27 kontrolü):
OpenCV ve python-multipart Apache-2.0; PyTorch, NumPy, pandas, SciPy, scikit-learn,
httpx ve uvicorn BSD; deep-sort-realtime, FastAPI, PyJWT ve PyYAML MIT; tqdm MPL-2.0/MIT;
yt-dlp Unlicense; matplotlib PSF.

Eğitim verileri:

| Veri | Kullanıldığı model | Lisans | Not |
|---|---|---|---|
| Roboflow `roboflow-jvuqo/football-players-detection-3zvbc` (v20, 372 görüntü) | Oyuncu/top tespiti | CC BY 4.0 | Ticari kullanım serbest, kaynak gösterilmeli |
| Roboflow `roboflow-jvuqo/football-field-detection-f07vi` (v18, 317 görüntü) | Saha noktaları | CC BY 4.0 | Ticari kullanım serbest, kaynak gösterilmeli |
| Kendi etiketlediğimiz TV kareleri (442 görüntü) | Saha noktaları (TV ince ayarı) | Etiketler bizim | Görüntüler TV yayınından; yayın hakları yayıncıda. Ticari modelde kullanmadan önce hukuki görüş alınmalı veya yerine lisanslı/kendi çekimimiz kullanılmalı |
