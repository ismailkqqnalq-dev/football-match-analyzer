# Proje Kuralları — football-match-analyzer

## 1. Git Kimliği ve Katkıcı Kuralı
- Tüm Git commit'lerinde yalnızca kullanıcı kimliği (`ismail kanal <ismailkqqnalq@gmail.com>`) kullanılmalıdır.
- Commit mesajlarına, dosya başlıklarına veya meta verilere kesinlikle `Co-Authored-By:` ya da herhangi bir yapay zekâ / asistan katkıcı imzası eklenmemelidir.

## 2. İletişim ve Dokümantasyon Dili
- Kullanıcıya verilen yanıtlar, üretilen analiz raporları (`ml_analysis_report.md`, `README.md` güncellemeleri vb.) ve CLI özetleri repo diliyle uyumlu şekilde Türkçe olmalıdır; Python modül, sınıf, fonksiyon ve sütun adları İngilizce tutulmalıdır.
- Teknik veya makine öğrenmesi terimleri açıklandığında jargona boğmadan, mümkün olduğunca futbol örnekleri ve günlük benzetmelerle sade anlatım tercih edilmelidir.

## 3. Veri Güvenliği ve Test Ortamı
- Mevcut örnek veri dosyaları (`data/exports/tracking_data.csv` ve `data/exports/tracking_data.db`) test veya doğrulama çalıştırmaları sırasında asla üzerine yazılmamalıdır; doğrulama çıktıları `tmp_path` veya `$env:TEMP` altına yönlendirilmelidir.
- Test paketi repo kökündeki `pytest.ini` (`testpaths = tests`) ile çalıştırılmalıdır. Windows sandbox ortamında `ultralytics` içe aktarma izin hatasını önlemek için testten önce `$env:YOLO_CONFIG_DIR="$env:TEMP\ultralytics_cfg"` tanımlanmalıdır.
