# Hacı Cognitive System

**Transformer + Retention Layer + OpenClaw Retention System + HaciCognitiveNet**

Bu proje, [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer)'ı temel alır ve üzerine **Retention Katmanı** ekler. İlham kaynağı:

> *Attention Is All You Need Until You Need Retention*  
> arXiv:2501.09166 (2025)

Retention Katmanı, standart self-attention mekanizmasını, **uzun vadeli bağımlılıkları** yakalayan ve sabit bağlam penceresinin ötesinde **kalıcı bellek** sağlayan özyineli bir bellek mekanizması ile değiştirir veya tamamlar.

Ayrıca bu depo, **OpenClaw Retention System** ve **HaciCognitiveNet** katmanlı bilişsel mimariyi içerir.

---

## 🔑 Temel Bileşenler

### Retention Katmanı (`MultiScaleRetention`)

Her başlık için **çok-ölçekli üstel bozunma** uygular, farklı bellek ufuklarını yakalar.

Özyineli durum güncellemesi:

$$S_t = \alpha \cdot S_{t-1} + h_t$$

Kapı mekanizması ile bağlam harmanlama:

$$y_t = \sigma(W_g x_t) \odot \tilde{S}_t + (1 - \sigma(W_g x_t)) \odot x_t$$

Nedensel yapıya sahiptir, opsiyonel padding maski destekler.

### Konfigürasyonlu Dikkat Mekanizması

Standart **Multi-Head Attention (MHA)** veya **Retention** arasında seçim:

```toml
[model]
attention_kind = "retention"   # "mha" veya "retention"
```

### Encoder / Decoder Entegrasyonu

- Encoder self-attention: MHA veya Retention
- Decoder self-attention: MHA veya Retention  
- Decoder cross-attention: kararlılık için MHA kalır

---

## 🧠 HaciCognitiveNet – Katmanlı Bilişsel Mimari

Retention çekirdeği üzerine inşa edilen tam bir bilişsel sistem: **dünya modelleri**, **kişilik gelişimi**, **sosyal zeka** ve **olumsuz sonuç öğrenme**.

### Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────┐
│                 SEVİYE 3: META-ÖĞRENME                      │
│  Meta-Öğrenici │ Dünya Modeli │ Kendiliğinden Evrim        │
├─────────────────────────────────────────────────────────────┤
│                  SEVİYE 2: AKTİF ÖĞRENME                    │
│  Merak Motoru │ Tahminsel Kodlama │ Aktif Zamanlayıcı      │
│  Kendi Kendine Denetimli Döngü (rüya döngüleri)            │
├─────────────────────────────────────────────────────────────┤
│                  SEVİYE 1: ÇEKİRDEK BİLİŞ                   │
│  Dünya Modeli (512-dim) │ Kişilik (1024-dim) │ Retention   │
│  Üst-Biliş │ Duygusal Durum                                 │
├─────────────────────────────────────────────────────────────┤
│                🤝 SOSYAL ZEKA                               │
│  Duygusal Zeka │ Kişilik Gelişimi │ Konuşma Zekası         │
│  🚫 Olumsuz Sonuç Öğrenici (7 sinyal kategorisi)           │
└─────────────────────────────────────────────────────────────┘
```

### Seviye 1: Çekirdek Biliş (`cognitive_net.py`)

- **DünyaModeli** (512-dim): Fiziksel ve sosyal dünyanın kodlanması
- **KişilikVektörleri** (1024-dim): 8 başlık × 128 boyut (sıcaklık, empati, mizah, kararlılık, sadakat, oyunculuk, bilgelik, yaramazlık)
- **ÜstBiliş**: Öğrenme verimliliği, bilgi boşlukları, güven kalibrasyonu öz izlemesi
- **DuygusalDurum**: Ruh hali, merak, güven, enerji takibi
- **RetentionÇekirdeği** (4-katmanlı): Öncelik tabanlı geri çağrılı uzun vadeli bellek

### Seviye 2: Aktif Öğrenme

- **MerakMotoru** (`curiosity_engine.py`): 28 ilgi konusu, merak puanlama, keşif
- **TahminselKodlama** (`predictive_coding.py`): Hata tahmini, şaşkınlık tespiti
- **AktifÖğrenmeZamanlayıcısı** (`active_learning_scheduler.py`): Öncelik tabanlı planlama
- **KendiKendineDenetimliDöngü** (`self_supervised_loop.py`): Gece rüya döngüleri, kalıp keşfi

### Seviye 3: Meta-Öğrenme

- **MetaÖğrenici** (`meta_learner.py`): Strateji seçimi, hiperparametre optimizasyonu
- **DünyaModeli** (`world_model.py`): Hayal gücü, simülasyon, gelecek tahmini
- **KendiliğindenEvrim** (`self_evolution.py`): Mimari mutasyon, A/B testi
- **DuyusalArayüz** (`sensory_interface.py`): Genişletilebilir sensör/aktüatör çerçevesi

### Sosyal Zeka (`social_trainer.py`)

- **KişilikGelişimi**: 8 özellik — sıcaklık, empati, mizah stili, kararlılık, sadakat, oyunculuk, bilgelik, yaramazlık
- **KonuşmaZekası**: Stil analizi, etkileşim tipi tespiti, tutarlılık takibi
- **Gelişim Aşamaları**: bebeklik → çocukluk → ergenlik → yetişkinlik → ustalık
- **OlumsuzSonuçÖğrenici** (`negative_learner.py`): 7 sinyal kategorisi, otomatik tespit, kalıcı "asla yapma" kuralları

### Entegrasyon Katmanı 🆕

- **Bilişselİzleyici** (`cognitive_watcher.py`): Gerçek zamanlı mesaj analizi (<400ms), duygu tespiti, etkileşim sınıflandırma, otomatik kişilik güncelleme
- **DünyaModeliV2** (`world_model_v2.py`): MEMORY.md'den bilgi grafiği, konu kümeleme, zaman çizelgesi, ilişki haritalama (1554 varlık, 5112 ilişki)
- **RüyaZamanlayıcısı** (`dream_scheduler.py`): Otomatik gece 03:00 rüya döngüleri, durum konsolidasyonu, rüya kaydı
- **HaftalıkRapor** (`weekly_report.py`): Kişilik gelişimi, öğrenme özetleri, sosyal zeka istatistikleri
- **BilişselEntegratör** (`cognitive_integrator.py`): Tüm alt sistemleri birleştiren unified giriş noktası

---

## 🚀 Kullanım

### HaciCognitiveNet CLI

```bash
# Eğitim
python haci_cognitive/main.py train

# Rüya döngüsü
python haci_cognitive/main.py dream

# Sosyal zeka raporu
python haci_cognitive/main.py social

# Kişilik gelişimi
python haci_cognitive/main.py personality

# Etkileşim kaydetme
python haci_cognitive/main.py interact
```

### Entegrasyon Katmanı

```bash
# Alt sistemleri başlat
python haci_cognitive/cognitive_integrator.py init

# Mesaj işle
python haci_cognitive/cognitive_integrator.py process "Merhaba Başkan"

# Sistem durumu
python haci_cognitive/cognitive_integrator.py status

# Haftalık rapor
python haci_cognitive/cognitive_integrator.py report

# Rüya döngüsü tetikle
python haci_cognitive/cognitive_integrator.py dream

# Dünya modelini doldur
python haci_cognitive/cognitive_integrator.py populate
```

### Eğitim Sonuçları

| Metrik | Değer |
|--------|-------|
| Başlangıç Loss | 4.82 |
| Final Loss | 0.077 |
| İyileşme | %98.4 |
| Epoch | 39 (erken durdurma) |
| Checkpoint | `cognitive_epoch_0039.pt` |

---

## 🧩 OpenClaw Retention System

**"Çocuk gibi büyüme" felsefesi ile çalışan otomatik bellek konsolidasyonu ve kişilik öğrenme sistemi.**

### Bileşenler

- ✅ **Otomatik Bellek Konsolidasyonu**: Önemli günlük bellekleri uzun vadeli belleğe aktarma
- ✅ **Retention Günlük Öğrenme**: Yeni belleklerle kişilik durumunu güncelleme
- ✅ **FAISS Semantik Arama**: Sentence-transformers + FAISS ile hızlı bellek geri çağırma
- ✅ **MultiScaleRetention Katmanı**: Bağlam sıkıştırma ile %56.8 token tasarrufu
- ✅ **Cron Job Otomasyonu**: Günlük konsolidasyon, öğrenme, yeniden indeksleme, otomatik ayarlama
- ✅ **Kalp Atışı Entegrasyonu**: Her 4. kalp atışında sağlık kontrolü ve aktif öğrenme
- ✅ **OpenClaw Eklentisi**: Yerel OpenClaw bellek yuvası entegrasyonu
- ✅ **Dönüştürücü Önem Puanlaması**: Bellek önemini tahminleyen transformer modeli
- ✅ **Kişilik Durumu Öğrenme**: Bellek kalıplarından uyarlanabilir kişilik gömmeleri

### Performans

| Metrik | Değer |
|--------|-------|
| Token Tasarrufu | %56.8 (7741 token) |
| Hız Artışı | 2.3× (FAISS vs lineer arama) |
| Bellek Türleri | 6 (karar, başarı, ders, tercih, proje, hatırlatıcı) |
| Geri Çağırma Doğruluğu | ~%85 (semantik benzerlik) |
| Sorgu Süresi | 0.58ms (ortalama) |

### Gereksinimler

```bash
pip install torch sentence-transformers faiss-cpu
# veya GPU için
pip install torch sentence-transformers faiss-gpu
```

### OpenClaw Eklentisi

`openclaw.json` dosyasına ekleyin:

```json
"plugins": {
  "load": {
    "paths": ["path/to/haci-memory-plugin"]
  },
  "slots": {
    "memory": "haci-memory"
  },
  "entries": {
    "haci-memory": {
      "enabled": true,
      "config": {
        "embedding": { "model": "all-MiniLM-L6-v2" },
        "faissIndexPath": "~/.openclaw/memory/faiss.index",
        "memoryPath": "~/.openclaw/workspace",
        "autoRecall": true
      }
    }
  }
}
```

### Cron Job'lar

```bash
# Bellek konsolidasyonu (her gün 23:55)
55 23 * * * python3 /path/to/run_consolidation.py

# Retention günlük öğrenme (her gün 23:55)
55 23 * * * python3 /path/to/retention_daily.py

# FAISS yeniden indeksleme (her gün 02:00)
0 2 * * * openclaw haci-memory rebuild
```

---

## ⚡️ Neden Retention?

Attention güçlüdür ancak sekans uzunluğuna göre kuadratik ($O(L^2 \cdot d)$) karmaşıklığa sahiptir ve bağlam penceresi ile sınırlıdır. Retention sunar:

- **Doğrusal zaman özyinelemesi**: $O(L \cdot d)$
- **Çıkarımda token başına $O(1)$ bellek**: akış dostu
- **Uzun vadeli kalıcılık**: çapraz-sekans bellek
- **Çok-ölçekli kernel'ler**: çeşitli zaman ufuklarını yakalama

### Kaynakça

- [RetNet: Retentive Network](https://arxiv.org/abs/2307.08621) — Sun vd., 2023
- *Attention Is All You Need Until You Need Retention* — arXiv:2501.09166, 2025

---

## 📂 Proje Yapısı

```
arch/
  attentions/
    multi_head_attention.py
    retention.py           # Retention katmanı
    __init__.py            # fabrika: make_attention()
  encoder/
    encoder_block.py       # retention desteği
  decoder/
    decoder_block.py       # retention desteği (self-attn)
tests/
  test_retention.py        # retention katman testleri

# OpenClaw Retention System
memory_consolidator.py     # Otomatik bellek konsolidasyonu
retention_daily.py         # Günlük öğrenme ve kişilik güncellemeleri
integrate_transformer.py   # Önem puanlaması için transformer entegrasyonu
simple_memory_transformer.py
train_memory.py            # Bellek gömme eğitim hattı
memory_dataset.py          # Bellek gömme veri seti

# HaciCognitiveNet
haci_cognitive/
  cognitive_net.py            # Seviye 1: Dünya Modeli, Kişilik, ÜstBiliş, Retention
  cognitive_state_manager.py  # Durum yönetimi
  cognitive_trainer.py        # Eğitim hattı
  dreaming_loop.py            # Gece rüya döngüleri
  curiosity_engine.py         # Seviye 2: Merak güdümlü keşif
  predictive_coding.py        # Hata tahmini ve şaşkınlık tespiti
  active_learning_scheduler.py # Öncelik tabanlı öğrenme planlaması
  self_supervised_loop.py     # Kendi kendine denetimli öğrenme
  meta_learner.py             # Seviye 3: Strateji seçimi
  world_model.py              # Dünya simülasyonu ve hayal gücü
  self_evolution.py           # Mimari mutasyon
  sensory_interface.py        # Duyusal arayüz çerçevesi
  social_trainer.py           # Sosyal Zeka: Duygusal zeka, kişilik gelişimi
  negative_learner.py         # Olumsuz sonuç öğrenme (7 sinyal kategorisi)
  world_model_v2.py           # Bilgi grafiği ve varlık çıkarımı
  cognitive_watcher.py        # Gerçek zamanlı mesaj analizi
  dream_scheduler.py          # Otomatik rüya zamanlayıcısı
  weekly_report.py            # Haftalık gelişim raporu
  cognitive_integrator.py     # Unified entegratör
  extract_conversations.py    # Konuşma verisi çıkarımı
  main.py                     # CLI arayüzü
```

---

## 📜 Lisans

Bu depo, orijinal deponun lisansını devam ettirir. Bakınız: `LICENSE` in [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer).

---

## 🙏 Teşekkürler

- Orijinal Transformer implementasyonu: [MayukhSobo/Transformer](https://github.com/MayukhSobo/Transformer)
- Retention konseptleri: Sun vd., *Retentive Network: A Successor to Transformer for Large Language Models*, 2023
- OpenClaw Retention System: OpenClaw AI Assistant için geliştirildi (https://openclaw.ai)
