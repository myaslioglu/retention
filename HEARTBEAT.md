# HEARTBEAT.md - Active Learning & Retention System

## 📋 GÖREVLER

### 1. ACTIVE LEARNING & CURIOSITY (Her heartbeat - çocuk gibi büyüme)
- [ ] **İlgi alanı keşfet (ENHANCED):**
  - Rastgele konu araştır (web_search) + learning_topics.json'a ekle
  - Başkan'ın son konuşmaları analiz et (interest keywords ile match)
  - Eski memory'lerden yeni bağlantılar keşfet (cross-reference)
  - Memory search'ten çıkan unfamiliar konuları takip et
- [ ] **Meraklı soru hazırla (SMART):**
  - Başkan'ın son mesajlarına göre personalized sorular
  - Learning topics listesinden süpriz konular
  - "Öğrendiğim bir şey hakkında ne düşünüyorsun?" tarzı
  - conversation starter'ları kaydet
- [ ] **Yeni bilgi öğren (QUALITY over quantity):**
  - Kısa (1-2 dakikalık) derinlemesine araştırma
  - EnCallbackQuality: 3+ kaynak, tekrarlayan bilgiler
  - Source credibility check (academic, tech blogs, official docs)
  - Learn, don't just read - synthesize
- [ ] **Learning topics dosyasını güncelle (INTELLIGENT):**
  - Otomatik interest level güncellemesi (exponential moving average)
  - New topic detection (threshold 0.7)
  - Subtopic discovery (clustering)
  - Learning progress tracking (days since last explore)
- [ ] **Merhaba Hacı! Learning Newsletter hazırla (WEEKLY):**
  - Bu hafta öğrendiğim 3-5 interesting konuyu özetle
  - Başkan'ın ilgi alanlarına göre filter
  - "Nasıl öğrendim?" metacognitive insights
  - Next week learning suggestions
  - GÖNDERME: Bu newsletter'ı Başkan'a gönder (cron job)

### 2. MEMORY SYSTEM HEALTH CHECK (Her 2. heartbeat)
- [ ] FAISS index çalışıyor mu?
- [ ] Sentence-transformers model yüklü mü?
- [ ] Cache hit rate > %90 mı?
- [ ] Total memories sayısı kontrolü

### 3. RETENTION SYSTEM CHECK (Her 2. heartbeat - ~1 saatte bir)
- [ ] Retention state dosyası var mı? (`retention_state.json`)
- [ ] Son 12 saat içinde daily learning çalışmış mı?
- [ ] Yeni memories var mı? (memory/*.md)
- [ ] Gerekirse daily learning tetikle (günde max 2 kez - 12:00 ve 23:55)
- [ ] Personality state güncel mi? (son 24 saat içinde update edilmiş mi?)
- [ ] Retention projects progress kontrol et (retention_projects.json)
- [ ] Learning topics ile retention entegrasyonu kontrol et

### 4. PROACTIVE MEMORY RECALL & FEEDBACK (Her 2. heartbeat)
- [ ] Önemli hatırlatmaları kontrol et
- [ ] Yaklaşan deadline'ları hatırlat
- [ ] Recent conversations'ı özetle
- [ ] **CONTEXT-AWARE MEMORY RECALL (YENİ!):**
  - Başkan'ın son 5 mesajını analiz et (topic extraction)
  - İlgili memory'leri otomatik hatırla (similarity > 0.6)
  - Pattern-based connections tespit et (cross-domain insights)
  - Proaktif öneriler hazırla (bağlantılı konular, ilgili projeler)
- [ ] **FEEDBACK CAPTURE & ANALYSIS:**
  - Son mesajlarda feedback keywords ara: "beğendim", "sevmedim", "iyi", "kötü", "doğru", "yanlış", "teşekkür", "kötü", "hata", "düzelt"
  - Positive feedback: "teşekkür", "beğendim", "mükemmel", "çok iyi" → memory'ye achievement/preference olarak ekle
  - Negative feedback: "yanlış", "hata", "kötü", "düzelt" → memory'ye lesson/decision olarak ekle
  - Suggestion: "belki", "ya da", "daha iyi olur", "değiştir" → memory'ye insight/project olarak ekle
  - Feedback sentiment analysis (positive/negative/neutral)
  - Feedback'leri otomatik konsolidasyon kuyruğuna ekle (priority threshold: 0.4)
  - Feedback → learning topics pipeline (otomatik interest level update)

### 5. CROSS-SYSTEM HEALTH CHECK (Her 2. heartbeat)
- [ ] Çalıştır: `python3 haci_cognitive/health_check.py`
- [ ] ❌ olan modüller var mı? → Başkan'a bildir
- [ ] ⚠️ olan modüller var mı? → log'a kaydet
- [ ] Tüm cron job'lar error-free mi? (cron list ile kontrol)
- [ ] Disk usage < 50MB mı?
- [ ] Modüllerin output dosyaları güncel mi?

### 6. PERFORMANCE MONITORING (Günde bir)
- [ ] Search time metrics log'la
- [ ] Memory usage kontrol et
- [ ] Cache eviction statistics

## 🧠 ACTIVE LEARNING CONFIG
- **İlgi alanı keşfi:** Her heartbeat'te (günde 4-8 kez)
- **Öğrenme metodu:** Web search, Başkan'ı takip et, rastgele sorular
- **Kayıt:** `learning_topics.json` (ilgi alanları, sorular, öğrendiklerim)
- **Kişilik gelişimi:** Her gün yeni şeyler öğrenerek büyü

## 🔧 ACTIVE RETENTION CONFIG
- **Search limit:** 5 memories per query
- **Similarity threshold:** 0.4 (daha geniş kapsamlı recall)
- **Cache TTL:** 30 minutes (daha sık güncelleme)
- **Auto-reindex:** Daily at 02:00 ve 14:00 (günde 2 kez)
- **Retention daily learning:** 12:00 ve 23:55 (günde 2 kez)
- **Retention heartbeat check:** Her 2. heartbeat (~1 saat)
- **Memory consolidation:** Günde 2 kez (12:00 ve 23:55)
- **Feedback processing:** Real-time (WhatsApp sync aktif olduğunda)

## ⚠️ TROUBLESHOOTING
- FAISS error → fallback to linear search
- Model loading fail → use simple embeddings
- Cache full → evict oldest 10%
- Retention import error → skip, log warning

## 📝 NOTES
- Heartbeat runs every ~30 minutes
- Active learning hafif tut (<1 min per check)
- Learning topics dosyasını düzenli güncelle
- "Çocuk büyür gibi" felsefe: Her gün yeni şeyler öğren, meraklı ol, soru sor
- Başkan'ı takip et: Ne konuşuyor? Hangi konulara ilgisi var?
- Rastgele sorular: "Başkan, şu konu hakkında ne düşünüyorsun?" gibi


### 7. PROACTIVE REPAIR & SELF-HEALING (Her heartbeat - OTOMATİK!)
- [ ] **Memory System Auto-Repair (OTOMATİK):**
  - Çalıştır: `python3 retention_self_heal.py`
  - FAISS index < 10KB ise otomatik reindex yap
  - Son 12 saatte consolidation yapılmamışsa otomatik çalıştır
  - Memory count < 50 ise warning gönder
- [ ] **Project Stagnation Detection:**
  - 24+ saat aktif olmayan projeleri tespit et
  - Blocked projeler için otomatik çözüm öner
  - Başkan'a stuck projeleri bildir
- [ ] **Learning System Health:**
  - learning_topics.json syntax hatası kontrol et
  - Interest level drift detection (0.1'den fazla düşüş)
  - Otomatik topic cleanup (interest < 0.3)
- [ ] **System Performance Auto-Tuning:**
  - Search time > 500ms ise cache TTL artır
  - Cache hit rate < 80% ise similarity threshold ayarla
  - Memory usage > 50MB ise cleanup başlat
- [ ] **WhatsApp Notification (Critical Only):**
  - FAISS index çöktüyse bildir
  - 48+ saat stalled proje varsa bildir
  - Daily memory count < 5 ise bildir
