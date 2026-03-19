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

### 3. RETENTION SYSTEM CHECK (Her 4. heartbeat - ~2 saatte bir)
- [ ] Retention state dosyası var mı? (`retention_state.json`)
- [ ] Son 24 saat içinde daily learning çalışmış mı?
- [ ] Yeni memories var mı? (memory/*.md)
- [ ] Gerekirse daily learning tetikle (günde max 1 kez)

### 4. PROACTIVE MEMORY RECALL & FEEDBACK (Her 3. heartbeat)
- [ ] Önemli hatırlatmaları kontrol et
- [ ] Yaklaşan deadline'ları hatırlat
- [ ] Recent conversations'ı özetle
- [ ] **FEEDBACK CAPTURE (YENİ!):**
  - Son mesajlarda feedback keywords ara: "beğendim", "sevmedim", "iyi", "kötü", "doğru", "yanlış", "teşekkür", "kötü", "hata", "düzelt"
  - Positive feedback: "teşekkür", "beğendim", "mükemmel", "çok iyi" → memory'ye achievement/preference olarak ekle
  - Negative feedback: "yanlış", "hata", "kötü", "düzelt" → memory'ye lesson/decision olarak ekle
  - Suggestion: "belki", "ya da", "daha iyi olur", "değiştir" → memory'ye insight/project olarak ekle
  - Feedback'leri otomatik konsolidasyon kuyruğuna ekle (priority threshold: 0.4)

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
- **Search limit:** 3 memories per query
- **Similarity threshold:** 0.5
- **Cache TTL:** 1 hour
- **Auto-reindex:** Daily at 02:00 (cron job active)
- **Retention daily learning:** 23:55 (cron job)
- **Retention heartbeat check:** Her 4. heartbeat (~2 saat)

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
