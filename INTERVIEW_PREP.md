# 🧠 SynapseAI: AI & NLP Engineer Mülakat Hazırlık Dosyası

Bu dosya, **SynapseAI** projesinin iş ilanı gereksinimleriyle olan teknik eşleşmesini ve mülakat sırasında sorulabilecek kritik sorulara yönelik hazırlık notlarını içerir.

---

## 🚀 Proje Özeti
**SynapseAI**, yerel LLM (Large Language Model) ve Vektör Veri Tabanı kullanarak döküman analizi, özetleme ve karar destek hizmeti sunan bir **Multi-Agent RAG (Retrieval-Augmented Generation)** sistemidir.

---

## 🎯 Gereksinim Eşleşme Analizi

### 1. Retrieval-Augmented Generation (RAG) Mimarisi
*   **Kullanılan Teknolojiler:** `ChromaDB`, `nomic-embed-text`, `FastAPI`.
*   **Teknik Derinlik:**
    *   **Vektör Depolama:** HNSW (Hierarchical Navigable Small World) indeksi ve Cosine Similarity kullanarak yüksek performanslı semantik arama gerçekleştirildi.
    *   **Akıllı Parçalama (Semantic Chunking):** Standart `CharacterSplitter` yerine, cümle bütünlüğünü koruyan ve kod bloklarını (Python/JS) yapısal olarak algılayan `SmartTextChunker` sınıfı geliştirildi.
    *   **Hibrit Arama:** Vektör aramasına ek olarak, dokümandan çıkarılan anahtar kelimelerle (Metadata Filtering) arama sonuçları optimize edildi.
*   **Mülakat Notu:** "Bağlam kaybını önlemek için %20 örtüşme (overlap) içeren kayan pencere (sliding window) tekniği ve semantik parselleme kullandım."

### 2. Çoklu NLP Görevleri (Uçtan Uca Çözümler)
*   **Kullanılan Bileşenler:** `AnalyzerAgent`, `SummarizerAgent`, `RecommenderAgent`.
*   **Görevler:**
    *   **Metin Sınıflandırma & Analiz:** Doküman türü tespiti ve konu ana başlıklarının çıkarılması.
    *   **Varlık Tanıma (NER):** Doküman içindeki tarih, kişi, kurum ve anahtar terimlerin otomatik ekstraksiyonu.
    *   **Özetleme & Soru-Cevap:** Hiyerarşik özetleme ve doküman tabanlı bağlam-duyarlı (context-aware) sohbet yeteneği.
*   **Mimari Karar:** "Single-Prompt" (tek komut) yerine "Multi-Agent" (çoklu ajan) yapısı kullanılarak modelin "halüsinasyon" riski azaltıldı ve her ajanın kendi görevine odaklanması sağlandı.

### 3. Çıkarım (Inference) Optimizasyonu
*   **Kullanılan Teknolojiler:** `Ollama`, `Llama 3.2`, `Q4_K_M Quantization`.
*   **Detay:** 
    *   **Kuantizasyon:** 4-bit kuantize edilmiş modeller kullanılarak GPU bellek kullanımı %70 azaltıldı.
    *   **Async I/O:** Backend'de `httpx` ve `FastAPI` asenkron yapısı kullanılarak, LLM yanıt verirken API'ın diğer istekleri kabul etmesi sağlandı (Non-blocking I/O).
*   **Mülakat Notu:** "Sistemi yerel donanım kısıtları altında çalışacak şekilde optimize etmek için model kuantizasyonu ve asenkron çıkarım süreçlerini yönettim."

### 4. Güvenilir ve Sorumlu Yapay Zekâ (XAI & Anti-Hallucination)
*   **Çözüm:** 
    *   **Citations (Atıflar):** Her yanıtın hangi doküman parçasından (`chunk_id`) geldiği arayüzde ve API yanıtında belirtilir.
    *   **Confidence Scoring:** Yanıtların güvenilirlik düzeyi metrikleştirilir.
    *   **JSON Repair:** LLM'den gelen veriler Pydantic ile valide edilir, hatalı formatlar otomatik tamir edilir.
*   **Anlatım:** "Sistemin halüsinasyon görmesini engellemek için 'Grounding' teknikleri ve katı JSON şemaları uyguladım."

---

## 🚫 Kullanılmayan Teknolojiler (Out of Scope)
*   **Transformer Eğitimi / Fine-tuning (LoRA, QLoRA):** Projede Fine-tuning yerine RAG mimarisi tercih edilmiştir (Veri güncelliği ve açıklanabilirlik için).
*   **Model Alignment (RLHF, DPO):** Kullanılmamıştır.
*   **Kubernetes / Docker:** Yerel çalıştırma ve standart cloud deployment (Render/Vercel) hedeflenmiştir.
*   **PyTorch/TensorFlow:** Doğrudan kod içinde model katmanlarıyla işlem yapılmamış, yüksek seviye LLM framework'leri (Ollama) kullanılmıştır.

---

## 💡 Kritik Mülakat Soruları ve Cevaplar

**Soru: Neden Fine-tuning yerine RAG kullandın?**
*   **Cevap:** Fine-tuning dökümanlar güncellendiğinde modeli tekrar eğitmeyi gerektirir ve pahalıdır. RAG ise doküman eklendiği an veriyi güncel tutar. Ayrıca RAG ile modelin yanıtını doküman parçalarıyla ispatlaması (Atıflar) mümkündür, bu da kurumsal güvenilirlik sağlar.

**Soru: Vektör veri tabanında 'Size boundary' ve 'Overlap' ayarlarını nasıl belirledin?**
*   **Cevap:** Llama 3.2'nin bağlam penceresini (context window) ve embedding modelinin (`nomic-embed-text`) 768 boyutlu yapısını göz önünde bulundurarak 1500 karakterlik parçalar ve 300 karakterlik örtüşme seçtim. Bu, semantik bütünlüğü korurken vektör arama hızını maksimize etti.

**Soru: Sistemde gecikmeyi (latency) nasıl yönettin?**
*   **Cevap:** `num_predict` (maksimum token) sınırlandırması, kuantize modeller ve asenkron API çağrıları kullanarak kullanıcı deneyimini iyileştirdim.
