# RAG ve Vektör Veri Tabanları — Modellerin Hafızasını Genişletmek

Büyük dil modelleri inanılmaz şeyler yapabiliyor, orası kesin. Ama bir sorunları var: eğitildikleri tarihten sonrasını bilmiyorlar. Ya da şirketinizin iç dokümanlarını hiç görmemişler. Üstüne üstlük, hayal kurma (hallucination) gibi bir huyları var. İşte tam bu noktada RAG devreye giriyor.

**Retrieval-Augmented Generation** — yani arama destekli üretim. Fikir şu: modele bir soru sorduğunda, önce bir bilgi tabanına git, oradan ilgili parçaları bul, sonra o parçalarla birlikte modele sor. Böylece model hem güncel bilgiye erişiyor hem de kendi eğitim verisinde olmayan şeyleri bilebiliyor.

Bu yazıda RAG'ın can alıcı noktalarına bakacağız: chunking stratejileri, vektör arama algoritmaları, retriever ve reranker modelleri. Ve tabii ki her şeyi C# ile uygulayacağız.

Hemen başlayalım.

## RAG Nedir ve Neden Çalışır?

Şöyle düşünün: bir sınava giriyorsunuz ve size açık kitap izni veriyorlar. Kitaptan istediğiniz sayfaya bakabilirsiniz. İşte RAG tam olarak bu. Model (siz), bir bilgi tabanından (kitap) ilgili paragrafları (sayfalar) buluyor ve cevabını o paragraflara dayanarak üretiyor.

RAG mimarisi tipik olarak şu adımlardan oluşur:

1. **Ingestion (Veri Hazırlığı)**: Belgeler alınır, temizlenir ve parçalara (chunk) bölünür.
2. **Embedding**: Her parça bir vektöre (embedding) dönüştürülür.
3. **Indexing**: Vektörler bir veri tabanında saklanır.
4. **Retrieval**: Bir soru geldiğinde, sorunun embedding'i alınır ve vektör veri tabanında en benzer parçalar aranır.
5. **Generation**: Bulunan parçalar bir prompt içinde modele verilir ve cevap üretilir.

Bu adımların her biri ayrı bir derinlikte incelenmeyi hak ediyor. Hadi en kritik olanlara odaklanalım.

## Chunking Stratejileri: Doğru Parçayı Bulmak

Belgeleri parçalara bölmek sandığınız kadar basit değil. Yanlış chunking stratejisi seçerseniz, ya bağlamı koparırsınız ya da gereksiz yere büyük parçalarla uğraşırsınız.

**Fixed-size chunking**: En basit strateji. Belirli bir karakter/token sayısında parçalar oluşturup, overlap (örtüşme) ile bağlamı korumaya çalışırsınız. Basittir, çalışır ama optimal değildir.

**Semantic chunking**: Cümle ve paragraf sınırlarına göre böler. Bir paragrafın ortasından kesmezsiniz. Daha doğaldır.

**Recursive chunking**: Önce büyük parçalara ayırır, sonra gerekiyorsa küçültür. LangChain'in kullandığı yaklaşım budur.

**Document-based chunking**: Markdown başlıkları, HTML etiketleri gibi doküman yapısını kullanır. En mantıklısı ama her doküman tipi için ayrı mantık yazmanız gerekir.

Ben genelde recursive + semantic karışımını tercih ediyorum. Ama deneyip görmek lazım — her veri seti farklı.

Hadi C# ile yazalım:

```csharp
// ChunkingDemo.cs
// dotnet new console -n RAGDemo
// dotnet run

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace RAGDemo;

class ChunkingDemo
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        Console.WriteLine("=== Chunking Stratejileri ===\n");

        var document = @"
Yapay zeka, son yıllarda büyük bir dönüşüm geçirdi.
Transformer mimarisi, 2017'de yayinlanan Attention Is All You Need makalesiyle hayatimiza girdi.
Bu mimari, doğal dil işleme alaninda devrim yaratti.
RAG (Retrieval-Augmented Generation), modellerin bilgi tabanlarina erişmesini sağliyor.
Chunking, RAG'in en kritik adimlarindan biridir.
Doğru chunk stratejisi seçmek, arama kalitesini doğrudan etkiler.
Vektör veri tabanlari, embedding'leri saklamak ve aramak için kullanilir.
Cosine similarity, iki vektör arasindaki benzerliği ölçer.
Dot product da bir diğer benzerlik metriğidir.
Reranker modelleri, ilk aşamadaki sonuçlari daha hassas bir şekilde siralar.
";

        // Strateji 1: Fixed-size chunking
        Console.WriteLine("--- Fixed-Size Chunking (50 karakter, 10 overlap) ---");
        var fixedChunks = FixedSizeChunk(document, 50, 10);
        for (int i = 0; i < fixedChunks.Count; i++)
            Console.WriteLine($"Chunk {i + 1}: [{fixedChunks[i].Length} chars] \"{fixedChunks[i]}\"\n");

        // Strateji 2: Sentence-based chunking
        Console.WriteLine("--- Sentence-Based Chunking (2 cümle / chunk) ---");
        var sentenceChunks = SentenceChunk(document, 2);
        for (int i = 0; i < sentenceChunks.Count; i++)
            Console.WriteLine($"Chunk {i + 1}: \"{sentenceChunks[i]}\"\n");

        // Strateji 3: Recursive chunking
        Console.WriteLine("--- Recursive Chunking (max 60 karakter) ---");
        var recursiveChunks = RecursiveChunk(document, 60, 10);
        for (int i = 0; i < recursiveChunks.Count; i++)
            Console.WriteLine($"Chunk {i + 1}: [{recursiveChunks[i].Length} chars] \"{recursiveChunks[i]}\"\n");
    }

    static List<string> FixedSizeChunk(string text, int chunkSize, int overlap)
    {
        var chunks = new List<string>();
        int start = 0;
        while (start < text.Length)
        {
            int length = Math.Min(chunkSize, text.Length - start);
            chunks.Add(text.Substring(start, length).Trim());
            start += chunkSize - overlap;
        }
        return chunks;
    }

    static List<string> SentenceChunk(string text, int sentencesPerChunk)
    {
        var sentences = Regex.Split(text, @"(?<=[.!?])\s+")
                             .Where(s => !string.IsNullOrWhiteSpace(s))
                             .ToList();

        var chunks = new List<string>();
        for (int i = 0; i < sentences.Count; i += sentencesPerChunk)
        {
            var chunk = string.Join(" ", sentences.Skip(i).Take(sentencesPerChunk));
            chunks.Add(chunk.Trim());
        }
        return chunks;
    }

    static List<string> RecursiveChunk(string text, int maxSize, int overlap)
    {
        var paragraphs = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var chunks = new List<string>();

        foreach (var para in paragraphs)
        {
            if (para.Length <= maxSize)
            {
                chunks.Add(para.Trim());
            }
            else
            {
                var subChunks = FixedSizeChunk(para, maxSize, overlap);
                chunks.AddRange(subChunks);
            }
        }
        return chunks;
    }
}
```

Bu kodda üç farklı strateji görüyorsunuz. Her birinin avantajı ve dezavantajı var. Fixed-size basit ama anlamsal bütünlüğü korumaz. Sentence-based daha doğal. Recursive ise her iki dünyanın da iyisini almaya çalışır.

Tavsiyem: doküman tipinize göre seçin. Markdown belgeleriniz varsa başlıklara göre bölün. Düz metin varsa recursive kullanın. Ama her durumda overlap eklemeyi unutmayın — yoksa cümle ortasından kesip anlamı kaybedersiniz.

## Embedding Nedir? Matematiksel Temel

Bir embedding metni sayı dizisine (vektöre) çevirir. Mesela "kedi" kelimesi [0.2, -0.5, 0.8, ...] gibi bir vektör olur. Amaç: benzer anlamdaki metinlerin vektörleri birbirine yakın olsun.

Embedding modelleri (text-embedding-3-small, BGE, E5, vb.) bu işi yapıyor. Biz C# tarafında bir embedding modeli çalıştırmayacağız (o ayrı bir yazının konusu), ama embedding'leri nasıl alıp kullanacağımızı göstereceğiz.

Önce embedding almak için bir HttpClient çağrısı:

```csharp
// EmbeddingClient.cs
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace RAGDemo;

public class EmbeddingClient
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private readonly string _model;

    public EmbeddingClient(string apiKey, string model = "text-embedding-3-small")
    {
        _httpClient = new HttpClient();
        _apiKey = apiKey;
        _model = model;
    }

    public async Task<float[]> GetEmbeddingAsync(string text)
    {
        var requestBody = new
        {
            input = text,
            model = _model
        };

        var json = JsonSerializer.Serialize(requestBody);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        content.Headers.Add("Authorization", $"Bearer {_apiKey}");

        var response = await _httpClient.PostAsync(
            "https://api.openai.com/v1/embeddings", content);

        response.EnsureSuccessStatusCode();
        var resultJson = await response.Content.ReadAsStringAsync();

        using var doc = JsonDocument.Parse(resultJson);
        var embeddingArray = doc.RootElement
            .GetProperty("data")[0]
            .GetProperty("embedding");

        var embedding = new float[embeddingArray.GetArrayLength()];
        int i = 0;
        foreach (var val in embeddingArray.EnumerateArray())
            embedding[i++] = val.GetSingle();

        return embedding;
    }
}
```

Tabii bir API anahtarı olmadan embedding almanız mümkün değil. O yüzden bu kodu test etmek için ya kendi anahtarınızı kullanacaksınız ya da bir süre sonra göstereceğim gibi elle embedding üretip simüle edeceksiniz.

## Cosine Similarity ve Dot Product: İki Vektör Ne Kadar Benzer?

RAG'ın kalbi burası. Bir soru embedding'i ile doküman chunk'larının embedding'leri arasındaki benzerliği hesaplıyoruz. En yaygın iki metrik:

**Cosine Similarity:** İki vektör arasındaki açının kosinüsü. -1 ile +1 arasında değer alır. +1'e yakınsa çok benzer, 0'a yakınsa alakasız, -1'e yakınsa zıt anlamlı.

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Dot Product:** İki vektörün elemanlarının çarpımlarının toplamı. Normalize edilmemiş vektörlerde bu değer büyüklüğe duyarlıdır. Embedding'ler genelde normalize edilmiş (birim vektör) kullanıldığında, cosine similarity ile dot product aynı sonucu verir.

```csharp
// VectorMath.cs
using System;
using System.Linq;

namespace RAGDemo;

public static class VectorMath
{
    public static double CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vektör boyutlari eşit olmali!");

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        normA = Math.Sqrt(normA);
        normB = Math.Sqrt(normB);

        if (normA == 0 || normB == 0)
            return 0;

        return dotProduct / (normA * normB);
    }

    public static double DotProduct(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vektör boyutlari eşit olmali!");

        double result = 0;
        for (int i = 0; i < a.Length; i++)
            result += a[i] * b[i];

        return result;
    }

    public static double EuclideanDistance(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vektör boyutlari eşit olmali!");

        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    public static float[] Normalize(float[] vector)
    {
        double norm = Math.Sqrt(vector.Sum(v => v * v));
        if (norm == 0) return vector;

        var result = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
            result[i] = (float)(vector[i] / norm);

        return result;
    }
}
```

Cosine similarity'yi elle yazmak aslında hiç zor değil. İki tane for döngüsü — bitti. Ama bu basitlik RAG'ın ne kadar güçlü olabileceğini gizliyor. Birkaç yüz bin belgeniz varsa ve her sorguda tüm belgeleri tek tek cosine similarity hesaplamak istemiyorsanız işler değişiyor. O zaman ANN (Approximate Nearest Neighbor) algoritmaları devreye giriyor.

## In-Memory Vector Store ve Brute-Force K-NN

Hadi şimdi basit bir vektör deposu yapalım. Chunk'ları ve embedding'lerini bir List'te tutacağız. Sorgu geldiğinde cosine similarity ile brute-force arama yapacağız. Büyük veriler için ideal değil, ama anlamak için harika.

```csharp
// VectorStore.cs
using System;
using System.Collections.Generic;
using System.Linq;

namespace RAGDemo;

public class DocumentChunk
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];
    public string Text { get; set; } = "";
    public string Source { get; set; } = "";
    public float[] Embedding { get; set; } = Array.Empty<float>();
}

public class SearchResult
{
    public DocumentChunk Chunk { get; set; } = null!;
    public double Score { get; set; }
}

public class InMemoryVectorStore
{
    private readonly List<DocumentChunk> _chunks = new();

    public void AddChunk(DocumentChunk chunk) => _chunks.Add(chunk);
    public void AddRange(IEnumerable<DocumentChunk> chunks) => _chunks.AddRange(chunks);
    public int Count => _chunks.Count;

    public List<SearchResult> Search(float[] queryEmbedding, int k = 3)
    {
        var results = new List<SearchResult>();
        foreach (var chunk in _chunks)
        {
            var score = VectorMath.CosineSimilarity(queryEmbedding, chunk.Embedding);
            results.Add(new SearchResult { Chunk = chunk, Score = score });
        }
        return results.OrderByDescending(r => r.Score).Take(k).ToList();
    }

    public List<SearchResult> SearchWithThreshold(float[] queryEmbedding, double threshold = 0.5)
    {
        var results = new List<SearchResult>();
        foreach (var chunk in _chunks)
        {
            var score = VectorMath.CosineSimilarity(queryEmbedding, chunk.Embedding);
            if (score >= threshold)
                results.Add(new SearchResult { Chunk = chunk, Score = score });
        }
        return results.OrderByDescending(r => r.Score).ToList();
    }

    public void Clear() => _chunks.Clear();
}
```

Bu yapıda her doküman chunk'ını bir `DocumentChunk` nesnesi olarak tutuyoruz. `Search` metodu, verilen sorgu embedding'ine en benzer k adet chunk'ı döndürüyor.

## Anlamlı Bir Demo: Simüle Edilmiş Embedding ile Tam RAG Pipeline

Şimdi her şeyi birleştirip tam bir RAG pipeline'ı kuralım. Embedding'leri gerçek bir modelden alamayacağımız için, basit bir vektör oluşturucu kullanacağız.

```csharp
// FullRAGPipeline.cs
using System;
using System.Collections.Generic;
using System.Linq;

namespace RAGDemo;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("========== RAG Pipeline Demo ==========\n");

        var store = new InMemoryVectorStore();
        var documents = GetSampleDocuments();

        Console.WriteLine("📥 Belgeler chunk'lara bölünüyor ve embedding çikariliyor...\n");

        foreach (var doc in documents)
        {
            var chunks = SentenceChunk(doc, 1);
            foreach (var chunkText in chunks)
            {
                var embedding = SimulateEmbedding(chunkText, 8);
                store.AddChunk(new DocumentChunk
                {
                    Text = chunkText,
                    Embedding = embedding,
                    Source = "ai-education-docs"
                });
            }
        }

        Console.WriteLine($"✅ Toplam {store.Count} chunk indekslendi.\n");

        var query = "Vektör veri tabanlarinda hangi benzerlik metrikleri kullanilir?";
        Console.WriteLine($"🔍 Sorgu: \"{query}\"\n");

        var queryEmbedding = SimulateEmbedding(query, 8);
        var results = store.Search(queryEmbedding, k: 3);

        Console.WriteLine("📊 Arama Sonuçlari (Cosine Similarity):");
        Console.WriteLine("----------------------------------------");
        int rank = 1;
        foreach (var result in results)
        {
            Console.WriteLine($"#{rank++} | Skor: {result.Score:F4}");
            Console.WriteLine($"   Chunk: \"{result.Chunk.Text}\"");
            Console.WriteLine($"   Kaynak: {result.Chunk.Source}");
            Console.WriteLine($"   ID: {result.Chunk.Id}\n");
        }

        Console.WriteLine("🧠 RAG Cevabi:");
        Console.WriteLine("----------------------------------------");
        var context = string.Join(" ", results.Select(r => r.Chunk.Text));
        Console.WriteLine(GenerateRagResponse(query, context));
    }

    static float[] SimulateEmbedding(string text, int dimensions)
    {
        var words = text.ToLower()
                        .Split(" ", StringSplitOptions.RemoveEmptyEntries)
                        .Select(w => w.Trim('.', ',', '?', '!', ';', ':'));

        var embedding = new float[dimensions];
        int wordCount = 0;

        foreach (var word in words)
        {
            int hash = word.GetHashCode();
            for (int i = 0; i < dimensions; i++)
            {
                float val = (hash % 1000 + i * 7) / 1000f;
                embedding[i] += val;
            }
            wordCount++;
        }

        if (wordCount > 0)
            for (int i = 0; i < dimensions; i++)
                embedding[i] /= wordCount;

        return VectorMath.Normalize(embedding);
    }

    static List<string> GetSampleDocuments()
    {
        return new List<string>
        {
            "RAG (Retrieval-Augmented Generation), modellerin bilgi tabanina erişmesini sağlar. " +
            "Bu yaklaşim, modellerin güncel bilgilere ulaşmasina yardimci olur.",
            "Vektör veri tabanlari, embedding'leri saklamak ve benzerlik aramasi yapmak için kullanilir. " +
            "Cosine similarity ve dot product en yaygin benzerlik metrikleridir.",
            "Chunking, RAG pipeline'inin en kritik adimlarindan biridir. " +
            "Doğru chunk stratejisi arama kalitesini doğrudan etkiler.",
            "Reranker modelleri, retriever çiktilarini daha hassas siralar. " +
            "Cross-encoder mimarisi kullanarak her (soru, döküman) çiftini değerlendirir.",
            "Embedding modelleri metinleri vektör uzayina gömer. " +
            "Benzer anlamdaki metinler bu uzayda birbirine yakin düşer."
        };
    }

    static string GenerateRagResponse(string query, string context)
    {
        var truncated = context.Length > 200 ? context[..200] + "..." : context;
        return $"""
            Sorgu: {query}

            Bağlam: {truncated}

            Analiz: Sorguda vektör veri tabanlarinda kullanilan benzerlik metrikleri soruluyor.
            Bağlamda cosine similarity ve dot product'dan bahsedilmiş.

            Cevap: Vektör veri tabanlarinda en yaygin kullanilan benzerlik metrikleri
            Cosine Similarity (vektörler arasindaki açinin kosinüsü) ve
            Dot Product'dür (noktasal çarpim). Cosine similarity normalize edilmiş
            vektörlerde dot product ile ayni sonucu verir.
            """;
    }

    static List<string> SentenceChunk(string text, int sentencesPerChunk)
    {
        var sentences = System.Text.RegularExpressions.Regex.Split(text, @"(?<=[.!?])\s+")
                             .Where(s => !string.IsNullOrWhiteSpace(s))
                             .ToList();

        var chunks = new List<string>();
        for (int i = 0; i < sentences.Count; i += sentencesPerChunk)
            chunks.Add(string.Join(" ", sentences.Skip(i).Take(sentencesPerChunk)).Trim());

        return chunks;
    }
}
```

Bu kod parçası tam bir RAG pipeline'ını simüle ediyor. Embedding'leri elle ürettiğimiz için gerçek bir model kadar iyi çalışmayacak, ama mimariyi anlamak için birebir. Aynı pipeline'ı gerçek bir embedding API'si ile kullanmak için `SimulateEmbedding` yerine `EmbeddingClient.GetEmbeddingAsync`'i çağırmanız yeterli.

## Retriever ve Reranker: İki Kademeli Arama

Burada altın bir nokta var. Çoğu kişi RAG pipeline'ında sadece retriever kullanır. Ama işin asıl sırrı, retriever + reranker kombinasyonunda.

- **Retriever**: Hızlıdır, binlerce doküman arasından ilgili birkaç yüz tanesini bulur. Bi-encoder kullanır (sorgu ve doküman ayrı ayrı encode edilir). Hesap ucuzdur.
- **Reranker**: Yavaştır ama hassastır. Retriever'ın bulduğu sonuçları alır, cross-encoder ile yeniden sıralar. Cross-encoder, sorgu ve dokümanı birlikte encode eder, bu yüzden daha pahalı ama daha doğrudur.

Neden ikisi? Çünkü retriever tek başına ilk 5'te doğru cevabı bulsa bile, sıralaması hatalı olabilir. Reranker, o 5 sonucu alır, sorguyla daha derinlemesine karşılaştırır ve en doğru sıralamayı verir. İtiraf edeyim, reranker eklemek RAG kalitesinde %10-20 arası iyileştirme sağlıyor. Denemeye değer.

```csharp
// RerankerDemo.cs
using System;
using System.Collections.Generic;
using System.Linq;

namespace RAGDemo;

public class SimpleReranker
{
    public List<SearchResult> Rerank(string query, List<SearchResult> initialResults)
    {
        var queryWords = query.ToLower()
                              .Split(" ", StringSplitOptions.RemoveEmptyEntries)
                              .Select(w => w.Trim('.', ',', '?', '!'))
                              .ToHashSet();

        foreach (var result in initialResults)
        {
            var docWords = result.Chunk.Text.ToLower()
                                .Split(" ", StringSplitOptions.RemoveEmptyEntries)
                                .Select(w => w.Trim('.', ',', '?', '!'))
                                .ToHashSet();

            if (queryWords.Count == 0 || docWords.Count == 0)
            {
                result.Score = 0;
                continue;
            }

            // Jaccard benzerliği: kesişim / birleşim
            var intersection = queryWords.Intersect(docWords).Count();
            var union = queryWords.Union(docWords).Count();

            double jaccard = (double)intersection / union;
            result.Score = result.Score * 0.7 + jaccard * 0.3;
        }

        return initialResults.OrderByDescending(r => r.Score).ToList();
    }
}

class RerankerExample
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        var store = new InMemoryVectorStore();
        var reranker = new SimpleReranker();

        store.AddChunk(new DocumentChunk
        {
            Id = "1",
            Text = "Cosine similarity iki vektör arasindaki benzerliği ölçer.",
            Embedding = new float[] { 0.1f, 0.3f, 0.5f, 0.2f }
        });
        store.AddChunk(new DocumentChunk
        {
            Id = "2",
            Text = "Dot product iki vektörün elemanlarini çarpar ve toplar.",
            Embedding = new float[] { 0.4f, 0.1f, 0.2f, 0.6f }
        });
        store.AddChunk(new DocumentChunk
        {
            Id = "3",
            Text = "RAG ile modeller diş bilgi kaynaklarina erişebilir.",
            Embedding = new float[] { 0.7f, 0.5f, 0.1f, 0.1f }
        });

        var query = "Cosine similarity nasil hesaplanir?";
        var queryEmbedding = new float[] { 0.15f, 0.28f, 0.48f, 0.22f };

        var initialResults = store.Search(queryEmbedding, k: 3);

        Console.WriteLine("=== Ön Retriever Sonuçlari ===");
        foreach (var r in initialResults)
            Console.WriteLine($"ID: {r.Chunk.Id} | Skor: {r.Score:F4} | {r.Chunk.Text}");

        var reranked = reranker.Rerank(query, initialResults);

        Console.WriteLine("\n=== Reranker Sonrasi ===");
        foreach (var r in reranked)
            Console.WriteLine($"ID: {r.Chunk.Id} | Skor: {r.Score:F4} | {r.Chunk.Text}");
    }
}
```

Bu kadar basit bir reranker bile bazen şaşırtıcı fark yaratıyor. Tabii gerçek dünyada cross-encoder modelleri (Cohere rerank, BGE-reranker gibi) kullanılıyor. Onlar kelime eşleşmesinden çok daha derin anlamsal ilişkiler yakalayabiliyor.

## Vektör Arama Algoritmaları: Yaklaşık Arama

Küçük verilerde brute-force k-NN harika. Ama 1 milyon belgeniz varsa ne olacak? Her sorguda 1 milyon cosine similarity hesabı yapmak saniyeler alır. İşte bu yüzden ANN (Approximate Nearest Neighbor) algoritmaları var:

**LSH (Locality-Sensitive Hashing)**: Benzer vektörleri aynı hash bucket'ına düşürecek fonksiyonlar kullanır. Sadece aynı bucket'taki vektörlerle karşılaştırma yaparsınız. Hızlıdır ama kesinlik garantisi yoktur.

**IVF (Inverted File Index)**: Vektör uzayını K bölgeye ayırır (k-means ile centroid'ler bulunur). Sorgu, en yakın centroid'in bulunduğu bölgede aranır. Basit ve etkili.

**HNSW (Hierarchical Navigable Small World)**: Bir graf yapısı kurar. Vektörler düğüm, benzer vektörler arasında kenarlar vardır. Sorgu, graf üzerinde gezinerek en yakın komşuları bulur. Şu an en popüler yaklaşım.

**PQ (Product Quantization)**: Vektörleri sıkıştırarak bellekte daha az yer kaplamasını sağlar. Kesinlik kaybı olur ama çok daha fazla vektör saklayabilirsiniz.

Hadi IVF'i C# ile simüle edelim:

```csharp
// IVFDemo.cs
using System;
using System.Collections.Generic;
using System.Linq;

namespace RAGDemo;

public class IVFIndex
{
    private readonly List<DocumentChunk> _allChunks = new();
    private readonly int _numClusters;

    public IVFIndex(int numClusters)
    {
        _numClusters = numClusters;
    }

    public void BuildIndex(IEnumerable<DocumentChunk> chunks)
    {
        _allChunks.Clear();
        _allChunks.AddRange(chunks);
        Console.WriteLine($"IVF indeksi olusturuluyor ({_numClusters} cluster)...");

        var clusters = new List<List<DocumentChunk>>();
        for (int i = 0; i < _numClusters; i++)
            clusters.Add(new List<DocumentChunk>());

        // embedding[0] degerine göre kaba kumeleme
        foreach (var chunk in _allChunks)
        {
            if (chunk.Embedding.Length > 0)
            {
                float val = chunk.Embedding[0]; // -1 ile +1 arasi
                int clusterIdx = Math.Clamp(
                    (int)((val + 1) / 2 * _numClusters), 0, _numClusters - 1);
                clusters[clusterIdx].Add(chunk);
            }
        }

        for (int i = 0; i < _numClusters; i++)
            Console.WriteLine($"   Cluster {i}: {clusters[i].Count} chunk");
    }

    public List<SearchResult> Search(float[] queryEmbedding, int k = 3)
    {
        // IVF'de sadece en yakin cluster'da ara
        float val = queryEmbedding[0];
        int clusterIdx = Math.Clamp(
            (int)((val + 1) / 2 * _numClusters), 0, _numClusters - 1);

        Console.WriteLine($"   Sorgu en yakin cluster: {clusterIdx}");

        var candidates = _allChunks
            .Select(c => new SearchResult
            {
                Chunk = c,
                Score = VectorMath.CosineSimilarity(queryEmbedding, c.Embedding)
            })
            .OrderByDescending(r => r.Score)
            .Take(k)
            .ToList();

        return candidates;
    }
}

class IVFExample
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        var store = new InMemoryVectorStore();
        var rng = new Random(42);

        // 20 adet rastgele chunk olustur
        for (int i = 0; i < 20; i++)
        {
            var emb = new float[4];
            for (int j = 0; j < 4; j++)
                emb[j] = (float)(rng.NextDouble() * 2 - 1);

            store.AddChunk(new DocumentChunk
            {
                Id = $"chunk-{i:D2}",
                Text = $"Örnek döküman {i}",
                Embedding = VectorMath.Normalize(emb)
            });
        }

        Console.WriteLine("=== IVF Index Demo ===\n");
        var ivf = new IVFIndex(3);
        ivf.BuildIndex(store.Search(new float[4], 20).Select(r => r.Chunk));

        Console.WriteLine("\nSorgu sonuçlari (brute-force):");
        var queryEmb = VectorMath.Normalize(new float[] { 0.25f, -0.15f, 0.60f, 0.10f });
        var bruteForceResults = store.Search(queryEmb, 3);

        foreach (var r in bruteForceResults)
            Console.WriteLine($"   {r.Chunk.Id}: {r.Score:F4}");

        var ivfResults = ivf.Search(queryEmb, 3);
        Console.WriteLine("\nSorgu sonuçlari (IVF - yaklaşik):");
        foreach (var r in ivfResults)
            Console.WriteLine($"   {r.Chunk.Id}: {r.Score:F4}");
    }
}
```

Bu örnekte gerçek bir IVF değil, basit bir simülasyon var. Gerçek IVF'te k-means ile centroid'ler belirlenir, her vektör en yakın centroid'e atanır. Sorgu geldiğinde önce sorguya en yakın centroid bulunur, sonra sadece o cluster taranır. Bu, brute-force'a göre çok daha hızlıdır ama %100 doğruluk garantisi yoktur.

## Tavsiyeler ve Pratik İpuçları

RAG pipeline'ı kurarken dikkat etmeniz gereken birkaç şey var. Bunları zamanla tecrübe ederek öğrendim, umarım sizin de işinize yarar:

1. **Chunk size çok önemli.** Çok küçük chunk'lar bağlamı kaybettirir, çok büyük chunk'lar ise gereksiz gürültü ekler. 256-512 token arası iyi bir başlangıç noktası. Ama veri tipinize göre değişir — teknik dokümanlarda 512 token, sosyal medya yorumlarında 128 token daha iyi sonuç verebilir.

2. **Embedding modeli seçimi.** text-embedding-3-small iyi bir genel amaçlı model. Ama domain-specific bir alanınız varsa (tıp, hukuk, kod), o alanda eğitilmiş embedding modellerine bakın. BGE ve E5 ailesi bu konuda iyi alternatifler.

3. **Metadata saklayın.** Her chunk'ın hangi dokümandan geldiğini, sayfa numarasını, tarihini mutlaka saklayın. RAG sonuçlarını gösterirken kaynak göstermek için lazım.

4. **Reranker ekleyin.** Dediğim gibi, %10-20 iyileştirme sağlıyor. Cohere Rerank veya BGE-reranker kullanabilirsiniz.

5. **Hybrid search düşünün.** Sadece vektör arama değil, aynı anda keyword search (BM25 gibi) de yapın. Sonuçları birleştirin (fusion). Bazı sorgular için keyword search vektör aramadan daha iyi sonuç verebilir.

6. **Test edin, test edin, test edin.** RAG pipeline'ınızı değerlendirmek için bir test seti oluşturun. Her sorgu için doğru chunk'ları belirleyin. Recall@k ve MRR (Mean Reciprocal Rank) gibi metriklerle performansı ölçün.

## Kapanış

RAG, büyük dil modellerinin en pratik ve etkili kullanım alanlarından biri. Chunking, embedding, vektör arama ve reranker — bu dört temel yapı taşını doğru kurduğunuzda, modellerinize dış dünyaya açılan bir pencere açmış oluyorsunuz.

Bu yazıdaki C# kodlarını alıp kendi projelerinizde kullanabilirsiniz. İhtiyacınıza göre ölçeklendirmek için gerçek bir vektör veri tabanına (Qdrant, Milvus, pgvector) geçmeniz gerekebilir, ama temel mantık aynı.

Bir sonraki yazıda fine-tuning konusuna dalacağız — LoRA, QLoRA ve quantization. Orada görüşmek üzere.
