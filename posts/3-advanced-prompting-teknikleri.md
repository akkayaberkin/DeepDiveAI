# Advanced Prompting Teknikleri — Sadece "Daha iyi cevap ver" Demek Yetmez

Prompt mühendisliği, son iki yıldır adını en çok duyduğumuz konulardan biri haline geldi. Ama dürüst olalım: bir modele "daha iyi cevap ver" demek, genelde daha iyi cevap getirmiyor. İşin püf noktası, **doğru yapıyı** kurgulamakta.

Bu yazıda Chain-of-Thought'tan Tree-of-Thoughts'a, ReAct döngülerinden structured output almaya kadar — hepsini hem teoride hem de C# ile pratikte göreceğiz. Hazırsanız başlayalım.

## Chain-of-Thought (CoT): Adım Adım Düşünmek

Chain-of-Thought, 2022'de Google'daki araştırmacıların yayınladığı bir teknik (Wei et al., 2022). Fikir basit: modele sadece cevabı değil, cevaba giden yolu da yazdır. Bunu yapınca karmaşık mantıksal akıl yürütme gerektiren problemlerde çok ciddi başarı artışı görülüyor.

Neden mi? Çünkü model her adımda kendi ürettiği ara çıktıları kullanarak bir sonraki adımı daha isabetli hesaplayabiliyor. Sanki bir matematik problemini çözerken her adımı yazmak gibi — arada kaybolmuyorsunuz.

### Zero-shot vs Few-shot CoT

İki varyasyonu var:

**Zero-shot CoT:** En basit hali. Prompt'un sonuna `"Adım adım düşünelim."` veya `"Let's think step by step."` ekliyorsunuz. Şaşırtıcı derecede iyi çalışıyor. Wei ve arkadaşları GSM8K verisetinde zero-shot CoT ile %10.4'ten %40.7'ye sıçrama elde etmiş. Ciddi bir fark.

**Few-shot CoT:** Birkaç örnek verip, her örneğin adım adım nasıl çözüldüğünü gösteriyorsunuz. Ardından asıl soruyu soruyorsunuz.

Aralarındaki farka gelecek olursak: zero-shot basit sorular için yeterli, few-shot ise daha karmaşık ve domain-specific problemler için ideal. Ama few-shot'ın bir bedeli var — context window'dan yiyor ve her örnekle maliyet artıyor.

Hadi bunu C# ile yapılandıralım. Önce bir CoT prompt yapısı oluşturalım:

```csharp
// CoTDemo.cs
// dotnet new console -n CoTDemo
// dotnet run

namespace CoTDemo;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // Chain-of-Thought prompt yapısını modelliyoruz
        var cotPrompt = new CoTPrompt
        {
            SystemMessage = "Sen bir matematik asistanısın. Her soruyu adım adım çöz.",
            Examples = new List<CoTExample>
            {
                new CoTExample
                {
                    Question = "Bir çiftlikte 12 koyun ve 8 tavuk var. Toplam kaç bacak var?",
                    Reasoning = new List<string>
                    {
                        "1. Her koyunun 4 bacağı var. 12 koyun x 4 = 48 bacak.",
                        "2. Her tavuğun 2 bacağı var. 8 tavuk x 2 = 16 bacak.",
                        "3. Toplam: 48 + 16 = 64 bacak."
                    },
                    Answer = "64 bacak"
                }
            },
            Question = "Bir otobüste 7 kişi var. İlk durakta 3 kişi bindi, 2 kişi indi. " +
                      "İkinci durakta 5 kişi bindi, 1 kişi indi. Kaç kişi kaldı?",
            Instruction = "Adım adım düşünelim."
        };

        Console.WriteLine("=== Chain-of-Thought Prompt ===\n");
        Console.WriteLine(BuildCotPrompt(cotPrompt));

        // Reasoning adımlarını simüle edelim
        Console.WriteLine("\n🧠 Model Şu Şekilde Düşünebilir:\n");
        SimulateReasoning(cotPrompt.Question);

        Console.WriteLine("\n=== Structured Çıktı (JSON) ===\n");
        var structuredOutput = new CotResult
        {
            Question = cotPrompt.Question,
            Steps = new List<string>
            {
                "Başlangıç: 7 kişi",
                "İlk durak: 7 + 3 - 2 = 8 kişi",
                "İkinci durak: 8 + 5 - 1 = 12 kişi"
            },
            FinalAnswer = "12 kişi"
        };

        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(
            structuredOutput, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    }

    static string BuildCotPrompt(CoTPrompt prompt)
    {
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("### Sistem Mesaji ###");
        sb.AppendLine(prompt.SystemMessage);
        sb.AppendLine();

        if (prompt.Examples.Any())
        {
            sb.AppendLine("### Örnekler ###");
            foreach (var example in prompt.Examples)
            {
                sb.AppendLine($"\nSoru: {example.Question}");
                sb.AppendLine("Çözüm:");
                foreach (var step in example.Reasoning)
                {
                    sb.AppendLine($"  {step}");
                }
                sb.AppendLine($"Cevap: {example.Answer}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("### Asil Soru ###");
        sb.AppendLine(prompt.Question);
        sb.AppendLine();
        sb.AppendLine("### Yönergeler ###");
        sb.AppendLine(prompt.Instruction);

        return sb.ToString();
    }

    static void SimulateReasoning(string question)
    {
        // Modelin nasıl reasoning yapacağını simüle ediyoruz
        int initial = 7;
        int afterFirst = initial + 3 - 2;
        int afterSecond = afterFirst + 5 - 1;

        Console.WriteLine($"  Başlangıç: {initial} kişi");
        Console.WriteLine($"  İlk durak: {initial} + 3 - 2 = {afterFirst} kişi");
        Console.WriteLine($"  İkinci durak: {afterFirst} + 5 - 1 = {afterSecond} kişi");
        Console.WriteLine($"\n  Cevap: {afterSecond} kişi");
    }
}

// --- Model Sınıfları ---

class CoTPrompt
{
    public string SystemMessage { get; set; } = "";
    public List<CoTExample> Examples { get; set; } = new();
    public string Question { get; set; } = "";
    public string Instruction { get; set; } = "";
}

class CoTExample
{
    public string Question { get; set; } = "";
    public List<string> Reasoning { get; set; } = new();
    public string Answer { get; set; } = "";
}

class CotResult
{
    public string Question { get; set; } = "";
    public List<string> Steps { get; set; } = new();
    public string FinalAnswer { get; set; } = "";
}
```

Bu kodun güzel yanı, CoT prompt yapısını nesne olarak modellemesi. İleride bunu bir API'ye göndermek isterseniz, sadece `BuildCotPrompt` çıktısını HTTP body'sine koymanız yeterli.

## Tree-of-Thoughts (ToT): Birden Fazla Yol

Chain-of-Thought'taki "tek bir mantıksal yol" fikrini beğendiniz mi? Peki ya o yol çıkmaz sokağa çıkarsa? İşte Tree-of-Thoughts (Yao et al., 2023) tam da bu sorunu çözmek için geliyor.

ToT, her adımda birden fazla olası düşünceyi değerlendiriyor, her birini puanlıyor ve en umut verici dallara devam ediyor. Adı üstünde — bir ağaç yapısı.

Düşünün ki bir satranç oynuyorsunuz. Her hamlede sadece bir hamle değil, birkaç olası hamleyi değerlendiriyor, her biri için rakibin tepkisini tahmin ediyor, sonra en iyi sonucu veren dala devam ediyorsunuz. ToT da aynen böyle çalışıyor.

```csharp
// ToTDemo.cs
// dotnet new console -n ToTDemo
// dotnet run

namespace ToTDemo;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        var problem = "3, 5 ve 7 sayılarini kullanarak (her birini tam olarak bir kez) " +
                     "toplama, çıkarma, çarpma ve bölme işlemleriyle 42'yi elde edin.";

        Console.WriteLine($"Problem: {problem}\n");

        var tree = new ThoughtTree(problem, maxBranches: 3, maxDepth: 4);
        tree.BuildTree();
        tree.PrintBestPath();
    }
}

class ThoughtNode
{
    public string Thought { get; set; }
    public double Score { get; set; }
    public int Depth { get; set; }
    public List<ThoughtNode> Children { get; set; } = new();
    public ThoughtNode? Parent { get; set; }

    public ThoughtNode(string thought, int depth, double score = 0)
    {
        Thought = thought;
        Depth = depth;
        Score = score;
    }
}

class ThoughtTree
{
    private readonly string _problem;
    private readonly int _maxBranches;
    private readonly int _maxDepth;
    private readonly Random _rng = new(42);
    private readonly ThoughtNode _root;

    public ThoughtTree(string problem, int maxBranches = 3, int maxDepth = 4)
    {
        _problem = problem;
        _maxBranches = maxBranches;
        _maxDepth = maxDepth;
        _root = new ThoughtNode($"Başlangıç: {problem}", 0, 1.0);
    }

    public void BuildTree()
    {
        var queue = new Queue<ThoughtNode>();
        queue.Enqueue(_root);
        int nodeCount = 0;

        while (queue.Count > 0 && nodeCount < 40)
        {
            var current = queue.Dequeue();
            if (current.Depth >= _maxDepth) continue;

            for (int i = 0; i < _maxBranches; i++)
            {
                if (nodeCount >= 40) break;

                var nextThought = GenerateThought(current.Depth + 1);
                var score = EvaluateThought(nextThought);

                var child = new ThoughtNode(nextThought, current.Depth + 1, score)
                {
                    Parent = current
                };

                current.Children.Add(child);
                queue.Enqueue(child);
                nodeCount++;

                Console.WriteLine($"  Dusunce [{current.Depth + 1}.{i + 1}]: {nextThought,-60} Skor: {score,5:F2}");
            }
        }

        Console.WriteLine($"\nToplam dugum sayisi: {CountNodes(_root)}");
    }

    private string GenerateThought(int depth)
    {
        var possibleThoughts = new Dictionary<int, string[]>
        {
            [1] = new[] {
                "Önce 7 × 5 = 35 yapabilirim.",
                "Önce 7 × 3 = 21 yapabilirim.",
                "Önce 5 × 3 = 15 yapabilirim."
            },
            [2] = new[] {
                "35 + 3 = 38, sonra 38 ile ne yapabilirim? 5 kalmışti.",
                "21 + 5 = 26, 26 + 7 = 33, yetmez.",
                "15 + 7 = 22, 22 + 5 = 27, yetmez."
            },
            [3] = new[] {
                "7 × 5 = 35, 35 + 7 = 42 ama 7'yi iki kere kullandim.",
                "7 × 3 = 21, 21 x 2 yok... sadece 5 kaldi, 21 + 5 = 26.",
                "5 × 3 = 15, 15 + 7 = 22, 22'yi 42 yapmak için ne lazim?"
            },
            [4] = new[] {
                "7 x 5 = 35, 35 + 7 geçersiz. Belki bölme denemeliyim?",
                "(7 - 3) x 5 = 20 olmuyor. (7 + 3) x 5 = 50 çok fazla.",
                "(7 x 3) x 2 = 42 olurdu ama 2 yok, sadece 5 kaldi."
            }
        };

        if (possibleThoughts.ContainsKey(depth))
            return possibleThoughts[depth][_rng.Next(possibleThoughts[depth].Length)];

        return $"{depth}. adimda alternatif yol deneniyor...";
    }

    private double EvaluateThought(string thought)
    {
        if (thought.Contains("42")) return _rng.NextDouble() * 0.5 + 0.5;
        if (thought.Contains("?")) return _rng.NextDouble() * 0.3;
        if (thought.Contains("yetmez") || thought.Contains("geçersiz"))
            return _rng.NextDouble() * 0.2 + 0.1;
        return _rng.NextDouble() * 0.4 + 0.3;
    }

    public void PrintBestPath()
    {
        var bestPath = new List<ThoughtNode>();
        FindBestPath(_root, new List<ThoughtNode>(), ref bestPath);

        Console.WriteLine("\nEn Iyi Dusunce Yolu:");
        for (int i = 0; i < bestPath.Count; i++)
        {
            string prefix = i == 0 ? "Baslangic" : $"Adim {i}";
            Console.WriteLine($"  {prefix}: {bestPath[i].Thought} (skor: {bestPath[i].Score:F2})");
        }

        Console.WriteLine($"\nToplam skor: {bestPath.Sum(n => n.Score):F2}");
    }

    private void FindBestPath(ThoughtNode node, List<ThoughtNode> current, ref List<ThoughtNode> best)
    {
        current.Add(node);

        if (!node.Children.Any())
        {
            double currentSum = current.Sum(n => n.Score);
            double bestSum = best.Sum(n => n.Score);
            if (currentSum > bestSum)
                best = new List<ThoughtNode>(current);
        }
        else
        {
            foreach (var child in node.Children)
                FindBestPath(child, current, ref best);
        }

        current.RemoveAt(current.Count - 1);
    }

    private int CountNodes(ThoughtNode node)
    {
        return 1 + node.Children.Sum(CountNodes);
    }
}
```

Bu örnekte gerçek bir LLM çağrısı yapmıyoruz tabii — öyle olsa her adımda API'ye istek atardık. Ama mantık aynı: her adımda birden fazla olasılık üret, puanla, en iyi dallara devam et.

ToT'in CoT'a üstünlüğü, hatalı bir yola girince geri dönüp başka bir dalı deneyebilmesi. CoT'ta bir kere yanlış adım attınız mı, sonraki tüm adımlar çöp oluyor. ToT'te ise bir arama algoritması işletiyorsunuz.

## ReAct: Reasoning + Acting

Chain-of-Thought sadece "düşünüyor". Tree-of-Thoughts "birden çok yol düşünüyor". Peki ya düşünüp **eyleme geçse**?

İşte ReAct (Yao et al., 2022) tam olarak bunu yapıyor. Reasoning (akıl yürütme) ve Acting (eylem) arasında gidip gelen bir döngü. Model bir düşünce üretiyor, bir eylem yapıyor (örneğin bir API çağrısı veya veritabanı sorgusu), gözlem yapıyor ve tekrar düşünüyor.

Şöyle bir döngü:

```
Thought: Kullanıcı İstanbul'daki hava durumunu soruyor. Hava durumu API'sine ihtiyacım var.
Action: call_weather_api[İstanbul]
Observation: { "temp": 28, "condition": "Açık" }
Thought: Hava 28 derece ve açık. Kullanıcıya bunu söyleyeyim.
Action: respond[İstanbul'da hava şu anda 28°C ve açık.]
```

Hadi bunu C# ile yazalım. Gerçek bir API çağrısı yaparak:

```csharp
// ReActDemo.cs
// dotnet new console -n ReActDemo
// dotnet run

using System.Net.Http.Json;
using System.Text.Json;

namespace ReActDemo;

class Program
{
    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        var agent = new ReActAgent();

        var soru = "İstanbul ve Ankara'nın hava durumunu karşılaştır ve hangi şehirde " +
                   "dışarı çıkmak için daha uygun olduğunu söyle.";

        var result = await agent.RunAsync(soru);

        Console.WriteLine("\n=== Nihai Cevap ===");
        Console.WriteLine(result);
        Console.WriteLine($"Toplam adim: {agent.TotalSteps}");
        Console.WriteLine($"Tahmini token: {agent.EstimatedTokens}");
    }
}

enum ReActStepType
{
    Thought,
    Action,
    Observation,
    FinalAnswer
}

class ReActStep
{
    public ReActStepType Type { get; set; }
    public string Content { get; set; } = "";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

class ReActAgent
{
    private readonly List<ReActStep> _steps = new();
    private readonly HttpClient _http = new();

    public int TotalSteps => _steps.Count;
    public int EstimatedTokens => _steps.Sum(s => s.Content.Split(' ').Length) * 2;

    public async Task<string> RunAsync(string userQuery)
    {
        Console.WriteLine($"Kullanici Sorusu: {userQuery}\n");
        Console.WriteLine("ReAct Döngüsü Başliyor...\n");

        AddStep(ReActStepType.Thought,
            "Kullanici iki sehrin hava durumunu karsilastirmak istiyor. " +
            "Önce Istanbul'un hava durumunu alayim.");

        AddStep(ReActStepType.Action, "get_weather(sehir: \"Istanbul\")");
        var istanbulWeather = await FetchWeatherAsync("Istanbul");
        AddStep(ReActStepType.Observation, istanbulWeather);

        AddStep(ReActStepType.Thought,
            "Istanbul verisi geldi. Simdi Ankara'nin verisini alayim.");
        AddStep(ReActStepType.Action, "get_weather(sehir: \"Ankara\")");
        var ankaraWeather = await FetchWeatherAsync("Ankara");
        AddStep(ReActStepType.Observation, ankaraWeather);

        AddStep(ReActStepType.Thought,
            "Iki sehrin de verileri elimde. Karsilastirma yapabilirim.");

        var finalAnswer = CompareAndRespond(istanbulWeather, ankaraWeather);
        AddStep(ReActStepType.FinalAnswer, finalAnswer);

        PrintSteps();
        return finalAnswer;
    }

    private async Task<string> FetchWeatherAsync(string city)
    {
        try
        {
            var url = $"https://wttr.in/{city}?format=j1";
            var response = await _http.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            var current = doc.RootElement.GetProperty("current_condition")[0];
            var temp = current.GetProperty("temp_C").GetString();
            var desc = current.GetProperty("weatherDesc")[0].GetProperty("value").GetString();
            var humidity = current.GetProperty("humidity").GetString();

            return $"{city}: {temp}°C, {desc}, Nem: %{humidity}";
        }
        catch
        {
            return SimulateWeather(city);
        }
    }

    private string SimulateWeather(string city)
    {
        var data = new Dictionary<string, string>
        {
            ["Istanbul"] = "Istanbul: 26°C, Parçali Bulutlu, Nem: %65",
            ["Ankara"] = "Ankara: 28°C, Açik, Nem: %40"
        };

        return data.GetValueOrDefault(city, $"{city}: 22°C, Bilinmiyor, Nem: %50");
    }

    private string CompareAndRespond(string city1, string city2)
    {
        return $"KARSILASTIRMA SONUCU:\n\n" +
               $"  Istanbul: {city1}\n" +
               $"  Ankara: {city2}\n\n" +
               $"Tavsiye: Her iki sehir de gayet güzel. Istanbul daha nemli ve bulutlu, " +
               $"Ankara daha kuru ve günesli. Eger nemden rahatsiz oluyorsaniz Ankara'yi, " +
               $"yoksa Istanbul'u tercih edebilirsiniz!";
    }

    private void AddStep(ReActStepType type, string content)
    {
        _steps.Add(new ReActStep { Type = type, Content = content });
    }

    private void PrintSteps()
    {
        foreach (var step in _steps)
        {
            string emoji = step.Type switch
            {
                ReActStepType.Thought => "💭",
                ReActStepType.Action => "⚡",
                ReActStepType.Observation => "👁",
                ReActStepType.FinalAnswer => "✅",
                _ => "?"
            };

            Console.WriteLine($"{emoji} [{step.Type,-12}]: {step.Content}\n");
        }
    }
}
```

Bu örnekte ReAct döngüsü su sekilde isliyor:

1. **Think:** Kullanicinin ne istedigini analiz et
2. **Act:** Hava durumu API'sine istek at
3. **Observe:** API'den dönen JSON'i isle
4. **Think:** Simdi bu veriyi nasil kullanacagini planla
5. **Act:** Ikinci API çagrisini yap
6. **Observe:** Ikinci sehrin verisini al
7. **Think:** Tüm verileri kullanarak karsilastirma yap
8. **Final Answer:** Kullaniciya cevabi göster

Her Thought adiminda model kendi kendine konusuyor. Bu, daha önce konustugumuz CoT'un bir parçasi. ReAct aslinda CoT'u temel aliyor, üstüne eylem yetenegi ekliyor.

## Structured Output: JSON ve XML ile Kesin Cevap

Bazen modelin serbest metin degil, belirli bir formatta çikti vermesini istersiniz. Bir JSON dönmesi, belirli alanlari doldurmasi veya XML etiketleri arasinda cevap vermesi gerekebilir.

Iste burada "structured output" veya diger adiyla "constrained decoding" devreye giriyor. Fikir: modelin çikti uzayini kisitlayarak sadece geçerli JSON üretmesini saglamak.

C# tarafinda bu genelde iki sekilde yapilir:

1. **Prompt engineering:** Modele JSON formatini detaylica anlatmak
2. **Post-processing:** Model çiktisini Regex veya parser ile islemek

Hadi ikisini de görelim:

```csharp
// StructuredOutputDemo.cs
// dotnet new console -n StructuredOutputDemo
// dotnet run

using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace StructuredOutputDemo;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        var customerReview = "Ürünü 3 gün önce aldım. İlk izlenimim harikaydı ama 2 gün sonra " +
                           "ekran donmaya başladı. Ses kalitesi iyi değil, çok cızırtılı. " +
                           "İade süreci çok kolaydı, onu beğendim. Genel olarak 2/10.";

        Console.WriteLine("=== Musteri Yorumu ===");
        Console.WriteLine(customerReview);
        Console.WriteLine();

        // 1. Yöntem: JSON prompt yapisi
        Console.WriteLine("YONTEM 1: JSON Prompt Yapisi");
        var jsonPrompt = $@"
Sistemi: Musteri yorumunu analiz et ve sadece JSON formatinda cevap ver.

Format:
{{
  ""memnuniyet_puani"": 1-10 arasi int,
  ""artilar"": [""en fazla 3 madde""],
  ""eksiler"": [""en fazla 3 madde""],
  ""ozet"": ""tek cumle"",
  ""iade_talebi"": true/false
}}

Yorum: {customerReview}
";
        Console.WriteLine(jsonPrompt);

        // Simüle edilmis model çiktisi
        var jsonOutput = @"{
  ""memnuniyet_puani"": 2,
  ""artilar"": [""Kargo hizli"", ""Iade sureci kolay""],
  ""eksiler"": [""Ekran donuyor"", ""Ses cizirtili"", ""Beklentileri karsilamadi""],
  ""ozet"": ""Tasarim guzel olsa da performans sorunlari nedeniyle hayal kirikligi."",
  ""iade_talebi"": true
}";

        Console.WriteLine("\nModel Ciktisi:");
        Console.WriteLine(jsonOutput);
        Console.WriteLine();

        // Post-processing ile dogrulama
        Console.WriteLine("YONTEM 2: Dogrulama ve Ayristirma");

        try
        {
            using var doc = JsonDocument.Parse(jsonOutput);
            var root = doc.RootElement;

            int puan = root.GetProperty("memnuniyet_puani").GetInt32();
            bool iade = root.GetProperty("iade_talebi").GetBoolean();

            if (puan < 1 || puan > 10)
                Console.WriteLine("  UYARI: Puan 1-10 arasinda olmali!");

            Console.WriteLine($"  Puan: {puan}/10 (gecerli: {puan >= 1 && puan <= 10})");
            Console.WriteLine($"  Iade: {iade}");
            Console.WriteLine("  JSON gecerli ve tutarli.");
        }
        catch (JsonException ex)
        {
            Console.WriteLine($"  HATA: Gecersiz JSON - {ex.Message}");
        }

        Console.WriteLine();

        // 3. Yöntem: XML
        Console.WriteLine("YONTEM 3: XML Etiketleri");

        var xmlOutput = @"<cevap>
  <memnuniyet>2</memnuniyet>
  <sorunlar>
    <sorun>Ekran donma sorunu</sorun>
    <sorun>Ses kalitesi dusuk</sorun>
  </sorunlar>
  <oneri>Yazilim guncellemesi ile ekran donmasi giderilebilir</oneri>
</cevap>";

        Console.WriteLine(xmlOutput);

        Console.WriteLine("\nRegex ile XML'den veri cekme:");
        var xmlMatch = Regex.Match(xmlOutput,
            @"<memnuniyet>(.*?)</memnuniyet>.*?<sorunlar>(.*?)</sorunlar>.*?<oneri>(.*?)</oneri>",
            RegexOptions.Singleline);

        if (xmlMatch.Success)
        {
            Console.WriteLine($"  Memnuniyet: {xmlMatch.Groups[1].Value}");
            Console.WriteLine($"  Oneri: {xmlMatch.Groups[3].Value}");

            var sorunMatches = Regex.Matches(xmlMatch.Groups[2].Value, @"<sorun>(.*?)</sorun>");
            Console.WriteLine("  Sorunlar:");
            foreach (Match m in sorunMatches)
                Console.WriteLine($"    - {m.Groups[1].Value}");
        }
    }
}
```

### Hangi Format Ne Zaman Kullanilmali?

- **JSON:** En yaygin. API entegrasyonu, veri analizi, her turlu yapısal çıktı için ideal.
- **XML:** Daha eski sistemlerle entegrasyon, belge odaklı çıktılar.
- **Markdown:** Okunabilirlik önemliyse (ama parse etmesi zor).
- **Özel delimiter:** `###START###` gibi. Basit regex ile parse edilebilir.

JSON'daki en büyük sorun, modelin bazen geçersiz JSON üretmesi. Eksik virgül, tırnak hatası, trailing comma... Bunun için en iyi çözüm: modele çok iyi bir şema vermek ve çıktıyı mutlaka validate etmek.

## Pratik İpuçları

Yazıyı bitirmeden önce, kendi deneyimlerimden derlediğim birkaç pratik tavsiye:

1. **Sıcaklık (temperature) ayarını unutmayın.** CoT için düşük temperature (0.0-0.3) daha tutarlı reasoning demek. Ama ToT'te dallarin çeşitlenmesi için biraz daha yüksek (0.5-0.7) iyi olabilir.

2. **Self-consistency.** Aynı soruyu 3-5 kere sorun, farklı CoT yollarından gelen cevapları oylayın. En çok tekrarlanan cevap genelde en doğrusu. Wang et al. 2022'de bunun GSM8K'de %74'ten %84'e çıkardığını göstermiş.

3. **ReAct'te hata yönetimi kritik.** API çagrıları başarısız olabilir, JSON bozuk gelebilir. Her Action adımında bir try-catch olmazsa olmaz.

4. **Context window'unuzu takip edin.** CoT ve özellikle ToT çok fazla token tüketir. Her adımda önceki tüm düşünceleri context'te tutmanız gerekir.

5. **Structured output'ta "output format constraints" kullanın.** Bazı modeller (GPT-4, Claude 3) artık native olarak JSON çıktı garantisi veriyor. Mümkünse bunu kullanın, yoksa post-processing ile idare edin.

## Kaynaklar

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 2022
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", 2022
- Kojima et al., "Large Language Models are Zero-Shot Reasoners", 2022
