# Gelişmiş Konular ve MLOps — AI Agents, Multi-Agent Sistemler, Model Deployment ve Monitoring

Beş haftadır sırayla tokenization'dan fine-tuning'e kadar geldik. Bu serinin son yazısına hoş geldiniz. Bu sefer işleri biraz daha karmaşık hale getiriyoruz: AI Agents, Multi-Agent sistemler, bir modeli gerçek dünyada nasıl deploy ederiz ve deploy ettikten sonra nasıl izleriz.

Kemerleri bağlayın, bu yazı biraz uzun olacak. Çünkü konu geniş — hem teorik hem pratik anlatmam gerekiyor.

---

## 1. AI Agent Nedir?

Bir language model'i alıp ona bir "araca" (tool) erişimi verdiğinizde, ona bir hedef tanımladığınızda ve kendi kendine adım adım ilerlemesine izin verdiğinizde — işte bu bir **agent** oluyor.

Düşünsenize, normal bir LLM kullanırken siz soru sorarsınız, o cevap verir. Bitmiştir. Agent modelinde ise model kendi kararlarını alır: "Şimdi bir API çağrısı yapmalıyım", "Bu sonucu bir dosyaya yazmalıyım", "Önce şu adımı yapıp sonucunu görmeliyim".

Bir agent'ın tipik akışı:

1. Kullanıcı bir hedef verir (task)
2. Agent mevcut durumu değerlendirir (perception)
3. Bir sonraki adımı planlar (reasoning)
4. Bir aracı çağırır (action / tool use)
5. Sonucu gözlemler (observation)
6. Hedefe ulaşana kadar 2-5 arasında döner

Bunu **ReAct** pattern'i olarak biliyoruz. Chain-of-Thought'un daha da uç hali.

C#'ta basit bir agent altyapısı şöyle görünebilir:

```csharp
// AgentDemo/Program.cs
// dotnet new console -n AgentDemo
// cd AgentDemo
// dotnet run

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AgentDemo;

// ---- Interface Tanımları ----

interface ITool
{
    string Name { get; }
    string Description { get; }
    Task<string> ExecuteAsync(string input);
}

interface IAgent
{
    string Name { get; }
    Task<string> RunAsync(string task, CancellationToken ct = default);
}

// ---- Tool Implementasyonları ----

class WeatherTool : ITool
{
    public string Name => "weather";
    public string Description => "Belirtilen şehir için hava durumu bilgisi verir. Input: şehir adı";

    public async Task<string> ExecuteAsync(string input)
    {
        // Gerçek hayatta bir API çağrısı yapardık
        await Task.Delay(200);
        var cities = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["istanbul"] = "18°C, Parçalı Bulutlu",
            ["ankara"] = "12°C, Açık",
            ["izmir"] = "22°C, Güneşli",
            ["antalya"] = "26°C, Açık"
        };

        return cities.TryGetValue(input.Trim(), out var weather)
            ? $"{input} için hava durumu: {weather}"
            : $"Üzgünüm, {input} için veri bulamadım.";
    }
}

class CalculatorTool : ITool
{
    public string Name => "calculator";
    public string Description => "Matematiksel ifadeleri hesaplar. Input: matematiksel ifade (ör: 2 + 3 * 4)";

    public Task<string> ExecuteAsync(string input)
    {
        try
        {
            // Basit bir expression evaluator — gerçek hayatta NCalc veya benzeri kullanılır
            var parts = input.Split(' ');
            if (parts.Length == 3
                && double.TryParse(parts[0], out var a)
                && double.TryParse(parts[2], out var b))
            {
                double result = parts[1] switch
                {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => b != 0 ? a / b : double.NaN,
                    _ => double.NaN
                };

                return Task.FromResult(double.IsNaN(result)
                    ? "Geçersiz işlem"
                    : $"{a} {parts[1]} {b} = {result}");
            }

            return Task.FromResult("İfadeyi anlayamadım. Örnek: 5 + 3");
        }
        catch
        {
            return Task.FromResult("Hesaplama sırasında hata oluştu.");
        }
    }
}

class NotepadTool : ITool
{
    public string Name => "notepad";
    public string Description => "Bir notu kaydeder veya okur. Input: 'kaydet: başlık | içerik' veya 'oku: başlık'";

    private static readonly Dictionary<string, string> Notes = new(StringComparer.OrdinalIgnoreCase);

    public Task<string> ExecuteAsync(string input)
    {
        if (input.StartsWith("kaydet:", StringComparison.OrdinalIgnoreCase))
        {
            var rest = input["kaydet:".Length..].Trim();
            var parts = rest.Split('|', 2);
            if (parts.Length == 2)
            {
                Notes[parts[0].Trim()] = parts[1].Trim();
                return Task.FromResult($"✅ '{parts[0].Trim()}' başlıklı not kaydedildi.");
            }
            return Task.FromResult("❌ Format: kaydet: başlık | içerik");
        }

        if (input.StartsWith("oku:", StringComparison.OrdinalIgnoreCase))
        {
            var title = input["oku:".Length..].Trim();
            return Notes.TryGetValue(title, out var content)
                ? Task.FromResult($"📝 {title}:\n{content}")
                : Task.FromResult($"❌ '{title}' başlıklı not bulunamadı.");
        }

        return Task.FromResult("Bilinmeyen komut. 'kaydet:' veya 'oku:' kullanın.");
    }
}

// ---- Agent ----

class ReActAgent : IAgent
{
    private readonly List<ITool> _tools;
    private readonly HttpClient _httpClient;
    public string Name { get; }

    public ReActAgent(string name, List<ITool> tools, HttpClient httpClient)
    {
        Name = name;
        _tools = tools;
        _httpClient = httpClient;
    }

    public async Task<string> RunAsync(string task, CancellationToken ct = default)
    {
        Console.WriteLine($"\n🤖 Agent '{Name}' görevi alıyor: \"{task}\"\n");
        Console.WriteLine(new string('-', 60));

        // Gerçek bir LLM API'sine istek atmak yerine simüle ediyoruz.
        // Ama yapı olarak ReAct döngüsünün mantığını gösterecek şekilde yapıyoruz.

        var context = new StringBuilder();
        context.AppendLine($"Görev: {task}");
        context.AppendLine();

        // Simüle ReAct döngüsü — 3 adım
        for (int step = 1; step <= 3; step++)
        {
            ct.ThrowIfCancellationRequested();

            Console.WriteLine($"\n📋 Adım {step}/3");

            // ---- THOUGHT ----
            string thought = step switch
            {
                1 => "Önce hava durumunu kontrol etmeliyim.",
                2 => "Hava durumuna göre bir not almalıyım.",
                3 => "Sıcaklık farkını hesaplamalıyım.",
                _ => "Devam ediyorum."
            };

            Console.WriteLine($"💭 Thought: {thought}");

            // ---- ACTION ----
            string? toolName = null;
            string? toolInput = null;

            if (step == 1)
            {
                toolName = "weather";
                toolInput = "İstanbul";
            }
            else if (step == 2)
            {
                toolName = "notepad";
                toolInput = "kaydet: Hava Durumu Raporu | İstanbul'da hava 18°C, parçalı bulutlu.";
            }
            else if (step == 3)
            {
                toolName = "calculator";
                toolInput = "26 - 18";
            }

            Console.WriteLine($"🔧 Action: {toolName}({toolInput})");

            // Tool'u bul ve çalıştır
            var tool = _tools.FirstOrDefault(t =>
                t.Name.Equals(toolName, StringComparison.OrdinalIgnoreCase));

            string observation = tool != null
                ? await tool.ExecuteAsync(toolInput ?? "")
                : $"Tool '{toolName}' bulunamadı.";

            Console.WriteLine($"👁️  Observation: {observation}");

            context.AppendLine($"Adım {step}:");
            context.AppendLine($"  Thought: {thought}");
            context.AppendLine($"  Action: {toolName}({toolInput})");
            context.AppendLine($"  Observation: {observation}");
            context.AppendLine();
        }

        // ---- FINAL ANSWER ----
        string finalAnswer = """
            📊 **Görev Tamamlandı!**
            
            İstanbul'da şu an hava 18°C ve parçalı bulutlu.
            Antalya ile arasında 8°C fark var.
            Notlarınıza kaydedildi.
            """;

        Console.WriteLine(new string('-', 60));
        Console.WriteLine($"✅ Final Answer:\n{finalAnswer}");

        return finalAnswer;
    }
}

// ---- Ana Uygulama (Dependency Injection benzeri yapı) ----

interface IAgentOrchestrator
{
    Task<string> ExecuteAsync(string task);
}

class AgentOrchestrator : IAgentOrchestrator
{
    private readonly IEnumerable<IAgent> _agents;
    private readonly ILogger _logger;

    public AgentOrchestrator(IEnumerable<IAgent> agents, ILogger logger)
    {
        _agents = agents;
        _logger = logger;
    }

    public async Task<string> ExecuteAsync(string task)
    {
        _logger.Log($"Orchestrator başlatılıyor. Mevcut agent sayısı: {_agents.Count()}");

        var results = new List<string>();
        foreach (var agent in _agents)
        {
            _logger.Log($"Agent '{agent.Name}' çalıştırılıyor...");
            var result = await agent.RunAsync(task);
            results.Add(result);
        }

        _logger.Log("Tüm agentlar tamamlandı.");
        return string.Join("\n\n---\n\n", results);
    }
}

interface ILogger
{
    void Log(string message);
}

class ConsoleLogger : ILogger
{
    public void Log(string message)
    {
        Console.WriteLine($"[LOG] {DateTime.Now:HH:mm:ss} — {message}");
    }
}

// ---- Entry Point ----

class Program
{
    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== AI Agent Demo — ReAct Döngüsü ===\n");

        // DI Container simülasyonu — manuel wiring
        using var httpClient = new HttpClient();

        var logger = new ConsoleLogger();

        var tools = new List<ITool>
        {
            new WeatherTool(),
            new CalculatorTool(),
            new NotepadTool()
        };

        var agent = new ReActAgent("Asistan-1", tools, httpClient);

        var orchestrator = new AgentOrchestrator(new[] { agent }, logger);

        // Görevi çalıştır
        var result = await orchestrator.ExecuteAsync(
            "İstanbul'da hava nasıl, bir not al ve Antalya ile sıcaklık farkını hesapla."
        );

        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("Nihai Çıktı:");
        Console.WriteLine(result);
        Console.WriteLine(new string('=', 60));
    }
}
```

Bu kodda ne yaptığımıza bakalım:

- **`ITool`**: Bir tool'un sahip olması gereken arayüz. Tool'lar tamamen birbirinden bağımsız.
- **`IAgent`**: Her agent'ın implement etmesi gereken arayüz.
- **`ReActAgent`**: Bir LLM'in Thought → Action → Observation döngüsünü simüle ediyor. Gerçek hayatta burada bir OpenAI/Anthropic API'sine istek atardık, prompt'a "şu araçların var, adım adım düşün" derdik.
- **`IAgentOrchestrator`**: Birden fazla agent'ı yönetmek için mediator benzeri bir katman. İleride çoklu agent yapısına geçince işe yarayacak.

Şimdi bunu çalıştırdığınızda göreceğiniz şey, agent'ın adım adım ilerlemesi. Her adımda önce düşünüyor ("ne yapmalıyım?"), sonra bir aracı çağırıyor ("weather tool'u kullan"), sonucu gözlemliyor ve bir sonraki adıma geçiyor.

İtiraf edeyim, ilk ReAct pattern'ini okuduğumda "ya bu sadece iyi prompt mühendisliği değil mi?" demiştim. Ama aslında öyle değil. Çünkü burada model, kendi eylemlerinin çıktılarını runtime'da görüyor ve bir sonraki adımını buna göre belirliyor. Bu, statik bir prompt'tan çok daha farklı bir şey.

---

## 2. Multi-Agent Sistemler

Tek agent iyi de, bazen bir işi birden fazla agent'a bölmek daha mantıklı oluyor. Mesela bir yazılım geliştirme sürecini düşünün:

- Bir agent kod yazsın
- Bir başkası test yazsın
- Bir diğeri code review yapsın
- Bir başkası da dokümantasyon hazırlasın

Her agent kendi uzmanlık alanında çalışır, diğerlerinin çıktısını alır ve üzerine ekler. Bu model, **AutoGen** (Microsoft), **CrewAI** gibi framework'lerin de temelini oluşturuyor.

C#'ta multi-agent bir yapı kurgulayalım:

```csharp
// MultiAgentDemo/Program.cs
// dotnet new console -n MultiAgentDemo
// dotnet run

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MultiAgentDemo;

interface IAgent
{
    string Role { get; }
    Task<AgentResult> ProcessAsync(AgentContext context);
}

record AgentResult(string AgentRole, string Output, bool Success);
record AgentContext(string Task, List<AgentResult> PreviousResults);

// ---- Uzman Agent'lar ----

class ResearcherAgent : IAgent
{
    public string Role => "Araştırmacı";

    public Task<AgentResult> ProcessAsync(AgentContext context)
    {
        Console.WriteLine($"🔍 {Role}: Konuyu araştırıyorum...");
        var output = context.Task switch
        {
            string t when t.Contains("RAG", StringComparison.OrdinalIgnoreCase) =>
                "RAG (Retrieval-Augmented Generation) hakkında araştırma yaptım. "
                + "En iyi chunk size 512 token civarı, overlap 64 token öneriliyor. "
                + "Cosine similarity en yaygın metrik. Reranker modelleri accuracy'yi %15-20 artırıyor.",
            string t when t.Contains("Transformer", StringComparison.OrdinalIgnoreCase) =>
                "Transformer mimarisinde multi-head attention 8-16 head ile çalışıyor. "
                + "Positional encoding olarak RoPER daha modern yaklaşım.",
            _ => $"'{context.Task}' konusunda genel bir araştırma yaptım. "
                + "Detaylı bilgi toplamam gerekiyor."
        };

        return Task.FromResult(new AgentResult(Role, output, true));
    }
}

class WriterAgent : IAgent
{
    public string Role => "Yazar";

    public Task<AgentResult> ProcessAsync(AgentContext context)
    {
        Console.WriteLine($"✍️ {Role}: İçerik yazıyorum...");

        var research = context.PreviousResults
            .FirstOrDefault(r => r.AgentRole == "Araştırmacı");

        if (research == null)
        {
            return Task.FromResult(new AgentResult(Role,
                "Araştırma yapılmadan yazı yazamam.", false));
        }

        var output = $"""
            # {context.Task} — Teknik Blog Yazısı

            ## Giriş
            Bu yazıda {context.Task.ToLower()} konusunu ele alacağız.

            ## Araştırma Bulguları
            {research.Output}

            ## Kod Örneği
            Aşağıda C# ile bir örnek göreceksiniz.

            ## Sonuç
            {context.Task} günümüzde çok önemli bir konu haline geldi.
            """;

        return Task.FromResult(new AgentResult(Role, output, true));
    }
}

class ReviewerAgent : IAgent
{
    public string Role => "Reviewer";

    public Task<AgentResult> ProcessAsync(AgentContext context)
    {
        Console.WriteLine($"🔎 {Role}: İçeriği inceliyorum...");

        var draft = context.PreviousResults
            .FirstOrDefault(r => r.AgentRole == "Yazar");

        if (draft == null)
        {
            return Task.FromResult(new AgentResult(Role, "Yazı bulunamadı.", false));
        }

        // Basit bir kalite kontrol simülasyonu
        var issues = new List<string>();
        if (!draft.Output.Contains("## Kod Örneği"))
            issues.Add("Kod örneği bölümü eksik");
        if (!draft.Output.Contains("## Sonuç"))
            issues.Add("Sonuç bölümü eksik");

        var output = issues.Count == 0
            ? "✅ İçerik onaylandı. Tüm bölümler mevcut."
            : $"⚠️ Düzeltme gerekiyor:\n{string.Join("\n", issues.Select(i => $"  - {i}"))}";

        return Task.FromResult(new AgentResult(Role, output, issues.Count == 0));
    }
}

// ---- Supervisor / Orchestrator ----

class Supervisor
{
    private readonly List<IAgent> _agents;
    private readonly List<AgentResult> _results = new();

    public Supervisor(List<IAgent> agents)
    {
        _agents = agents;
    }

    public async Task RunAsync(string task)
    {
        Console.WriteLine($"\n🎯 Görev: {task}\n");
        Console.WriteLine(new string('=', 50));

        var context = new AgentContext(task, []);

        foreach (var agent in _agents)
        {
            Console.WriteLine($"\n--- {agent.Role} çalışıyor ---");
            var result = await agent.ProcessAsync(context);
            _results.Add(result);

            Console.WriteLine($"Çıktı ({agent.Role}):");
            Console.WriteLine(result.Output.Truncate(150));

            context = context with { PreviousResults = _results.ToList() };

            // Eğer reviewer onaylamazsa döngüye sokabiliriz (detaylı simülasyon)
            if (!result.Success && agent is ReviewerAgent)
            {
                Console.WriteLine("\n⚠️ Revizyon gerekiyor!");
            }
        }

        Console.WriteLine("\n" + new string('=', 50));
        Console.WriteLine("✅ Multi-Agent Süreç Tamamlandı!");
    }
}

static class StringExtensions
{
    public static string Truncate(this string s, int maxChars) =>
        s.Length <= maxChars ? s : s[..maxChars] + "...";
}

class Program
{
    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== Multi-Agent Sistemi Demo ===\n");

        var agents = new List<IAgent>
        {
            new ResearcherAgent(),
            new WriterAgent(),
            new ReviewerAgent()
        };

        var supervisor = new Supervisor(agents);
        await supervisor.RunAsync("RAG ve Vektör Veri Tabanları");
    }
}
```

Burada asıl güzel olan şey şu: her agent bir sonrakinin input'unu etkiliyor. Araştırmacı bilgi topluyor, yazar o bilgiyle yazı yazıyor, reviewer da yazıyı kontrol ediyor. Tıpkı gerçek bir ekip gibi.

Multi-agent sistemlerde dikkat etmeniz gereken birkaç şey var:

- **Context window** — Her agent'a o ana kadarki tüm konuşmayı vermek zorundasınız. Context penceresi hızla dolabilir.
- **Coordination** — Agent'ların birbiriyle çakışmaması lazım. Aynı kaynağa yazmaya çalışan iki agent kaosa yol açar.
- **Error recovery** — Bir agent hata yaparsa ne olacak? Bunu handle etmezseniz tüm pipeline çöker.

Burada Supervisor pattern'i kullandık — merkezi bir kontrolcü var ve sırayı o belirliyor. Bir diğer yaklaşım da **decentralized** — agent'lar bir mesaj kuyruğu üzerinden haberleşiyor. Hangi yaklaşımı seçeceğiniz tamamen kullanım senaryosuna bağlı.

---

## 3. Model Deployment

Modeli eğittiniz, fine-tune ettiniz, test ettiniz. Şimdi bunu gerçek kullanıcılara açmanız lazım. İşte deployment bu noktada devreye giriyor.

Bir modeli deploy etmenin birkaç yolu var:

**a) REST API (En yaygın)**
Modeli bir HTTP servisi olarak ayağa kaldırıyorsunuz. İstek geliyor, model çalışıyor, cevap dönüyor.

**b) Batch Inference**
Toplu verileri işliyorsunuz. Gerçek zamanlı olması gerekmiyor. Genelde bir scheduler ile gece çalıştırılıyor.

**c) Edge / On-Device**
Modeli cihazın kendisinde çalıştırıyorsunuz. ONNX Runtime, TensorFlow Lite, ML.NET gibi teknolojilerle.

**d) Serverless**
Azure Functions, AWS Lambda gibi ortamlarda model inference'ı. Soğuk başlangıç sorunu var ama ölçeklenmesi kolay.

En yaygın yaklaşım REST API olduğu için C#'ta bir minimal API örneği yapalım. .NET 8'in minimal API'leri bu iş için biçilmiş kaftan:

```csharp
// InferenceAPI/Program.cs
// dotnet new webapi -n InferenceAPI --use-minimal-apis
// cd InferenceAPI
// dotnet run

using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Servisleri kaydediyoruz — DI Container
builder.Services.AddSingleton<IModelService, MockModelService>();
builder.Services.AddSingleton<IMetricsTracker, MetricsTracker>();
builder.Services.AddSingleton<IRateLimiter, RateLimiter>();

var app = builder.Build();

// Global middleware pipeline
app.UseMiddleware<RequestLoggingMiddleware>();
app.UseMiddleware<RequestValidationMiddleware>();

// Health check endpoint (load balancer'lar için)
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

// Metrics endpoint (Prometheus scrape edebilir)
app.MapGet("/metrics", (IMetricsTracker metrics) =>
{
    var snapshot = metrics.GetSnapshot();
    return Results.Ok(snapshot);
});

// Inference endpoint
app.MapPost("/v1/completions", async (
    CompletionRequest request,
    IModelService model,
    IMetricsTracker metrics,
    IRateLimiter rateLimiter,
    HttpContext httpContext) =>
{
    var clientIp = httpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown";

    // Rate limiting kontrolü
    if (!rateLimiter.TryConsume(clientIp))
    {
        metrics.RecordRateLimited();
        return Results.Json(
            new { error = "Rate limit exceeded. Try again later." },
            statusCode: 429);
    }

    // Input validation
    if (string.IsNullOrWhiteSpace(request.Prompt))
    {
        return Results.BadRequest(new { error = "Prompt cannot be empty." });
    }

    if (request.MaxTokens is < 1 or > 4096)
    {
        return Results.BadRequest(new { error = "max_tokens must be between 1 and 4096." });
    }

    // Inference
    var sw = System.Diagnostics.Stopwatch.StartNew();
    try
    {
        var result = await model.GenerateAsync(request.Prompt, request.MaxTokens ?? 256);
        sw.Stop();

        metrics.RecordInference(sw.ElapsedMilliseconds, result.GeneratedTokens);

        var response = new CompletionResponse
        {
            Id = Guid.NewGuid().ToString("N"),
            Model = "deepdive-demo-v1",
            Choices =
            [
                new Choice
                {
                    Index = 0,
                    Text = result.Text,
                    FinishReason = "stop"
                }
            ],
            Usage = new Usage
            {
                PromptTokens = CountTokens(request.Prompt),
                CompletionTokens = result.GeneratedTokens,
                TotalTokens = CountTokens(request.Prompt) + result.GeneratedTokens
            }
        };

        return Results.Ok(response);
    }
    catch (Exception ex)
    {
        sw.Stop();
        metrics.RecordError();
        Console.Error.WriteLine($"Inference hatası: {ex.Message}");
        return Results.Json(
            new { error = "Internal server error" },
            statusCode: 500);
    }
});

app.Run();

// ---- Servis Tanımları ----

interface IModelService
{
    Task<(string Text, int GeneratedTokens)> GenerateAsync(string prompt, int maxTokens);
}

interface IMetricsTracker
{
    void RecordInference(long latencyMs, int tokens);
    void RecordError();
    void RecordRateLimited();
    MetricsSnapshot GetSnapshot();
}

interface IRateLimiter
{
    bool TryConsume(string clientIp);
}

// ---- Implementasyonlar ----

class MockModelService : IModelService
{
    private static readonly Random Rng = new();

    public Task<(string Text, int GeneratedTokens)> GenerateAsync(string prompt, int maxTokens)
    {
        // Gerçek bir model çağrısı simülasyonu
        var simulatedResponse = $"""
            Prompt'unuzu aldım. İşte yanıt:
            
            Merhaba! "{prompt.Truncate(50)}" hakkında konuşalım.
            Bu bir simülasyon çıktısıdır. Gerçek bir model deploy ettiğinizde
            burada modelinizin ürettiği metin olacak.
            """;

        var tokenCount = Rng.Next(10, Math.Min(maxTokens, 100));
        return Task.FromResult((simulatedResponse, tokenCount));
    }
}

class MetricsTracker : IMetricsTracker
{
    private long _totalInferences;
    private long _totalErrors;
    private long _totalRateLimited;
    private long _totalLatencyMs;
    private long _totalTokens;

    public void RecordInference(long latencyMs, int tokens)
    {
        Interlocked.Increment(ref _totalInferences);
        Interlocked.Add(ref _totalLatencyMs, latencyMs);
        Interlocked.Add(ref _totalTokens, tokens);
    }

    public void RecordError()
    {
        Interlocked.Increment(ref _totalErrors);
    }

    public void RecordRateLimited()
    {
        Interlocked.Increment(ref _totalRateLimited);
    }

    public MetricsSnapshot GetSnapshot()
    {
        var inferences = Interlocked.Read(ref _totalInferences);
        return new MetricsSnapshot
        {
            TotalInferences = inferences,
            TotalErrors = Interlocked.Read(ref _totalErrors),
            TotalRateLimited = Interlocked.Read(ref _totalRateLimited),
            AvgLatencyMs = inferences > 0
                ? Interlocked.Read(ref _totalLatencyMs) / (double)inferences
                : 0,
            TotalTokensProcessed = Interlocked.Read(ref _totalTokens),
            UptimeSeconds = (long)(DateTime.UtcNow - _startTime).TotalSeconds
        };
    }

    private static readonly DateTime _startTime = DateTime.UtcNow;
}

class RateLimiter : IRateLimiter
{
    private readonly Dictionary<string, (int Count, DateTime WindowStart)> _clients = new();
    private readonly int _maxRequests = 60; // 60 requests per minute
    private readonly TimeSpan _window = TimeSpan.FromMinutes(1);

    public bool TryConsume(string clientIp)
    {
        lock (_clients)
        {
            var now = DateTime.UtcNow;
            if (!_clients.TryGetValue(clientIp, out var entry) ||
                now - entry.WindowStart > _window)
            {
                _clients[clientIp] = (1, now);
                return true;
            }

            if (entry.Count >= _maxRequests)
                return false;

            _clients[clientIp] = (entry.Count + 1, entry.WindowStart);
            return true;
        }
    }
}

// ---- Middleware'ler ----

class RequestLoggingMiddleware
{
    private readonly RequestDelegate _next;

    public RequestLoggingMiddleware(RequestDelegate next) => _next = next;

    public async Task InvokeAsync(HttpContext context)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Console.WriteLine($"[{DateTime.UtcNow:O}] → {context.Request.Method} {context.Request.Path}");

        await _next(context);

        sw.Stop();
        Console.WriteLine($"[{DateTime.UtcNow:O}] ← {context.Response.StatusCode} ({sw.ElapsedMilliseconds}ms)");
    }
}

class RequestValidationMiddleware
{
    private readonly RequestDelegate _next;

    public RequestValidationMiddleware(RequestDelegate next) => _next = next;

    public async Task InvokeAsync(HttpContext context)
    {
        if (context.Request.ContentLength > 1024 * 100) // 100KB limit
        {
            context.Response.StatusCode = 413;
            await context.Response.WriteAsync("Request too large.");
            return;
        }

        await _next(context);
    }
}

// ---- Data Transfer Objects (record'lar) ----

record CompletionRequest
{
    [JsonPropertyName("prompt")]
    public string Prompt { get; init; } = "";

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("temperature")]
    public double Temperature { get; init; } = 0.7;
}

record CompletionResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = "";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("choices")]
    public List<Choice> Choices { get; init; } = [];

    [JsonPropertyName("usage")]
    public Usage Usage { get; init; } = new();
}

record Choice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("text")]
    public string Text { get; init; } = "";

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; init; } = "";
}

record Usage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}

record MetricsSnapshot
{
    [JsonPropertyName("total_inferences")]
    public long TotalInferences { get; init; }

    [JsonPropertyName("total_errors")]
    public long TotalErrors { get; init; }

    [JsonPropertyName("total_rate_limited")]
    public long TotalRateLimited { get; init; }

    [JsonPropertyName("avg_latency_ms")]
    public double AvgLatencyMs { get; init; }

    [JsonPropertyName("total_tokens_processed")]
    public long TotalTokensProcessed { get; init; }

    [JsonPropertyName("uptime_seconds")]
    public long UptimeSeconds { get; init; }
}

// ---- Utility ----

static class StringExtensions
{
    public static string Truncate(this string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";
}

static int CountTokens(string text)
{
    // Çok kaba bir token sayacı — gerçekte tokenizer kullanılırdı
    return text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
}
```

Bu API'de şu özellikleri ekledim:

- **Health check**: Load balancer'ların ve Kubernetes probe'larının kullandığı endpoint
- **Metrics**: Prometheus'un scrape edebileceği bir endpoint. Buradan latency, error rate, throughput gibi metrikleri izleyebilirsiniz
- **Rate limiting**: Abuse'e karşı koruma. Her IP'den dakikada maksimum 60 istek
- **Request validation**: Gelen isteklerin geçerliliğini kontrol ediyoruz — boş prompt yok, token limiti aşılmasın
- **Structured logging**: Her isteğin ne zaman geldiğini, ne kadar sürdüğünü logluyoruz
- **Error handling**: Try-catch ile beklenmedik hataları yakalıyoruz ve 500 dönüyoruz

Bu yapının güzel tarafı, her şeyin **interface** üzerinden bağlanmış olması. MockModelService'i bir gün gerçek bir model çağrısıyla değiştirmek isterseniz, sadece yeni bir class yazıp DI'a kaydetmeniz yeterli.

---

## 4. Model Monitoring

Deploy ettiniz, güzel. Şimdi ne olacak? Modeliniz production'da çalışırken takip etmeniz gereken şeyler var:

**Performance Metrics:**
- **Latency**: P50, P95, P99 — istekler ne kadar sürede cevaplanıyor?
- **Throughput**: Saniyede kaç istek işleniyor?
- **Error Rate**: 500 hatalarının oranı
- **Token Throughput**: Saniyede üretilen token sayısı

**Data Quality Metrics (drift monitoring):**
- **Input drift**: Gelen prompt'ların dağılımı eğitim verisiyle uyuşuyor mu?
- **Output drift**: Modelin ürettiği çıktıların kalitesi değişiyor mu?
- **Concept drift**: Modelin doğru cevap verme oranı zamanla düşüyor mu?

**Business Metrics:**
- Kullanıcı memnuniyeti (thumbs up/down oranı)
- Dönüşüm oranı (eğer bir ürün tavsiye ediyorsa)
- Kullanım sıklığı

Bunları manuel takip etmek zor. Genelde Prometheus + Grafana + Alertmanager üçlüsü kullanılır. Ama temel bir monitoring servisi yazmak da hiç zor değil:

```csharp
// MonitoringDemo/Program.cs
// dotnet new console -n MonitoringDemo
// dotnet run

using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;

namespace MonitoringDemo;

// ---- Metric Types ----

interface IMetric
{
    string Name { get; }
    string GetValue();
}

class Counter : IMetric
{
    public string Name { get; }
    private long _value;
    
    public Counter(string name) => Name = name;
    public void Increment() => Interlocked.Increment(ref _value);
    public void Add(long n) => Interlocked.Add(ref _value, n);
    public string GetValue() => $"{Name} {{count=\"{Interlocked.Read(ref _value)}\"}}";
}

class Gauge : IMetric
{
    public string Name { get; }
    private long _value;
    
    public Gauge(string name) => Name = name;
    public void Set(long value) => Interlocked.Exchange(ref _value, value);
    public string GetValue() => $"{Name} {{gauge=\"{Interlocked.Read(ref _value)}\"}}";
}

class Histogram : IMetric
{
    public string Name { get; }
    private readonly ConcurrentBag<double> _values = new();
    private static readonly double[] Percentiles = [50, 95, 99];
    
    public Histogram(string name) => Name = name;
    
    public void Observe(double value) => _values.Add(value);
    
    public string GetValue()
    {
        if (_values.IsEmpty)
            return $"{Name} no_data";
            
        var sorted = _values.OrderBy(v => v).ToArray();
        var n = sorted.Length;
        
        var results = string.Join(" ", Percentiles.Select(p =>
        {
            var index = (int)Math.Ceiling(p / 100.0 * n) - 1;
            index = Math.Clamp(index, 0, n - 1);
            return $"p{p}={sorted[index]:F2}";
        }));
        
        return $"{Name} {{n={n}}} {results}";
    }
}

// ---- Monitoring Registry ----

class MetricRegistry
{
    private readonly ConcurrentDictionary<string, IMetric> _metrics = new();
    
    public Counter CreateCounter(string name) =>
        (Counter)_metrics.GetOrAdd(name, _ => new Counter(name));
    
    public Gauge CreateGauge(string name) =>
        (Gauge)_metrics.GetOrAdd(name, _ => new Gauge(name));
    
    public Histogram CreateHistogram(string name) =>
        (Histogram)_metrics.GetOrAdd(name, _ => new Histogram(name));
    
    public string Export() =>
        string.Join("\n", _metrics.Values.OrderBy(m => m.Name).Select(m => m.GetValue()));
}

// ---- Model Monitor ----

class ModelMonitor
{
    private readonly MetricRegistry _registry;
    private readonly Random _rng = new();
    
    public MetricRegistry Registry => _registry;
    
    // Anlık metrikler
    public Counter TotalRequests { get; }
    public Counter TotalErrors { get; }
    public Histogram LatencyMs { get; }
    public Gauge ActiveConnections { get; }
    public Counter TokensGenerated { get; }
    
    public ModelMonitor()
    {
        _registry = new MetricRegistry();
        
        TotalRequests = _registry.CreateCounter("model_requests_total");
        TotalErrors = _registry.CreateCounter("model_errors_total");
        LatencyMs = _registry.CreateHistogram("model_latency_ms");
        ActiveConnections = _registry.CreateGauge("model_active_connections");
        TokensGenerated = _registry.CreateCounter("model_tokens_generated");
    }
    
    public async Task SimulateTrafficAsync(int requests, CancellationToken ct)
    {
        var tasks = Enumerable.Range(0, requests)
            .Select(_ => SimulateOneRequestAsync(ct));
        
        await Task.WhenAll(tasks);
    }
    
    private async Task SimulateOneRequestAsync(CancellationToken ct)
    {
        ActiveConnections.Set(Interlocked.Increment(ref _activeCount));
        
        TotalRequests.Increment();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        
        try
        {
            // Random latency: 50-500ms (simüle inference)
            var latency = _rng.Next(50, 500);
            await Task.Delay(latency, ct);
            
            // Random hata: ~%5 error rate
            if (_rng.NextDouble() < 0.05)
            {
                TotalErrors.Increment();
                Console.WriteLine($"⚠️  Request failed (simulated error at {latency}ms)");
            }
            
            TokensGenerated.Add(_rng.Next(50, 500));
            LatencyMs.Observe(latency);
            
            Console.WriteLine($"✓ Request completed in {latency}ms");
        }
        catch (OperationCanceledException)
        {
            TotalErrors.Increment();
        }
        finally
        {
            ActiveConnections.Set(Interlocked.Decrement(ref _activeCount));
        }
    }
    
    private int _activeCount;
}

// ---- Alert Rules ----

class AlertRule
{
    public string Name { get; init; } = "";
    public string Description { get; init; } = "";
    public Func<MetricRegistry, bool> Condition { get; init; } = _ => false;
    public Action<string>? OnFire { get; init; }
}

class AlertManager
{
    private readonly List<AlertRule> _rules = new();
    private readonly HashSet<string> _firedAlerts = new();
    
    public void AddRule(AlertRule rule) => _rules.Add(rule);
    
    public void Evaluate(MetricRegistry registry)
    {
        foreach (var rule in _rules)
        {
            var isFiring = rule.Condition(registry);
            
            if (isFiring && _firedAlerts.Add(rule.Name))
            {
                // Alert ilk defa tetikleniyor
                rule.OnFire?.Invoke($"🔴 ALERT: {rule.Name} - {rule.Description}");
            }
            else if (!isFiring && _firedAlerts.Remove(rule.Name))
            {
                Console.WriteLine($"🟢 RESOLVED: {rule.Name}");
            }
        }
    }
}

// ---- Main ----

class Program
{
    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== Model Monitoring Demo ===\n");
        
        var monitor = new ModelMonitor();
        
        var alertManager = new AlertManager();
        alertManager.AddRule(new AlertRule
        {
            Name = "HighErrorRate",
            Description = "Error rate %5'i aştı",
            Condition = reg =>
            {
                var totalRequests = long.Parse(
                    ((Counter)reg.CreateCounter("model_requests_total"))
                    .GetValue().Split('"')[1]);
                var totalErrors = long.Parse(
                    ((Counter)reg.CreateCounter("model_errors_total"))
                    .GetValue().Split('"')[1]);
                    
                return totalRequests > 0 && (double)totalErrors / totalRequests > 0.05;
            },
            OnFire = msg => Console.WriteLine($"\n🚨 {msg}\n")
        });
        
        alertManager.AddRule(new AlertRule
        {
            Name = "HighLatency",
            Description = "P99 latency 400ms'i aştı",
            Condition = reg =>
            {
                var hist = (Histogram)reg.CreateHistogram("model_latency_ms");
                return hist.GetValue().Contains("p99=") 
                    && double.Parse(hist.GetValue().Split("p99=")[1].Split()[0]) > 400;
            },
            OnFire = msg => Console.WriteLine($"\n🚨 {msg}\n")
        });
        
        // Simüle trafik
        Console.WriteLine("📡 Simüle trafik başlatılıyor (100 istek)...\n");
        var cts = new CancellationTokenSource();
        
        await monitor.SimulateTrafficAsync(100, cts.Token);
        
        // Alert'leri değerlendir
        Console.WriteLine("\n--- Alert Değerlendirmesi ---");
        alertManager.Evaluate(monitor.Registry);
        
        // Metrikleri export et
        Console.WriteLine("\n--- Metrik Dökümü ---");
        Console.WriteLine(monitor.Registry.Export());
        
        Console.WriteLine("\n✅ Monitoring demo tamamlandı!");
    }
}
```

Burada üç temel metrik türü var:

- **Counter**: Sadece artabilen metrikler (toplam istek sayısı, hata sayısı, üretilen token sayısı). `Interlocked` ile thread-safe.
- **Gauge**: Anlık değer ölçen metrikler (aktif bağlantı sayısı, bellek kullanımı). Artıp azalabilir.
- **Histogram**: Değerlerin dağılımını tutan metrikler (gecikme süreleri). P50, P95, P99 hesaplar.

AlertManager ise belirlediğiniz kurallara göre uyarı gönderiyor. Error rate %5'i geçince veya P99 latency 400ms'i aşınca "alarm" çalıyor. Gerçek hayatta bu bildirimler Slack'e, e-postaya veya PagerDuty'ye giderdi.

---

## 5. Seriyi Toparlarken

Beş haftalık bir yolculuk oldu. Tokenization'dan başladık, attention mekanizmasına geçtik, prompting tekniklerini inceledik, RAG ile modellerin bilgi tabanını genişlettik, fine-tuning ile modelleri kendi verimize uyarladık ve son olarak agent'lar, deployment ve monitoring ile bu seriyi tamamladık.

Açıkçası bu son yazıyı yazarken hem heyecanlandım hem de biraz üzüldüm — güzel bir seriydi. Ama biten her şey gibi bunun da bir sonu olmalı.

Umarım bu yazılar size bir şeyler katmıştır. Ben yazarken çok şey öğrendim. Özellikle C# tarafında bu konseptleri uygulamak, "aa bunu böyle de anlatabilirim" dedirtti bazen.

Serideki tüm kodlara GitHub reposundan ulaşabilirsiniz. Bir sorunuz olursa veya bir konuda derinlemesine konuşmak isterseniz, her zaman buradayım.

Görüşmek üzere! 🚀

---

*Bu yazı DeepDiveAI serisinin 6. ve son yazısıdır.*