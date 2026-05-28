# Teknik Fine-Tuning — Parametre Verimliliği ve Model Quantization'i

Bir haftadır RAG ile modellerin bilgi tabanını genişletiyorduk. Bu hafta işi biraz daha derine taşıyoruz: modelin kendisini eğitmek (ince ayar yapmak) ve bunu mümkün olduğunca verimli yapmak.

Model fine-tuning'i dediğimiz şey aslında şu: büyük bir model alıyorsunuz (mesela 7B parametrelik bir model) ve onu kendi verinizle biraz daha eğitiyorsunuz. Sorun şu ki, full fine-tuning yaparsanız tüm parametreleri güncelliyorsunuz — bu hem çok pahalı hem de çok büyük bir model dosyasıyla sonuçlanıyor. 

İşte bu noktada **parameter-efficient fine-tuning (PEFT)** ve **quantization** devreye giriyor.

Ben bu yazıda size üç şey anlatacağım: **LoRA** ve **QLoRA** tam olarak nasıl çalışıyor (matematiksel olarak), **quantization** nedir ve bir modelin ağırlıklarını nasıl küçültürüz, ve bunların C# ile uygulamalarını göreceğiz.

Hazırsanız başlayalım.

## Fine-Tuning nedir, ne değildir?

Fine-tuning, önceden eğitilmiş (pre-trained) bir modeli alıp, spesifik bir görev için tekrar eğitme işlemidir. Transfer learning'in ta kendisi yani.

Bir modeli sıfırdan eğitmek... düşünmesi bile yorucu. Milyarlarca token, haftalarca GPU time, on binlerce dolarlık elektrik faturası. Fine-tuning ise çok daha makul. Zaten model dilin yapısını, grameri, dünya bilgisini öğrenmiş. Siz sadece "şu görevi de öğren" diyorsunuz.

**Full fine-tuning** dediğimiz yaklaşımda modelin tüm ağırlıkları güncellenir. Bu, 7B parametrelik bir model için 7B parametrenin hepsini güncellemek demek. GPU belleği açısından bakarsak, sadece ağırlıkları saklamak için ~14 GB (FP16'da) gerekiyor. Bir de gradient'lar, optimizer state'leri derken bu rakam rahatlıkla 3-4 katına çıkıyor. Bir 7B modeli full fine-tuning ile eğitmek 80 GB GPU'da bile zorlanabilir.

Peki alternatif? LoRA.

## LoRA — Low-Rank Adaptation

LoRA, 2021'de Microsoft tarafından yayınlanan bir yöntem ([LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)). Fikir inanılmaz basit ve zarif.

Normalde bir modelde ağırlıklar W ∈ ℝ^(d×k) şeklinde bir matristir. Fine-tuning yaparken W'yi W + ΔW olarak güncelleriz. ΔW de W ile aynı boyutta. 4096×4096'lık bir matris düşünün — 16 milyondan fazla parametre.

LoRA'nın tezi: ΔW aslında düşük ranklı bir matris. Yani ΔW = BA şeklinde iki küçük matrisin çarpımı olarak ifade edilebilir. B ∈ ℝ^(d×r) ve A ∈ ℝ^(r×k) olmak üzere, r ≪ min(d,k).

Şimdi bir örnekle görelim. Diyelim ki d=4096, k=4096 ve r=8 seçtik.

- Full ΔW: 4096×4096 = **16,777,216** parametre
- LoRA (r=8): 4096×8 + 8×4096 = **65,536** parametre

**Tam olarak 256 kat daha az parametre güncelliyoruz.** Ve inanılmaz olan şu: çoğu görevde bu kadar parametre yetiyor. Hatta r=4 veya r=8 ile full fine-tuning'e çok yakın sonuçlar alabiliyorsunuz.

Neden düşük rank? Çünkü modeller aşırı parametrize edilmiş durumda. Ağırlık güncellemeleri aslında düşük boyutlu bir manifold üzerinde yaşıyor. LoRA bu manifoldun yapısını kullanarak işi kolaylaştırıyor.

Forward pass sırasında ne oluyor? Normalde h = Wx. LoRA ile:

h = Wx + BAx

Yani orijinal ağırlıkların çıktısına LoRA adaptörünün katkısını ekliyorsunuz. İnference sırasında isterseniz W' = W + BA alıp tek matris olarak da hesaplayabilirsiniz — böylece ek latency olmaz.

Gelelim C# uygulamasına:

```csharp
// LoRADemo.cs
// dotnet new console -n LoRADemo
// dotnet add package MathNet.Numerics
// dotnet run

using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace LoRADemo;

class Program
{
    static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== LoRA: Low-Rank Adaptation Demo ===\n");

        // Rastgele bir orijinal ağırlık matrisi W: 10x8
        int d = 10; // giriş boyutu
        int k = 8;  // çıkış boyutu
        int r = 2;  // LoRA rank'i

        Console.WriteLine($"Model boyutları: d={d}, k={k}");
        Console.WriteLine($"LoRA rank'i: r={r}");
        Console.WriteLine($"Full fine-tuning güncellenecek: {d * k} parametre");
        Console.WriteLine($"LoRA ile güncellenecek: {d * r + r * k} parametre\n");

        var random = new Random(42);

        // W matrisi: 10x8 (orijinal, dondurulmuş ağırlıklar)
        var W = Matrix<double>.Build.Random(d, k, random.Next());
        Console.WriteLine("Orijinal ağırlık matrisi W (10x8):");
        PrintMatrix(W);

        // LoRA: W' = W + BA
        // B: 10x2,  A: 2x8
        // BA: 10x8 (düşük rank)
        var B = Matrix<double>.Build.Random(d, r, random.Next());
        var A = Matrix<double>.Build.Random(r, k, random.Next());

        Console.WriteLine("\nLoRA B matrisi (10x2):");
        PrintMatrix(B);
        Console.WriteLine("LoRA A matrisi (2x8):");
        PrintMatrix(A);

        // LoRA güncellemesini hesapla: ΔW = BA
        var deltaW = B * A;
        Console.WriteLine("\nLoRA güncellemesi ΔW = B×A (10x8):");
        PrintMatrix(deltaW);

        // Yeni ağırlıklar: W' = W + BA
        var WPrime = W + deltaW;
        Console.WriteLine("\nGüncellenmiş ağırlıklar W' = W + ΔW (10x8):");
        PrintMatrix(WPrime);

        // Forward pass: h = Wx + BAx  (veya h = W'x)
        Console.WriteLine("\n--- Forward Pass ---");
        var x = Vector<double>.Build.Random(k, random.Next());
        Console.WriteLine($"Giriş vektörü x (boyut {k}):");
        Console.WriteLine(x.ToVectorString());

        // Orijinal modelin çıktısı
        var hOriginal = W * x;
        Console.WriteLine($"\nOrijinal model çıktısı h = Wx:");
        Console.WriteLine(hOriginal.ToVectorString());

        // LoRA katkısı
        var loraContribution = B * (A * x);  // BAx = B(Ax) — daha verimli
        Console.WriteLine($"\nLoRA katkısı BAx:");
        Console.WriteLine(loraContribution.ToVectorString());

        // Güncellenmiş model çıktısı
        var hLora = WPrime * x;
        Console.WriteLine($"\nLoRA modeli çıktısı h = W'x:");
        Console.WriteLine(hLora.ToVectorString());

        // Doğrulama: W'x = Wx + BAx
        var expected = hOriginal + loraContribution;
        Console.WriteLine($"\nDoğrulama: Wx + BAx == W'x ? {hLora.AlmostEqual(expected, 1e-10)}");

        // Bellek kullanımı karşılaştırması
        Console.WriteLine("\n=== Bellek Kullanımı Karşılaştırması ===");
        long fullParams = d * k;
        long loraParams = d * r + r * k;
        double savingsPercent = (1.0 - (double)loraParams / fullParams) * 100;
        Console.WriteLine($"Full fine-tuning parametre sayısı: {fullParams}");
        Console.WriteLine($"LoRA parametre sayısı: {loraParams}");
        Console.WriteLine($"Bellek tasarrufu: %{savingsPercent:F2}");
        Console.WriteLine($"\nDiyelim ki 7B parametrelik bir modelde 512 boyutundaki attention katmanlarına LoRA uyguluyoruz:");
        Console.WriteLine($"  Full: 512×512 = 262,144 parametre/katman");
        Console.WriteLine($"  LoRA (r=8): 512×8 + 8×512 = 8,192 parametre/katman");
        Console.WriteLine($"  Azalma: %{(1.0 - 8192.0/262144.0) * 100:F2}");
    }

    static void PrintMatrix(Matrix<double> m)
    {
        var rows = m.RowCount;
        var cols = m.ColumnCount;
        for (int i = 0; i < Math.Min(rows, 5); i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{m[i, j],8:F4}");
                if (j < cols - 1) Console.Write(", ");
            }
            Console.WriteLine(" ]");
        }
        if (rows > 5)
            Console.WriteLine("  ...");
        Console.WriteLine();
    }
}

// Vector for print helper
public static class VectorExtensions
{
    public static string ToVectorString(this Vector<double> v)
    {
        return "[" + string.Join(", ", v.Select(x => $"{x:F4}").Take(8))
            + (v.Count > 8 ? ", ..." : "") + "]";
    }
}
```

Bu kodda ne yaptığımızı adım adım açıklayayım. Önce rastgele bir W matrisi oluşturduk. Sonra B ve A matrislerini oluşturup ΔW = BA'yı hesapladık. Son adımda W' = W + ΔW ile güncellenmiş modeli elde ettik.

Dikkat ettiyseniz forward pass'te B×(A×x) şeklinde bir hesaplama yaptık. Bunun sebebi verimlilik — B×(A×x) önce vektör-matriks çarpımı yapıp (küçük r sayesinde) sonra tekrar matris-vektör çarpımı yapıyor. Full (BA)x hesaplamaktan çok daha hızlı.

### LoRA nereye uygulanır?

Orijinal makalede LoRA sadece **attention katmanlarının** Wq, Wk, Wv ve Wo matrislerine uygulanıyor. Neden? Çünkü attention en çok parametrenin olduğu yer ve aynı zamanda en kritik kısım. Sonraki çalışmalar LoRA'nın her yere uygulanabileceğini gösterdi — FFN katmanları, bias terimleri, vs.

Pratikte en çok işe yarayan yerler:
- Attention projeksiyonları (Q, K, V, O) — standart
- FFN (Feed-Forward Network) katmanları — özellikle domain adaptation'da
- LayerNorm bias'ları — ama bunlar çok az parametre barındırıyor
- Embedding katmanları — dikkatli olmak lazım, overfit riski var

## Quantization: Ağırlıkları Küçültmek

Şimdi konuyu biraz daha ilginçleştirelim. Quantization, model ağırlıklarının hassasiyetini düşürerek hem boyutlarını küçültmek hem de hesaplamayı hızlandırmak anlamına geliyor.

Normalde bir model FP32 (32-bit floating point) ile eğitilir. 7B parametre = 28 GB. Çoğu GPU'ya sığmaz. FP16'ya düşürürsek 14 GB olur — daha iyi ama hâlâ büyük. INT8'e düşürürsek sadece 7 GB. Hatta 4-bit quantization ile 3.5 GB'a kadar inebiliyoruz.

Peki quantization nasıl çalışıyor?

### Symmetric Quantization

En basit yöntem. FP32/FP16 değerlerinizi [-127, 127] aralığına scale ediyorsunuz. Bir scale faktörü hesaplıyorsunuz: s = 127 / max(|w|). Sonra q = round(w × s).

Dequantization ise w ≈ q / s.

### Asymmetric Quantization

FP16 değerlerin min/max'ını alıp [0, 255] aralığına map ediyorsunuz. Zero-point (sıfır noktası kayması) da hesaba katıyorsunuz. Daha esnek ama biraz daha karmaşık.

İtiraf edeyim ki ilk okuduğumda kafam karışmıştı. "Nasıl yani 3.14 sayısını bir tamsayıya indirgeyip işlem yapıyoruz?" diye düşünmüştüm. Oysa mantık çok basit: büyük değerler küçük bir aralığa map edilirken bilgi kaybı oluyor ama modelin genel başarımı çok fazla etkilenmiyor. Sinir ağları aslında şaşırtıcı derecede gürbüzdür (robust). Küçük yuvarlama hatalarına toleranslıdırlar.

Gelelim C# kodumuza:

```csharp
// QuantizationDemo.cs
// dotnet new console -n QuantizationDemo
// dotnet run

using System;

namespace QuantizationDemo;

class Program
{
    static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== Model Quantization Demo ===\n");

        // Rastgele bir ağırlık matrisi simüle edelim (FP32)
        var random = new Random(42);
        var weights = new float[4, 6]; // 4x6'lık küçük bir ağırlık matrisi

        Console.WriteLine("Orijinal FP32 Ağırlık Matrisi (4x6):");
        for (int i = 0; i < 4; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < 6; j++)
            {
                weights[i, j] = (float)(random.NextDouble() * 4 - 2); // [-2, 2] aralığı
                Console.Write($"{weights[i, j],8:F4}");
                if (j < 5) Console.Write(", ");
            }
            Console.WriteLine(" ]");
        }

        Console.WriteLine("\n============================================");
        Console.WriteLine("FP16'ya Dönüşüm (Half Precision)");
        Console.WriteLine("============================================");

        var fp16Weights = new float[4, 6];
        Console.WriteLine("FP16 Simülasyonu (mantis: 10 bit, üs: 5 bit):");
        for (int i = 0; i < 4; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < 6; j++)
            {
                fp16Weights[i, j] = SimulateFP16(weights[i, j]);
                Console.Write($"{fp16Weights[i, j],8:F4}");
                if (j < 5) Console.Write(", ");
            }
            Console.WriteLine(" ]");
        }

        float mseFp16 = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                mseFp16 += MathF.Pow(weights[i, j] - fp16Weights[i, j], 2);
        mseFp16 /= 24;
        Console.WriteLine($"\nFP16 MSE Hatası: {mseFp16:F8}");

        Console.WriteLine("\n============================================");
        Console.WriteLine("INT8 Symmetric Quantization");
        Console.WriteLine("============================================");

        float maxAbs = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                maxAbs = Math.Max(maxAbs, Math.Abs(weights[i, j]));

        float scaleInt8 = 127.0f / maxAbs;
        Console.WriteLine($"Max |w| = {maxAbs:F4}");
        Console.WriteLine($"Scale factor (FP32→INT8): s = 127 / {maxAbs:F4} = {scaleInt8:F4}");

        var int8Weights = new sbyte[4, 6];
        Console.WriteLine("\nINT8 Quantize Edilmiş Ağırlıklar:");
        for (int i = 0; i < 4; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < 6; j++)
            {
                int8Weights[i, j] = (sbyte)Math.Round(weights[i, j] * scaleInt8);
                int8Weights[i, j] = Math.Clamp(int8Weights[i, j], (sbyte)-128, (sbyte)127);
                Console.Write($"{int8Weights[i, j],5}");
                if (j < 5) Console.Write(", ");
            }
            Console.WriteLine(" ]");
        }

        Console.WriteLine("\nDequantize Edilmiş Ağırlıklar (q / s):");
        var dequantized = new float[4, 6];
        float mseInt8 = 0;
        for (int i = 0; i < 4; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < 6; j++)
            {
                dequantized[i, j] = int8Weights[i, j] / scaleInt8;
                Console.Write($"{dequantized[i, j],8:F4}");
                if (j < 5) Console.Write(", ");
                mseInt8 += MathF.Pow(weights[i, j] - dequantized[i, j], 2);
            }
            Console.WriteLine(" ]");
        }
        mseInt8 /= 24;
        Console.WriteLine($"\nINT8 MSE Hatası: {mseInt8:F6}");
        Console.WriteLine($"FP32→INT8 Bilgi Kaybı (relative): %{(mseInt8 / (mseFp16 + 0.00001f)) * 100:F2}");

        Console.WriteLine("\n============================================");
        Console.WriteLine("INT4 Quantization (2^4 = 16 seviye)");
        Console.WriteLine("============================================");

        float scaleInt4 = 7.0f / maxAbs;
        Console.WriteLine($"Scale factor (FP32→INT4): s = 7 / {maxAbs:F4} = {scaleInt4:F4}");

        int mseInt4 = 0;
        Console.WriteLine("\nINT4 Quantize Edilmiş (dequantize edilmiş hali):");
        for (int i = 0; i < 4; i++)
        {
            Console.Write("[ ");
            for (int j = 0; j < 6; j++)
            {
                int quantized = (int)Math.Round(weights[i, j] * scaleInt4);
                quantized = Math.Clamp(quantized, -8, 7);
                float reconstructed = quantized / scaleInt4;
                Console.Write($"{reconstructed,8:F4}");
                if (j < 5) Console.Write(", ");
                mseInt4 += (int)(MathF.Pow(weights[i, j] - reconstructed, 2) * 10000);
            }
            Console.WriteLine(" ]");
        }

        Console.WriteLine("\n============================================");
        Console.WriteLine("Quantized Matris Çarpımı Demo");
        Console.WriteLine("============================================");

        var input = new float[6];
        for (int j = 0; j < 6; j++)
            input[j] = (float)(random.NextDouble() * 2 - 1);

        Console.WriteLine("\nGiriş vektörü x:");
        Console.WriteLine("[" + string.Join(", ", input.Select(x => $"{x,8:F4}")) + "]");

        // FP32 çarpımı
        var outputFp32 = new float[4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                outputFp32[i] += weights[i, j] * input[j];

        // INT8 quantized çarpım
        var outputInt8 = new float[4];
        for (int i = 0; i < 4; i++)
        {
            int sum = 0;
            for (int j = 0; j < 6; j++)
                sum += int8Weights[i, j] * (int)Math.Round(input[j] * 100);
            outputInt8[i] = sum / (scaleInt8 * 100);
        }

        Console.WriteLine("\nFP32 Model Çıktısı:");
        Console.WriteLine("[" + string.Join(", ", outputFp32.Select(x => $"{x,8:F4}")) + "]");

        Console.WriteLine("\nINT8 Quantized Model Çıktısı:");
        Console.WriteLine("[" + string.Join(", ", outputInt8.Select(x => $"{x,8:F4}")) + "]");

        Console.WriteLine("\nFark (FP32 - INT8):");
        Console.WriteLine("[" + string.Join(", ", outputFp32.Zip(outputInt8, (a, b) => $"{a-b,8:F4}")) + "]");

        Console.WriteLine("\n=== Bellek Karşılaştırması ===");
        Console.WriteLine($"FP32: 24 değer × 4 byte = {24 * 4} bytes");
        Console.WriteLine($"FP16: 24 değer × 2 byte = {24 * 2} bytes");
        Console.WriteLine($"INT8: 24 değer × 1 byte = {24 * 1} bytes");
        Console.WriteLine($"INT4: 24 değer × 0.5 byte = {24 / 2} bytes");
        Console.WriteLine();
        Console.WriteLine($"7B model için:");
        Console.WriteLine($"  FP32: {7.0 * 4:F0} GB");
        Console.WriteLine($"  FP16: {7.0 * 2:F0} GB");
        Console.WriteLine($"  INT8: {7.0 * 1:F0} GB");
        Console.WriteLine($"  INT4: ~{7.0 * 0.5:F1} GB (GGUF formatı ile)");
    }

    static float SimulateFP16(float value)
    {
        if (value == 0) return 0;
        float scale = MathF.Pow(2, MathF.Floor(MathF.Log2(MathF.Abs(value))) - 10);
        return MathF.Round(value / scale) * scale;
    }
}
```

Bunu çalıştırdığınızda şöyle bir şey göreceksiniz: FP16'ya geçince çok az hata oluyor (genelde ihmal edilebilir seviyede). INT8'de hata biraz artıyor ama modelin genel başarımı genelde %1-2'den fazla düşmüyor. INT4'te ise daha belirgin kayıplar olabiliyor.

Ama işin ilginci, quantization-aware training (QAT) ile bu kayıpları çok aza indirebiliyorsunuz. Yani quantize edilmiş modeli biraz daha eğiterek quantization hatalarını telafi ediyorsunuz.

## QLoRA: Quantized LoRA

QLoRA, 2023'te yayınlanan ve LoRA ile 4-bit quantization'ı birleştiren bir teknik ([makale](https://arxiv.org/abs/2305.14314)). Fikir şu:

1. Model ağırlıklarını 4-bit'e quantize et
2. LoRA adaptörlerini (B ve A) FP16'da tut
3. Forward pass: 4-bit ağırlıkları o an için FP16'ya dequantize et, LoRA katkısını ekle
4. Sadece LoRA parametrelerine gradient hesapla

Bu sayede bir 65B modeli (normalde 130 GB FP32) sadece 48 GB GPU belleğiyle fine-tune edilebiliyor. Cidden etkileyici.

QLoRA'nın iki önemli tekniği daha var:

**NF4 (NormalFloat4)**: Normal dağılıma sahip ağırlıklar için optimize edilmiş bir 4-bit quantization yöntemi. Standart INT4'ten daha iyi çalışıyor çünkü quantization seviyelerini normal dağılıma göre ayarlıyor. Yani {-8, -7, ..., 7} yerine, değerlerin yoğun olduğu bölgelerde daha sık, seyrek olduğu bölgelerde daha seyrek seviye kullanıyor.

**Double Quantization**: Scale faktörlerini de quantize ediyor. Normalde her blok için bir scale faktörü saklarsınız (FP32). Double quantization'da bu scale faktörlerini de INT8'e çeviriyorsunuz.

Şimdi QLoRA'nın mantığını C# ile göstereyim:

```csharp
// QLoRADemo.cs
// dotnet new console -n QLoRADemo
// dotnet add package MathNet.Numerics
// dotnet run

using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;

namespace QLoRADemo;

class Program
{
    static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("=== QLoRA: Quantized LoRA Demo ===\n");

        // Bir linear katman simüle edelim: 256 giriş, 256 çıkış
        int d = 256;
        int k = 256;
        int r = 4;

        var random = new Random(42);

        Console.WriteLine("Adım 1: 4-bit NormalFloat (NF4) Quantization");
        Console.WriteLine("=============================================");

        // Orijinal ağırlıklar (normal dağılım)
        var W_full = Matrix<double>.Build.Random(d, k, new Normal(0, 0.5, random));
        var flatWeights = W_full.Enumerate().ToArray();

        // NF4 quantization seviyeleri (normal dağılımın ters CDF'sinden)
        var nf4Levels = new double[16];
        for (int i = 0; i < 16; i++)
        {
            double p = (i + 0.5) / 16.0;
            nf4Levels[i] = Normal.InvCDF(0, 1, p);
        }

        Console.WriteLine("NF4 Seviyeleri (normale göre optimize edilmiş):");
        Console.WriteLine(string.Join(", ", nf4Levels.Select(x => $"{x,8:F4}")));

        // Blok bazında quantization (block_size = 64)
        int blockSize = 64;
        int numBlocks = (d * k + blockSize - 1) / blockSize;
        var quantizedWeights = new byte[d * k / 2];
        var scaleFactors = new double[numBlocks];

        Console.WriteLine($"\nBlok boyutu: {blockSize}, Blok sayısı: {numBlocks}");

        for (int block = 0; block < numBlocks; block++)
        {
            int start = block * blockSize;
            int end = Math.Min(start + blockSize, d * k);

            double blockMax = 0;
            for (int idx = start; idx < end; idx++)
                blockMax = Math.Max(blockMax, Math.Abs(flatWeights[idx]));

            double scale = blockMax > 1e-10 ? blockMax / nf4Levels[15] : 1.0;
            scaleFactors[block] = scale;

            for (int idx = start; idx < end; idx++)
            {
                double normalized = flatWeights[idx] / scale;
                int nearestLevel = 0;
                double minDist = double.MaxValue;

                for (int l = 0; l < 16; l++)
                {
                    double dist = Math.Abs(normalized - nf4Levels[l]);
                    if (dist < minDist) { minDist = dist; nearestLevel = l; }
                }

                int bytePos = idx / 2;
                if (idx % 2 == 0)
                    quantizedWeights[bytePos] = (byte)((quantizedWeights[bytePos] & 0xF0) | (byte)nearestLevel);
                else
                    quantizedWeights[bytePos] = (byte)((quantizedWeights[bytePos] & 0x0F) | (byte)(nearestLevel << 4));
            }
        }

        Console.WriteLine("Adım 2: Double Quantization — Scale Faktörlerini de Quantize Et");
        Console.WriteLine("===============================================");

        double dqScale = 127.0 / scaleFactors.Max();
        var dqScaled = scaleFactors.Select(s => (sbyte)Math.Round(s * dqScale)).ToArray();

        Console.WriteLine("Scale faktörleri (FP32):");
        Console.WriteLine(string.Join(", ", scaleFactors.Take(8).Select(x => $"{x:F6}")) + " ...");
        Console.WriteLine("\nDouble quantize edilmiş scale faktörleri (INT8):");
        Console.WriteLine(string.Join(", ", dqScaled.Take(8).Select(x => $"{x,4}")) + " ...");

        double dqMemoryFp32 = scaleFactors.Length * 4;
        double dqMemoryInt8 = dqScaled.Length * 1;
        Console.WriteLine($"\nScale faktörleri belleği (FP32): {dqMemoryFp32} bytes");
        Console.WriteLine($"Scale faktörleri belleği (INT8): {dqMemoryInt8} bytes");
        Console.WriteLine($"Double quantization tasarrufu: %{(1.0 - dqMemoryInt8 / dqMemoryFp32) * 100:F1}");

        Console.WriteLine("\nAdım 3: LoRA Adaptörlerini Hesapla (FP16'da)");
        Console.WriteLine("============================================");

        // LoRA adaptörleri — FP16'da tutulur
        var B = Matrix<double>.Build.Random(d, r, new Normal(0, 0.01, random));
        var A = Matrix<double>.Build.Random(r, k, new Normal(0, 0.01, random));
        var loraDelta = B * A;

        Console.WriteLine($"LoRA B matrisi: {d}x{r} = {d * r} parametre");
        Console.WriteLine($"LoRA A matrisi: {r}x{k} = {r * k} parametre");
        Console.WriteLine($"Toplam LoRA parametreleri (FP16): {(d * r + r * k) * 2} bytes");

        Console.WriteLine("\nAdım 4: Bellek Karşılaştırması");
        Console.WriteLine("============================================");

        long totalParams = d * k;
        double fp32Mem = totalParams * 4;
        double fp16Mem = totalParams * 2;
        double nf4Mem = totalParams * 0.5 + dqMemoryInt8;
        double loraMem = (d * r + r * k) * 2;
        double qloraMem = nf4Mem + loraMem;

        Console.WriteLine($"Model parametreleri: {totalParams:N0}");
        Console.WriteLine($"  FP32: {fp32Mem / 1024 / 1024:F2} MB");
        Console.WriteLine($"  FP16: {fp16Mem / 1024 / 1024:F2} MB");
        Console.WriteLine($"  NF4:  {nf4Mem / 1024 / 1024:F2} MB");
        Console.WriteLine($"  LoRA: {loraMem / 1024 / 1024:F2} MB");
        Console.WriteLine($"  QLoRA: {qloraMem / 1024 / 1024:F2} MB");
        Console.WriteLine($"  QLoRA vs FP32: %{(1.0 - qloraMem / fp32Mem) * 100:F1} daha az bellek");
    }
}
```

## Pratikte Fine-Tuning Süreci

Şimdi anlatacaklarımı bir de pratik bağlama oturtalım. Diyelim ki bir Türkçe soru-cevap modeli fine-tune etmek istiyorsunuz. Ne yapardınız?

**1. Veri Toplama**: Modelin öğrenmesini istediğiniz görev için veri toplarsınız. Soru-cevap için binlerce soru-cevap çifti. Format genelde şöyle:

```
```
Soru: Türkiye'nin başkenti neresidir?
Cevap: Ankara'dır.

Soru: Einstein hangi yıl Nobel ödülü almıştır?
Cevap: 1921 yılında Nobel Fizik Ödülü'nü almıştır.
```

**2. Format Kararı**: Hangi formatı kullanacaksınız? En yaygın olanlar:
- **Chat template**: `<|im_start|>user\nSoru\n<|im_end|>\n<|im_start|>assistant\nCevap\n<|im_end|>`
- **Instruction format**: `### Instruction:\nSoru\n\n### Response:\nCevap`

**3. Hiperparametre Seçimi**: LoRA rank'i (genelde 8-64 arası), learning rate (1e-4 civarı), batch size (mümkün olduğunca büyük), epoch sayısı (2-3 genelde yeterli).

**4. Eğitim**: QLoRA ile tek bir 24 GB GPU'da 7B model rahatça fine-tune edilebilir.

**5. Değerlendirme**: Eğitimden sonra modeli test setinde değerlendirirsiniz. Ama unutmayın: fine-tuning yaparken modelin eski yeteneklerini unutması (catastrophic forgetting) riski var. Bu yüzden evaluation'da sadece yeni görevi değil, orijinal görevleri de test etmek önemli.

## Adapters ve Model Birleştirme

LoRA'nın en güzel yanlarından biri: adaptörler ayrı dosyalar olarak saklanır. Yani aynı temel model için binlerce farklı adaptörünüz olabilir, sadece ~10-50 MB yer kaplarlar. İstediğiniz adaptörü takıp çıkarabilirsiniz.

Üstelik birden fazla LoRA adaptörünü birleştirebilir (merge) veya aralarında ağırlıklı ortalama alabilirsiniz. Bu, model birleştirme (model merging) tekniklerinin temelini oluşturur:

W_toplam = W + α₁·ΔW₁ + α₂·ΔW₂

Burada α'lar her adaptörün katkı ağırlığı. İki farklı görevde fine-tune edilmiş adaptörü birleştirip tek bir model elde edebilirsiniz. Modelleri birleştirme konusu başlı başına bir yazılık konu — belki ileride onu da yazarım.

## Özet

Fine-tuning, büyük modelleri kendi verinize uyarlamanın en etkili yolu. Full fine-tuning pahalı ve gereksiz yere büyük. LoRA ile parametrelerin sadece %0.1-1'ini güncelleyerek full fine-tuning'e çok yakın sonuçlar alabiliyorsunuz.

Quantization ise modelleri daha küçük ve hızlı hale getiriyor. FP16 günlük kullanım için ideal, INT8 çoğu senaryoda yeterli, INT4 ise edge cihazlar ve büyük modeller için can kurtarıcı.

QLoRA bu ikisini birleştiriyor: 4-bit quantize edilmiş bir modeli LoRA ile fine-tune ediyorsunuz. Sonuç? 65B parametrelik bir modeli 48 GB GPU'da fine-tune edebiliyorsunuz. Bu, birkaç yıl önce imkansız denebilecek bir şeydi.


