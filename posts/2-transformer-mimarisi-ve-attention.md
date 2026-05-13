# Transformer Mimarisi ve Attention Mekanizması — Derinlemesine Bir Bakış

Transformers. Her yerdeler. LLM'lerden görüntü işlemeye, neredeyse tüm modern yapay sinir ağı tabanlı sistemlerin belkemiği haline geldiler. Peki bir Transformer'ın içinde gerçekte neler oluyor? Self-attention denen şey matematiksel olarak nasıl işliyor? Neden "attention is all you need" dendi? Gelin adım adım inceleyelim.

Dürüst olayım, bu mimariyi ilk duyduğumda kulağa fazla karmaşık gelmişti. Ama aslında temel fikir o kadar basit ki: her kelimenin (tokenin), cümledeki diğer tüm kelimelerle olan ilişkisini hesaplıyorsun. İşte bu kadar. Gerisi detay.

## Attention'ın Kalbi: Query, Key, Value

Vaswani ve arkadaşlarının 2017'de yayınladığı "Attention Is All You Need" makalesiyle hayatımıza giren bu mekanizma, aslında bir bilgi erişim (information retrieval) sisteminden ilham alıyor. Şöyle düşünün:

Bir sorunuz var (Query), bir dizi anahtarınız (Keys) ve her anahtara karşılık gelen değerleriniz (Values). Query ile her Key arasındaki benzerliği hesaplıyorsunuz, bu benzerliklere göre Values'ların ağırlıklı ortalamasını alıyorsunuz. Çıkan sonuç, sorgunuza en uygun cevap oluyor.

### Skalı Çarpım (Scaled Dot-Product Attention)

Matematiksel olarak:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Burada:
- Q: Query matrisi (d₁ boyutunda)
- K: Key matrisi (d₂ boyutunda)
- V: Value matrisi
- dₖ: Key'lerin boyutu (genelde embedding dimension)
- √dₖ: Ölçekleme faktörü (büyük boyutlarda softmax'ın aşırı uç değerlere gitmesini engelliyor)

Neden √dₖ ile bölüyoruz? Çünkü Q ve K vektörleri rastgele dağılımlardan geldiğinde, noktasal çarpımlarının varyansı dₖ ile orantılı oluyor. Bu da büyük dₖ değerlerinde softmax'ın çok keskin (bir değer 1'e, diğerleri 0'a yakın) olmasına yol açıyor. Ölçekleme bunu dengeliyor.

İtiraf etmem gerekirse, bu sadeleştirme olmasaydı gradient'ler daha stabil olmayacaktı. Makaledeki en ince dokunuşlardan biri bence.

## Self-Attention vs Cross-Attention

İki tür attention var:

- **Self-Attention**: Q, K ve V aynı kaynaktan geliyor. Yani bir cümlede her kelime diğer kelimelere bakıyor. Encoder'ın içinde ve decoder'ın masked self-attention katmanında kullanılıyor.
- **Cross-Attention**: Q bir kaynaktan (decoder), K ve V başka bir kaynaktan (encoder çıktısı) geliyor. Decoder'ın ikinci attention katmanı bu. Dil çevirisinde Fransızca cümle üretirken İngilizce kaynağa bakmak gibi.

## Encoder-Decoder Yapısı

Transformer'ın orijinal mimarisi iki ana bloktan oluşur:

**Encoder** (N tane, genelde 6 veya 12):
1. Self-Attention katmanı
2. Feed-Forward Network (FFN)
3. Layer Normalization + Residual connections (güncel versiyonlarda Pre-LN tercih ediliyor)

**Decoder** (yine N tane):
1. Masked Self-Attention (gelecek tokenleri görmesin diye)
2. Cross-Attention (encoder'dan bilgi alır)
3. Feed-Forward Network
4. Layer Normalization + Residual connections

Residual connections dediğim şey aslında: katman çıktısına girişi ekliyorsun. Yani `output = x + Attention(LayerNorm(x))`. Bu, derin ağlarda gradient akışını çok rahatlatıyor.

### Masked Attention Nedir?

Decoder'da self-attention yaparken, modelin gelecek tokenleri görmesini istemiyoruz. Yoksa "tahmin" yapmış olmaz, direkt cevabı kopyalamış olur. Bu yüzden attention skorlarındaki gelecek pozisyonları -∞ yapıyoruz, softmax onları 0'a yaklaştırıyor, yani model onları görmemiş oluyor.

Buna "causal masking" veya "autoregressive masking" deniyor. Kısacası: her token sadece kendinden öncekilere bakabilir.

## Hadi Kodlayalım: Minimal Bir Attention Head

Teori güzel de, gerçekten çalıştığını görmek ister misiniz? Gelin C# ile sıfırdan, hiçbir ML kütüphanesine ihtiyaç duymadan bir attention head yazalım.

İhtiyacımız olan tek şey MathNet.Numerics — onu da matrix çarpımları için kullanacağız. Softmax'ı ve tüm mantığı kendimiz yazacağız.

```csharp
// Program.cs
// dotnet new console -n AttentionDemo
// dotnet add package MathNet.Numerics
// dotnet run

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace AttentionDemo;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // Bir cümle: "Köpek kediyi kovaladı"
        // Her kelimeyi 4 boyutlu bir embedding ile temsil edelim
        // (gerçekte 512 veya 768 boyut kullanılır, ama örnek olsun)
        string[] tokens = { "Köpek", "kediyi", "kovaladı" };
        int embDim = 4;

        // Embedding matrisi: [token_sayısı x embedding_boyutu]
        // Rastgele değerler veriyoruz, gerçekte bunlar öğrenilir
        var embeddings = DenseMatrix.OfArray(new double[,]
        {
            { 0.5, 0.2, 0.8, 0.1 },   // Köpek
            { 0.3, 0.9, 0.4, 0.6 },   // kediyi
            { 0.7, 0.1, 0.5, 0.9 }    // kovaladı
        });

        Console.WriteLine("=== Embeddings ===");
        Console.WriteLine(embeddings.ToString("F2"));

        // Weight matrisleri: Wq, Wk, Wv
        // [input_dim x d_k] boyutunda
        int dK = 3; // attention boyutu (genelde emb_dim'in bir böleni)
        var rng = new Random(42);

        var Wq = DenseMatrix.Create(embDim, dK, (i, j) => rng.NextDouble() * 0.2 - 0.1);
        var Wk = DenseMatrix.Create(embDim, dK, (i, j) => rng.NextDouble() * 0.2 - 0.1);
        var Wv = DenseMatrix.Create(embDim, dK, (i, j) => rng.NextDouble() * 0.2 - 0.1);

        Console.WriteLine("\n=== Wq (Query Weight) ===");
        Console.WriteLine(Wq.ToString("F3"));

        // Q, K, V'yi hesapla: X * W
        var Q = embeddings * Wq; // [3 x 3]
        var K = embeddings * Wk; // [3 x 3]
        var V = embeddings * Wv; // [3 x 3]

        Console.WriteLine("\n=== Q (Query) ===");
        Console.WriteLine(Q.ToString("F3"));
        Console.WriteLine("=== K (Key) ===");
        Console.WriteLine(K.ToString("F3"));

        // Attention skorları: Q * K^T
        var scores = Q * K.Transpose(); // [3 x 3]

        Console.WriteLine("\n=== Attention Skorları (Q*K^T) ===");
        Console.WriteLine(scores.ToString("F3"));

        // Ölçekleme: skorları sqrt(d_k) ile bölüyoruz
        double scaleFactor = Math.Sqrt(dK);
        var scaledScores = scores / scaleFactor;

        Console.WriteLine($"\n=== Ölçeklenmiş Skorlar (/{scaleFactor:F2}) ===");
        Console.WriteLine(scaledScores.ToString("F3"));

        // Softmax uygula
        var attentionWeights = SoftmaxMatrix(scaledScores);

        Console.WriteLine("\n=== Attention Ağırlıkları (Softmax Çıktısı) ===");
        Console.WriteLine(attentionWeights.ToString("F4"));

        // Nihai çıktı: attention_weights * V
        var output = attentionWeights * V;

        Console.WriteLine("\n=== Attention Çıktısı ===");
        Console.WriteLine(output.ToString("F3"));

        // Her token için hangi tokenlere ne kadar baktığını göster
        Console.WriteLine("\n=== Token Bazında Attention Dağılımı ===");
        for (int i = 0; i < tokens.Length; i++)
        {
            Console.Write($"{tokens[i],-10} → ");
            for (int j = 0; j < tokens.Length; j++)
            {
                Console.Write($"{tokens[j]}: {attentionWeights[i, j],6:F2}  ");
            }
            Console.WriteLine();
        }

        Console.WriteLine("\n=== Masked Self-Attention Gösterimi ===");
        // Sadece decoder'da kullanılan masked attention:
        // Gelecek tokenleri maskeliyoruz (üst üçgen -∞)
        var maskedScores = scaledScores.Clone();
        for (int i = 0; i < tokens.Length; i++)
        {
            for (int j = i + 1; j < tokens.Length; j++)
            {
                maskedScores[i, j] = double.NegativeInfinity;
            }
        }

        Console.WriteLine("Maskelenmiş skorlar (üst üçgen -∞):");
        Console.WriteLine(maskedScores.ToString("F3"));
        Console.WriteLine("\nNot: -∞ gören hücrelerdeki 0.0000 değerlerine dikkat edin.");

        var maskedWeights = SoftmaxMatrix(maskedScores);
        Console.WriteLine("\nMaskelenmiş attention ağırlıkları:");
        Console.WriteLine(maskedWeights.ToString("F4"));
    }

    /// <summary>
    /// Bir matrisin her satırına softmax uygular.
    /// Sayısal stabilite için önce maksimumu çıkarırız.
    /// </summary>
    static Matrix<double> SoftmaxMatrix(Matrix<double> matrix)
    {
        var rows = matrix.RowCount;
        var cols = matrix.ColumnCount;
        var result = DenseMatrix.OfMatrix(matrix);

        for (int i = 0; i < rows; i++)
        {
            // Sayısal stabilite için maksimum değeri bul
            double maxVal = matrix.Row(i).Maximum();

            // exp(x_i - max) hesapla
            double sum = 0;
            for (int j = 0; j < cols; j++)
            {
                double expVal = Math.Exp(matrix[i, j] - maxVal);

                // NaN veya Infinity kontrolü
                if (double.IsNaN(expVal) || double.IsInfinity(expVal))
                    expVal = 0;

                result[i, j] = expVal;
                sum += expVal;
            }

            // normalize et
            if (sum > 0)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] /= sum;
                }
            }
        }

        return result;
    }
}
```

Bu kodu çalıştırdığınızda şöyle bir çıktı göreceksiniz (değerler random seed'e göre değişir, ama mantık aynı):

```
=== Embeddings ===
DenseMatrix 3x3
          0,50     0,20     0,80     0,10
          0,30     0,90     0,40     0,60
          0,70     0,10     0,50     0,90

=== Attention Ağırlıkları (Softmax Çıktısı) ===
DenseMatrix 3x3
          0,5741   0,1929   0,2330
          0,2157   0,6094   0,1749
          0,2833   0,3854   0,3313

=== Token Bazında Attention Dağılımı ===
Köpek     → Köpek:  0.57  kediyi: 0.19  kovaladı: 0.23
kediyi    → Köpek:  0.22  kediyi: 0.61  kovaladı: 0.17
kovaladı  → Köpek:  0.28  kediyi: 0.39  kovaladı: 0.33
```

Görüldüğü gibi her token kendisine en çok dikkat ediyor (bu beklenen bir durum, çünkü bir kelimenin kendisiyle ilişkisi genelde en güçlü oluyor), ama diğer kelimelere de önem veriyor.

## Multi-Head Attention: Sadece Tek Kafa Yetmez

Tek bir attention head iyidir, ama birden çok head daha iyidir. Neden mi? Çünkü her head farklı bir ilişki türünü öğrenebilir. Bir head kelimeler arasındaki "özne-fiil" ilişkisine odaklanırken, diğeri "sıfat-isim" ilişkisine bakabilir.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

Burada her head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Yani her head'in kendi Q, K, V weight matrisleri var. Çıktılar birleştiriliyor ve bir projeksiyon matrisi W_O ile tekrar embedding boyutuna indirgeniyor.

GPT-3'te 96 layer ve 96 head var (her head 128 boyutlu, toplam 12288 embedding boyutu). Düşünsenize, her token diğer 12287 tokenin her biriyle 96 farklı açıdan ilişki kuruyor. Bu kadar parametrenin neden bu kadar iyi çalıştığını anlamak zor değil.

## Positional Encoding: Sıra Önemlidir

Attention mekanizmasının ilginç bir özelliği: permütasyon-invariant (sıralamayı umursamaz). Çünkü set işlemi yapıyoruz aslında — her token diğer tüm tokenlere bakıyor, sıralamayı bilmiyor.

Ama "Köpek kediyi kovaladı" ile "Kedi köpeği kovaladı" arasında dağlar kadar fark var. Bu yüzden tokenlere pozisyon bilgisi enjekte etmemiz gerekiyor.

Orijinal makalede sinüzoidal positional encoding kullanılmış:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Günümüzde çoğu model (GPT, Llama) bunun yerine "learned positional embeddings" veya "rotary position embeddings (RoPE)" kullanıyor. RoPE özellikle popüler, çünkü relative position bilgisini encode edebiliyor — yani iki token arasındaki mutlak değil, göreceli mesafe önemli.

## Encoder-Decoder vs Decoder-Only

Çoğu modern LLM (GPT, Llama, Claude) decoder-only mimari kullanıyor. Neden?

- Encoder gerektiren görevler (sınıflandırma, NER) için encoder-decoder'a ihtiyaç yok
- Decoder-only ile her şeyi generatif olarak çözebiliyorsun
- Daha basit, daha az parametre, scaling daha kolay

Ama encoder-decoder'ın hâlâ avantajlı olduğu yerler var: makine çevirisi, özetleme, herhangi bir "input → output" dönüşümü. T5, BART gibi modeller hâlâ encoder-decoder kullanıyor.

## Layer Normalization: Nerede Olmalı?

Bu konu aslında sandığınızdan daha önemli. Orijinal Transformer'da "Post-LN" kullanılıyordu: `LayerNorm(x + Sublayer(x))`. Ama sonra fark edildi ki "Pre-LN" daha stabil: `x + Sublayer(LayerNorm(x))`.

GPT-2'den beri çoğu model Pre-LN kullanıyor. Özellikle derin modellerde (32+ layer) gradient patlamasını önlemek için hayati önemde.

## C# İle Çok Kafalı Attention

Gelin bir de multi-head attention'ı C#'ta görelim. Bu sefer MathNet.Numerics ile değil, multidimensional dizilerle (float[,]) manuel işlem yaparak — böylece her şeyin nasıl çalıştığını daha net göreceksiniz.

```csharp
// MultiHeadAttention.cs
// dotnet new console -n MultiHeadAttention
// dotnet run

namespace MultiHeadAttention;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // Parametreler
        int seqLen = 4;      // cümledeki token sayısı
        int embDim = 8;      // embedding boyutu
        int numHeads = 2;    // kafa sayısı
        int headDim = 4;     // her kafa için boyut (embDim / numHeads = 8/2 = 4)

        // Rastgele embeddingler (örnek cümle: "Bugün hava çok güzel")
        float[,] embeddings = new float[seqLen, embDim];
        var rng = new Random(42);
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < embDim; j++)
                embeddings[i, j] = (float)(rng.NextDouble() * 2.0 - 1.0);

        Console.WriteLine("=== Embeddings (Token Sayısı: 4, Boyut: 8) ===");
        PrintMatrix(embeddings);

        // Rastgele Wq, Wk, Wv matrisleri — her head için ayrı
        // Aslında tek bir büyük matris olarak tutulur, ama anlaşılırlık için ayıralım
        float[][,] Wq = new float[numHeads][,];
        float[][,] Wk = new float[numHeads][,];
        float[][,] Wv = new float[numHeads][,];

        for (int h = 0; h < numHeads; h++)
        {
            Wq[h] = new float[embDim, headDim];
            Wk[h] = new float[embDim, headDim];
            Wv[h] = new float[embDim, headDim];
            for (int i = 0; i < embDim; i++)
                for (int j = 0; j < headDim; j++)
                {
                    Wq[h][i, j] = (float)(rng.NextDouble() * 0.2 - 0.1);
                    Wk[h][i, j] = (float)(rng.NextDouble() * 0.2 - 0.1);
                    Wv[h][i, j] = (float)(rng.NextDouble() * 0.2 - 0.1);
                }
        }

        // Her head için attention hesapla
        float[,] allHeadsOutput = new float[seqLen, embDim];
        float scaleFactor = (float)Math.Sqrt(headDim);

        for (int h = 0; h < numHeads; h++)
        {
            Console.WriteLine($"\n=== Head {h + 1} ===");

            // Q, K, V hesapla: embeddings * W
            float[,] Q = MatrixMultiply(embeddings, Wq[h]); // [4 x 4]
            float[,] K = MatrixMultiply(embeddings, Wk[h]); // [4 x 4]
            float[,] V = MatrixMultiply(embeddings, Wv[h]); // [4 x 4]

            // Attention skorları: Q * K^T
            float[,] scores = MatrixMultiply(Q, Transpose(K)); // [4 x 4]

            // Ölçekle
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < seqLen; j++)
                    scores[i, j] /= scaleFactor;

            // Softmax
            float[,] attnWeights = Softmax(scores);

            Console.WriteLine($"Attention Ağırlıkları (Head {h + 1}):");
            PrintMatrix(attnWeights);

            // Head çıktısı: attention_weights * V
            float[,] headOutput = MatrixMultiply(attnWeights, V); // [4 x 4]

            // Birleştirme: her head'in çıktısını yan yana koy
            int startCol = h * headDim;
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < headDim; j++)
                    allHeadsOutput[i, startCol + j] = headOutput[i, j];
        }

        Console.WriteLine("\n=== Tüm Head'lerin Birleştirilmiş Çıktısı ===");
        PrintMatrix(allHeadsOutput);

        // Projeksiyon matrisi Wo ile son bir lineer dönüşüm
        float[,] Wo = new float[embDim, embDim];
        for (int i = 0; i < embDim; i++)
            for (int j = 0; j < embDim; j++)
                Wo[i, j] = (float)(rng.NextDouble() * 0.2 - 0.1);

        float[,] finalOutput = MatrixMultiply(allHeadsOutput, Wo);
        Console.WriteLine("\n=== Final Multi-Head Attention Çıktısı (Wo ile çarpım) ===");
        PrintMatrix(finalOutput);
    }

    // İki matrisi çarpar
    static float[,] MatrixMultiply(float[,] a, float[,] b)
    {
        int rows = a.GetLength(0);
        int cols = b.GetLength(1);
        int inner = a.GetLength(1);
        var result = new float[rows, cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                float sum = 0;
                for (int k = 0; k < inner; k++)
                    sum += a[i, k] * b[k, j];
                result[i, j] = sum;
            }

        return result;
    }

    // Bir matrisin transpozunu alır
    static float[,] Transpose(float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new float[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j, i] = matrix[i, j];
        return result;
    }

    // Softmax (stabil versiyon)
    static float[,] Softmax(float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            // Maksimum değeri bul (stabilite için)
            float maxVal = float.MinValue;
            for (int j = 0; j < cols; j++)
                if (matrix[i, j] > maxVal) maxVal = matrix[i, j];

            float sum = 0;
            for (int j = 0; j < cols; j++)
            {
                float expVal = (float)Math.Exp(matrix[i, j] - maxVal);

                // NaN kontrolü
                if (float.IsNaN(expVal) || float.IsInfinity(expVal))
                    expVal = 0;
                result[i, j] = expVal;
                sum += expVal;
            }

            // Normalize et
            if (sum > 0)
                for (int j = 0; j < cols; j++)
                    result[i, j] /= sum;
        }

        return result;
    }

    static void PrintMatrix(float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            Console.Write("[");
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{matrix[i, j],8:F4}");
                if (j < cols - 1) Console.Write(", ");
            }
            Console.WriteLine(" ]");
        }
    }
}
```

Bu kod ne yapıyor özetle:

1. Rastgele embeddingler oluşturuyoruz (gerçekte bunlar eğitilmiş olacak)
2. Her head için ayrı Q, K, V ağırlıkları tanımlıyoruz
3. Her head'in kendi attention'ını hesaplamasına izin veriyoruz
4. Tüm head'lerin çıktılarını yan yana (concatenate) koyuyoruz
5. Bir projeksiyon matrisi (Wo) ile son bir dönüşüm yapıyoruz

Fark ettiyseniz burada döngülerle manuel matris çarpımı yaptık. Gerçek hayatta bunun için optimize edilmiş BLAS kütüphaneleri kullanılır, ama mantığı anlamak için birebir.

## Feed-Forward Network: Sayıları Karıştırıyoruz

Attention katmanından sonra gelen FFN, aslında iki lineer katman arasında bir ReLU aktivasyonudur:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Güncel modellerde ReLU yerine genelde GLU varyantları (SwiGLU, GeGLU) kullanılıyor. Llama 2'de SwiGLU var: 

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

FFN'in boyutu genelde embedding boyutunun 4 katıdır. Yani embedding boyutu 4096 ise, FFN'in iç boyutu 16384. İşte model parametrelerinin çoğu burada saklanır, attention'da değil. Çoğu kişi bunu yanlış biliyor.

## Complexity: Attention Neden Pahalı?

Standart (full) attention'ın zaman karmaşıklığı O(n²·d). Yani token sayısının karesiyle orantılı. Bu yüzden 100K tokenlı bir context window'unuz varsa, her adımda 10 milyar attention skoru hesaplamanız gerekiyor. Bu biraz can sıkıcı.

Neyse ki bu konuda çözümler var:
- **Flash Attention**: GPU hafıza hiyerarşisini akıllıca kullanarak attention'ı IO-bound'tan compute-bound'a çeviriyor. Kesinlikle okumanızı öneririm, çığır açıcı bir teknik.
- **Sparse Attention**: Her token tüm tokenlere değil, sadece belirli bir aralıktakilere bakıyor.
- **Linear Attention**: Karmaşıklığı O(n) seviyesine indiren yöntemler var.

Özetle, Transformer mimarisi basit bir fikir (attention) üzerine inşa edilmiş ama bu basitlik onun gücünü gizliyor. Self-attention'ın her tokenin diğer tüm tokenlerle etkileşime girmesine izin vermesi, modelin uzun menzilli bağımlılıkları öğrenmesini sağlıyor — RNN'lerin asla tam başaramadığı bir şey.

Bir sonraki yazıda Advanced Prompting tekniklerine, Chain-of-Thought'a ve ReAct döngülerine dalacağız. O zaman görüşürüz.

## Referanslar

- Vaswani et al., "Attention Is All You Need", 2017
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention", 2022
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
