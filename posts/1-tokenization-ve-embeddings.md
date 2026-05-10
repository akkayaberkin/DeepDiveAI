# Tokenization ve Embeddings: Dil Modellerinin Alfabesi

Bir dil modeline "Merhaba, nasılsın?" yazdığınızda aslında ne gönderiyorsunuz? Karakterler mi? Kelimeler mi? Yoksa sayılar mı?

Cevap: sayılar. Ama bu sayılara dönüşüm sandığınız kadar basit değil. Gelin lafı uzatmadan dalalım.

## Tokenization Nedir ve Neden Önemlidir?

İnsan diliyle makinenin anlayabileceği format arasındaki ilk köprü tokenization. Model, ham metni olduğu gibi işleyemez — ona sayısal girdiler vermeniz gerekir. Tokenization, metni parçalara (token) ayırma ve bu token'ları bir sözlük üzerinden ID'lere eşleme süreci.

Ama işin püf noktası şu: Nasıl parçaladığınız, modelin dili anlama biçimini kökten etkiler.

### Üç Ana Tokenization Yaklaşımı

**Word-level Tokenization:** En sezgisel yaklaşım. Her kelime bir token. "Merhaba dünya" → ["Merhaba", "dünya"]. Basit, anlaşılır, ama sorunlu. Türkçe gibi sondan eklemeli dillerde "ev", "eve", "evde", "evden" ayrı token'lar. Kelime dağarcığınız şişer, seyreklik (sparsity) can sıkar. İngilizce için bile 50k+ token gerekir.

**Character-level Tokenization:** Her karakter bir token. "Merhaba" → ["M","e","r","h","a","b","a"]. Kelime dağarcığı çok küçük (birkaç yüz token), dizi uzunluğu ise çok büyük. "Merhaba, nasılsın?" cümlesi 20+ token. Transformer'ların O(n²) karmaşıklığıyla bu pratik değil. Hiç deneyen olmadı değil, ama pek verimli olmuyor.

**Subword Tokenization (Altın Oran):** İkisinin arası. Sık geçen kelimeler bütün halinde, az geçenler alt parçalara ayrılır. "Merhaba" bütün kalabilirken "anlayamadım" → ["anla", "ya", "ma", "dım"] şeklinde bölünebilir.

Şu anda her ciddi dil modeli subword tokenization kullanıyor. En yaygın üç algoritma:

### Byte-Pair Encoding (BPE)

BPE, GPT modellerinin kullandığı yöntem. İşleyişi şöyle:

1. Her karakteri ayrı bir token olarak başlat
2. En sık geçen karakter çiftini bul
3. Bu çifti birleştirip yeni bir token oluştur
4. Hedef kelime dağarcığı boyutuna ulaşana kadar tekrarla

```python
import re
from collections import Counter

def bpe_train(corpus, vocab_size=300):
    # Başlangıç: her kelimeyi karakterlerine ayır
    words = Counter(re.findall(r'\w+', corpus.lower()))
    vocab = {}
    
    # Her karakteri ayrı token yap
    for word in words:
        for char in word:
            if char not in vocab:
                vocab[char] = len(vocab)
    
    # Şimdi en sık geçen çiftleri birleştir
    tokenized_words = {word: list(word) for word in words}
    
    while len(vocab) < vocab_size:
        pairs = Counter()
        for word, tokens in tokenized_words.items():
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i+1])] += words[word]
        
        if not pairs:
            break
            
        # En sık çifti bul
        best_pair = pairs.most_common(1)[0][0]
        merged = ''.join(best_pair)
        vocab[merged] = len(vocab)
        
        # Birleştir
        new_tokenized = {}
        for word, tokens in tokenized_words.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokenized[word] = new_tokens
        tokenized_words = new_tokenized
    
    return vocab

# Basit bir kullanım
corpus = "Merhaba dünya merhaba herkese merhaba bugün"
vocab = bpe_train(corpus, vocab_size=50)
print(f"Vocab boyutu: {len(vocab)}")
print(f"Token'lar: {list(vocab.keys())[:10]}...")
```

Bu algoritmayı en saf haliyle gördünüz. OpenAI'nin tiktoken kütüphanesi BPE'nin çok daha optimize edilmiş bir versiyonunu kullanır.

### WordPiece (BERT)

Google'ın BERT modelinin tercihi. BPE'ye benzer ama çift seçim kriteri farklıdır. BPE frekansa bakarken, WordPiece birleşimin olasılık kazancını (likelihood increase) maksimize etmeye çalışır.

```
score(a, b) = freq(ab) / (freq(a) * freq(b))
```

Yani a ve b'nin ayrı ayrı geçme olasılığına göre birlikte geçme olasılığına bakılır. Bu, daha anlamlı birleşimleri bulmayı sağlar.

### Unigram (SentencePiece)

Google'ın bir diğer yaklaşımı. Tersine çalışır: büyük bir kelime dağarcığıyla başlayıp, en az katkı sağlayan token'ları teker teker çıkarır. Her adımda hangi token'ın çıkarılmasının loss'u en az artırdığı hesaplanır.

SentencePiece'in güzel tarafı, önceden tokenize edilmiş metin gerektirmemesi — ham metinle çalışır ve boşlukları da token olarak ele alır. Bu, diller arası geçişlerde çok işe yarar.

SentencePiece'i kullanmak oldukça basit:

```python
# sentencepiece kurulu değilse: pip install sentencepiece
import sentencepiece as spm

# Model eğit
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='unigram',  # veya 'bpe'
    character_coverage=1.0,
)

# Kullanım
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode('Merhaba dünya', out_type=str)
ids = sp.encode('Merhaba dünya', out_type=int)
print(f"Token'lar: {tokens}")  # ['▁Merhaba', '▁dünya']
print(f"ID'ler: {ids}")       # [42, 137]
```

## Tokenization'ın Modele Etkisi

"Tokenization sadece bir ön işleme adımı" derseniz, yanılırsınız. Tokenization'ın seçimi modelin performansını doğrudan etkiler.

**Kelime dağarcığı boyutu:** Çok küçük → uzun diziler, yavaş eğitim. Çok büyük → embedding matrisi şişer, seyrek gradient'ler.

**OOV (Out-of-Vocabulary) sorunu:** Subword tokenization bunu büyük ölçüde çözer. "Hipopotomonstroseskipedaliofobi" gibi bir kelime görseniz bile, alt parçalarına ayırabilirsiniz.

**Hata yayılımı:** Tokenization hataları (yanlış bölünmüş kelimeler) tüm model boyunca yayılır. "New York" → ["New", "York"] iyidir, ama "Newyork" → ["Newy", "ork"] anlam kaybına yol açar.

Benim gözlemim: Tokenizer seçimi genelde "hmm, herkes BERT kullanıyor, biz de WordPiece kullanalım" mantığıyla yapılıyor. Oysa kendi verinize özel bir tokenizer eğitmek, özellikle domain-specific projelerde (tıp, hukuk) çok daha iyi sonuç veriyor.

## Embedding'ler: Sayıların Dili

Tokenization'dan sonra elimizde bir dizi token ID'si var. Ama bu ID'ler arasında anlamsal bir ilişki yok — "kral" = 42, "kraliçe" = 57 olsun, ikisinin 15 birim uzaklıkta olması bir şey ifade etmez. İşte embedding'ler tam bu noktada devreye giriyor.

### Word Embedding'lerin Matematiği

Her token, n boyutlu bir vektöre eşlenir. Basit bir embedding katmanı:

```python
import torch
import torch.nn as nn

vocab_size = 32000
embedding_dim = 768  # GPT-2 small için

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Rastgele bir token ID'si
token_id = torch.tensor([42])
vector = embedding_layer(token_id)
print(f"Vektör boyutu: {vector.shape}")  # torch.Size([1, 768])
```

Bu vektörlerin her boyutu, kelimenin bazı gizli özelliklerini temsil eder. Ama bu özellikler insan tarafından etiketlenmez — model kendi keşfeder.

Kelime vektörlerinin en büyüleyici özelliği: **anlamsal benzerlik**, vektör uzayında **mesafe** olarak kodlanır.

```
v("kral") - v("erkek") + v("kadın") ≈ v("kraliçe")
```

Bu, 2013'te Mikolov'un Word2Vec makalesinde gösterildiğinde herkesi şaşırtmıştı. (İtiraf edeyim, ben de ilk okuduğumda "yok artık" demiştim.)

### Farklı Embedding Yaklaşımları

**Word2Vec (Mikolov et al., 2013):** İki varyantı var:
- CBOW (Continuous Bag of Words): Bağlamdan kelimeyi tahmin et
- Skip-gram: Kelimeden bağlamı tahmin et

Skip-gram seyrek verilerde daha iyiyken, CBOW daha hızlı eğitilir.

**GloVe (Pennington et al., 2014):** Kelime birlikte geçme istatistiklerini (co-occurrence matrix) kullanır. Vektörleri, kelime çiftlerinin birlikte geçme olasılıklarının oranını yansıtacak şekilde optimize eder.

**FastText (Bojanowski et al., 2016):** Her kelimeyi karakter n-gramlarının toplamı olarak temsil eder. "apple" → ["ap", "app", "ppl", "ple", "le"] gibi. Bu, görülmemiş kelimeler için bile vektör üretebilme avantajı sağlar. Türkçe gibi sondan eklemeli dillerde harika çalışır.

### Contextual Embeddings (Transformer Çağı)

Statik embedding'lerin büyük sorunu: Her kelimenin tek bir vektörü var. Oysa "bank" kelimesi "nehir bankı" ve "para bankası"nda farklı anlamlar taşır.

Modern modeller (BERT, GPT) bunu **contextual embedding'lerle** çözer. Aynı token, farklı bağlamlarda farklı vektörlere sahip olur.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# İki farklı bağlamda "bank"
texts = [
    "I sat by the river bank.",
    "I need to go to the bank."
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # [CLS] token'ın embedding'ini alalım (tüm cümleyi temsil eder)
    embedding = outputs.last_hidden_state[:, 0, :]
    print(f"'{text}' → embedding shape: {embedding.shape}")
```

BERT'in çıktısındaki `last_hidden_state`'in shape'i `(batch_size, seq_len, hidden_dim)` şeklindedir. Her token pozisyonu için ayrı bir embedding. Bu sayede "bank" token'ı, cümle 1 ve cümle 2'de farklı vektörlere sahip olur.

## Positional Encoding: Sıra Bilgisi

Transformer'lar doğal olarak sıralı bilgiyi anlamaz. Self-attention mekanizması tüm token'ları aynı anda görür — hangisinin önce geldiğini bilmez. Bu yüzden positional encoding eklenir.

GPT'nin kullandığı learned positional embedding'ler:

```python
max_seq_len = 1024
hidden_dim = 768

position_ids = torch.arange(max_seq_len).unsqueeze(0)  # [1, 1024]
positional_embeddings = nn.Embedding(max_seq_len, hidden_dim)
pos_vectors = positional_embeddings(position_ids)  # [1, 1024, 768]
```

BERT ise farklı bir yaklaşım kullanır: **sinusoidal positional encoding**. Bunun güzelliği, modelin görmediği uzunluktaki dizilere bile genelleme yapabilmesidir.

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
import numpy as np

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]  # [seq_len, 1]
    i = np.arange(d_model)[np.newaxis, :]     # [1, d_model]
    
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    pos_encoding = pos * angle_rates
    
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])  # çift indeksler
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])  # tek indeksler
    
    return pos_encoding

pe = sinusoidal_positional_encoding(50, 768)
print(f"Positional encoding shape: {pe.shape}")  # (50, 768)
```

Bu sinüs-cos dalgalarının frekansı pozisyona göre değişir. Düşük pozisyonlarda hızlı değişen (yüksek frekans), yüksek pozisyonlarda yavaş değişen (düşük frekans) dalgalar. Bu sayede model hem yakın token'ları (benzer dalga fazı) hem de uzak token'ları (farklı faz) ayırt edebilir.

## Pratik Tavsiyeler

Kendi projenizde tokenizer seçerken şunlara dikkat edin:

1. **Dil önemli.** Türkçe, Fince, Macarca gibi diller subword tokenization'a çok ihtiyaç duyar. İngilizce karakter setiyle eğitilmiş bir tokenizer, "ç" veya "ğ" harflerini byte seviyesine kadar indirgeyebilir.

2. **Domain-specific tokenizer eğitin.** Tıp metinleri, kod, hukuk belgeleri — her domainin kendine özgü kelime dağarcığı var. Genel tokenizer'lar bu terimleri verimsiz tokenize eder.

3. **Vocab size sweet spot'u bulun.** Genelde 32k-50k arası iyi çalışıyor. Ama daha büyük modeller (LLaMA 65B gibi) 32k ile yetinirken, çok dilli modeller 250k'ya kadar çıkabiliyor.

4. **Embedding boyutunu dengeli seçin.** 768 (BERT-base) genel amaçlı projeler için yeterli. 4096 (GPT-4) büyük modeller ve zorlu görevler için. Ama embedding boyutu arttıkça parametre sayısı katlanarak artar (vocab_size × embedding_dim).

5. **Fine-tune sırasında embedding'leri dondurmak** bazen işe yarayabilir, ama genelde ince ayarın embedding katmanını da güncellemesine izin vermek daha iyidir.

## Kaynaklar

- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
- Pennington et al., "GloVe: Global Vectors for Word Representation", 2014. [ACL 2014](https://aclanthology.org/D14-1162/)
- Bojanowski et al., "Enriching Word Vectors with Subword Information", 2016. [arXiv:1607.04606](https://arxiv.org/abs/1607.04606)
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Vaswani et al., "Attention Is All You Need", 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer", 2018. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)

---

Bir sonraki yazıda Transformer mimarisine ve attention mekanizmasına matematiksel olarak dalacağız. Self-attention'ın QKV matrislerinden çok başlı attention'a kadar her şeyi didik didik edeceğiz. Görüşmek üzere.
