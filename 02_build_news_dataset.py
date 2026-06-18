import pandas as pd
import json

try:
    df = pd.read_csv('stock_news.csv')
    print("Veri seti başarıyla yüklendi!")
except Exception as e:
    print(f"Hata: 'stock_news.csv' dosyası bulunamadı! Hata: {e}")
    exit()

df = df.fillna('')
dataset = []

for index, row in df.iterrows():
    # Haber başlığı ve açıklamasını birleştirerek modele zengin bir metin veriyoruz
    haberler = f"Başlık: {row['title']} | Özet: {row['description']}"
    
    # impact_tier değerine göre (Genelde pozitif değerler yükseliş, negatifler düşüştür)
    # Eğer veri setinde impact_tier sadece sayıysa veya metinse güvenli bir dönüşüm yapıyoruz:
    try:
        val = float(row['impact_tier'])
        if val > 0:
            durum = "YUKSELIS (Olumlu)"
        elif val < 0:
            durum = "DUSUS (Olumsuz)"
        else:
            durum = "BELIRSIZ"
    except:
        # Eğer sayı değilse (metinse) içinde olumlu/olumsuz kelimeleri arayalım
        val_str = str(row['impact_tier']).lower()
        if 'pos' in val_str or 'up' in val_str or 'high' in val_str:
            durum = "YUKSELIS (Olumlu)"
        elif 'neg' in val_str or 'down' in val_str or 'low' in val_str:
            durum = "DUSUS (Olumsuz)"
        else:
            durum = "BELIRSIZ"
            
    # Llama 3.1 Şablonu
    prompt_semasi = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Sen haber başlıklarından borsa yönü tahmin eden yapay zeka tabanlı bir finans stratejistisin.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Bugünün kritik ekonomi manşetleri şunlardır:\n1: {haberler}\n\nBu haberler ışığında piyasanın bugünkü kapanış yönü ne olur?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"Verilen haber başlıkları analiz edildiğinde piyasanın bugünkü genel eğilimi: {durum}.<|eot_id|>"
    )
    dataset.append({"text": prompt_semasi})

# %90 Train, %10 Valid bölmesi
train_size = int(len(dataset) * 0.9)
train_data = dataset[:train_size]
valid_data = dataset[train_size:]

# Dosyaları doğrudan news_data klasörünün içerisine yazıyoruz
with open('news_data/train.jsonl', 'w', encoding='utf-8') as f:
    for entry in train_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open('news_data/valid.jsonl', 'w', encoding='utf-8') as f:
    for entry in valid_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Bitti! Yeni gerçek veriler 'news_data/' klasörüne yazıldı. Toplam: {len(dataset)} adet.")
