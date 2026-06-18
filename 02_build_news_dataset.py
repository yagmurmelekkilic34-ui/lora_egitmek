import pandas as pd
import json

try:
    # stock_news.csv dosyasını oku
    df = pd.read_csv('stock_news.csv')
    print("Veri seti yüklendi. Sütunlar:", df.columns.tolist())
except Exception as e:
    print(f"Hata: 'stock_news.csv' dosyası bulunamadı! Hata: {e}")
    exit()

df = df.fillna('')
dataset = []

# Haber sütunlarını seç (ilk 5 sütun)
columns = df.columns.tolist()
label_col = 'Label' if 'Label' in columns else ('label' if 'label' in columns else columns[1])
top_cols = [c for c in columns if c not in [label_col, 'Date', 'date', 'id', 'ID']][:5]

for index, row in df.iterrows():
    haber_listesi = [f"{i+1}: {row[col]}" for i, col in enumerate(top_cols) if row[col]]
    haberler = " | ".join(haber_listesi) if haber_listesi else "Haber bulunamadı."
    
    try:
        durum = "YUKSELIS (Olumlu)" if int(row[label_col]) == 1 else "DUSUS (Olumsuz)"
    except:
        durum = "BELIRSIZ"
    
    # Llama 3.1 Şablonu
    prompt_semasi = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Sen haber başlıklarından borsa yönü tahmin eden yapay zeka tabanlı bir finans stratejistisin.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Bugünün kritik ekonomi manşetleri şunlardır:\n{haberler}\n\nBu haberler ışığında piyasanın bugünkü kapanış yönü ne olur?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"Verilen haber başlıkları analiz edildiğinde piyasanın bugünkü genel eğilimi: {durum}.<|eot_id|>"
    )
    dataset.append({"text": prompt_semasi})

# %90 Train, %10 Valid bölmesi
train_size = int(len(dataset) * 0.9)
train_data = dataset[:train_size]
valid_data = dataset[train_size:]

# ESKİ DOSYALARI SİLMEMEK İÇİN YENİ İSİMLERLE KAYDEDİYORUZ
with open('train_news.jsonl', 'w', encoding='utf-8') as f:
    for entry in train_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open('valid_news.jsonl', 'w', encoding='utf-8') as f:
    for entry in valid_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Bitti! 'train_news.jsonl' ({len(train_data)}) ve 'valid_news.jsonl' ({len(valid_data)}) oluşturuldu!")
