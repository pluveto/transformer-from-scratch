import os
import torch
import re
from transformer import Transformer
from vocab import tokenize

MODEL_PATH = 'save/train_checkpoint.pt'
SRC_VOCAB_PATH = 'save/wiki_vocab.pt'

src_vocab = torch.load(SRC_VOCAB_PATH)
tgt_vocab = src_vocab
src_idx2word = {idx: word for word, idx in src_vocab.items()}
tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

MAX_LEN = 128
model = Transformer(
    src_vocab=src_vocab_size, 
    tgt_vocab=tgt_vocab_size, 
    d_model=512, 
    num_layers=6, 
    num_heads=8, 
    d_ff=2048, 
    dropout=0.1, 
    max_len=MAX_LEN
)


model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def encode(text, word2idx, max_len=50):
    tokens = tokenize(text)
    tokens = tokens[:max_len-2]  # 保留位置给<sos>和<eos>
    tokens = ['<sos>'] + tokens + ['<eos>']
    return [word2idx.get(token, word2idx['<unk>']) for token in tokens]

def decode(token_ids, idx2word):
    tokens = [idx2word[idx] for idx in token_ids]
    return ' '.join(tokens).replace('<sos>', '').replace('<eos>', '').strip()

def generate_response(model, src_text, src_vocab, tgt_vocab, src_idx2word, tgt_idx2word, max_len=50):
    src_ids = encode(src_text, src_vocab, max_len)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # 添加batch维度

    tgt_ids = [tgt_vocab['<sos>']]
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_len):
        output = model(src_tensor, tgt_tensor)
        next_token = output.argmax(dim=-1)[:, -1].item()
        tgt_ids.append(next_token)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        if next_token == tgt_vocab['<eos>']:
            break
    
    return decode(tgt_ids, tgt_idx2word)

if __name__ == "__main__":
    print("ask question and <quit> to exit.")
    while True:
        src_text = input(">")
        if src_text.lower() == 'quit':
            break
        
        response = generate_response(model, src_text, src_vocab, tgt_vocab, src_idx2word, tgt_idx2word)
        print(f"{response}")
