import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import transformer  
from vocab import tokenize

data_dir = './data'
wiki_file = os.path.join(data_dir, 'wiki.txt')

src_vocab = torch.load('save/wiki_vocab.pt')
tgt_vocab = src_vocab

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

print(f"Vocabulary Size: {src_vocab_size}")

MAX_LEN = 128

class WikiDataset(Dataset):
    def __init__(self, file_path, word2idx, max_len=MAX_LEN):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def encode(self, text):
        tokens = tokenize(text)
        tokens = tokens[:self.max_len - 2]  # 留出 <sos> 和 <eos>
        tokens = ['<sos>'] + tokens + ['<eos>']
        token_ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        input_ids, target_ids = self.encode(line)
        return input_ids, target_ids

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, padding_value=src_vocab['<pad>'], batch_first=True)
    targets_padded = nn.utils.rnn.pad_sequence(targets, padding_value=tgt_vocab['<pad>'], batch_first=True)
    return inputs_padded, targets_padded

dataset = WikiDataset(wiki_file, src_vocab, max_len=MAX_LEN)

BATCH_SIZE = 64
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = transformer.Transformer(
    src_vocab=src_vocab_size, 
    tgt_vocab=tgt_vocab_size, 
    d_model=512, 
    num_layers=6, 
    num_heads=8, 
    d_ff=2048, 
    dropout=0.1, 
    max_len=MAX_LEN
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

NUM_EPOCHS = 100
CLIP = 1.0

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Transformer 模型接收 src 和 tgt，且 tgt 是 shifted
        output = model(inputs, targets[:, :-1])

        # output 的形状应该是 [batch_size, seq_len -1, vocab_size]
        output = output.reshape(-1, output.size(-1))
        targets = targets[:, 1:].reshape(-1)

        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs, targets[:, :-1])
            output = output.reshape(-1, output.size(-1))
            targets = targets[:, 1:].reshape(-1)

            loss = criterion(output, targets)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# 分割数据为训练集和验证集（例如 90% 训练，10% 验证）
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

os.makedirs('save', exist_ok=True)

start_epoch = 1
last_epoch = 0
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    # try to load the pre-trained model
    if os.path.exists(f"save/pretrain_epoch_{epoch}.pt"):
        last_epoch = epoch
    else:
        break
if last_epoch > 0:
    model.load_state_dict(torch.load(f"save/pretrain_epoch_{last_epoch}.pt"))
    start_epoch = last_epoch + 1
    print(f"Load pre-trained model from epoch {last_epoch}")

patience = 5
best_val_loss = float('inf')
counter = 0

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}/{NUM_EPOCHS}")

    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'save/pretrain_checkpoint.pt')  # Save the best model
        print("Validation loss improved, saving model.")
    else:
        counter += 1
        print(f"No improvement in validation loss. Early stopping counter: {counter} out of {patience}")

    # Early stopping condition
    if counter >= patience:
        print("Early stopping")
        break

    torch.save(model.state_dict(), f"save/pretrain_epoch_{epoch}.pt")
