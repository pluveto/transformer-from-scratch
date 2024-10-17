import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import transformer
from util import read_tsv
from vocab import tokenize

MAX_LEN = 128

class TaskDataset(Dataset):
    def __init__(self, data, src_word2idx, tgt_word2idx, max_len=MAX_LEN):
        self.data = data
        self.src_word2idx = src_word2idx
        self.tgt_word2idx = tgt_word2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def encode(self, text, word2idx):
        tokens = tokenize(text)
        tokens = tokens[:self.max_len - 2]  # 留出 <sos> 和 <eos>
        tokens = ['<sos>'] + tokens + ['<eos>']
        return [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = self.encode(src, self.src_word2idx)
        tgt_ids = self.encode(tgt, self.tgt_word2idx)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

class TestDataset(Dataset):
    def __init__(self, data, src_word2idx, max_len=MAX_LEN):
        self.data = data
        self.src_word2idx = src_word2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def encode(self, text, word2idx):
        tokens = tokenize(text)
        tokens = tokens[:self.max_len-2]
        tokens = ['<sos>'] + tokens + ['<eos>']
        return [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    
    def __getitem__(self, idx):
        src = self.data[idx]
        src_ids = self.encode(src, self.src_word2idx)
        return torch.tensor(src_ids, dtype=torch.long), src

def collate_fn(batch, src_word2idx, tgt_word2idx):
    src_batch, tgt_batch = zip(*batch)
    
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=src_word2idx['<pad>'], batch_first=True)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tgt_word2idx['<pad>'], batch_first=True)
    
    return src_padded, tgt_padded

if __name__ == '__main__':

    data_dir = './data'
    train_file = os.path.join(data_dir, 'birth_places_train.tsv')
    dev_file = os.path.join(data_dir, 'birth_dev.tsv')
    test_file = os.path.join(data_dir, 'birth_test_inputs.tsv')

    train_data = read_tsv(train_file, has_label=True)
    dev_data = read_tsv(dev_file, has_label=True)
    test_data = read_tsv(test_file, has_label=False)

    src_vocab = torch.load('save/wiki_vocab.pt')
    tgt_vocab = src_vocab

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(f"Source Vocabulary Size: {src_vocab_size}")
    print(f"Target Vocabulary Size: {tgt_vocab_size}")


    # 创建数据集
    train_dataset = TaskDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TaskDataset(dev_data, src_vocab, tgt_vocab)
    test_dataset = TestDataset(test_data, src_vocab)

    # 创建数据加载器
    BATCH_SIZE = 64

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, src_vocab, tgt_vocab))
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, src_vocab, tgt_vocab))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: collate_fn([x], src_vocab, tgt_vocab))

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

    # 加载预训练模型
    pretrain_model_path = 'save/pretrain_checkpoint.pt'
    if os.path.exists(pretrain_model_path):
        model.load_state_dict(torch.load(pretrain_model_path))
        print("Pretrain model loaded.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    NUM_EPOCHS = 20
    CLIP = 1.0

    def train_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        epoch_loss = 0
        
        for src, tgt in tqdm(dataloader, desc="Training", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt[:, :-1])
            
            output = output.reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
                src = src.to(device)
                tgt = tgt.to(device)
                
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.size(-1))
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)

    os.makedirs('save', exist_ok=True)

    patience = 5
    best_dev_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss = evaluate(model, dev_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

        # Check for improvement
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            counter = 0
            torch.save(model.state_dict(), 'save/train_checkpoint.pt')  # Save the best model
            print("Validation loss improved, saving model.")
        else:
            counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {counter} out of {patience}")

        # Early stopping condition
        if counter >= patience:
            print("Early stopping")
            break

        torch.save(model.state_dict(), f"save/train_epoch_{epoch}.pt")
