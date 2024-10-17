import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 嵌入维度
        max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建一个位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置的div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 填充sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # 增加批次维度
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        d_model: 嵌入维度
        num_heads: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性层用于线性变换 Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def attention(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch_size, num_heads, seq_len, d_k]
        mask: [batch_size, 1, 1, seq_len] 或其他兼容的形状
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
        return output, attn
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attn = self.attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: 嵌入维度
        d_ff: 前馈网络的隐藏层维度
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # 自注意力子层
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # 前馈网络子层
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, tgt_mask=None, enc_mask=None):
        # 自注意力子层
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # 编码器-解码器注意力子层
        tgt2 = self.enc_attn(tgt, enc_output, enc_output, enc_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络子层
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        
        # 编码器和解码器堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # 输出线性层
        self.out = nn.Linear(d_model, tgt_vocab)
        
        self.dropout = nn.Dropout(dropout)
    
    def make_src_mask(self, src):
        # 假设 src 的形状为 [batch_size, src_seq_len]
        # mask 的形状为 [batch_size, 1, 1, src_seq_len]
        mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return mask  # [batch_size, 1, 1, src_seq_len]
    
    def make_tgt_mask(self, tgt):
        # 假设 tgt 的形状为 [batch_size, tgt_seq_len]
        batch_size, tgt_len = tgt.size()
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch_size,1,1,tgt_len]
        
        # 创建上三角矩阵，确保预测时不看到未来的信息
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask & subsequent_mask.unsqueeze(0).unsqueeze(1)
        return tgt_mask  # [batch_size,1,tgt_len,tgt_len]
    
    def forward(self, src, tgt):
        """
        src: [batch_size, src_seq_len]
        tgt: [batch_size, tgt_seq_len]
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 嵌入和位置编码
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        # 编码器
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 解码器
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)
        
        output = self.out(dec_output)
        return output
