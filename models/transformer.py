import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, feature_size=5, d_model=128, num_layers=4, output_size=1, nhead=8, dropout=0.01):
        super(TransformerModel, self).__init__()
        
        # 維度擴張：將原始特徵轉為 d_model 維度
        self.src_to_dmodel = nn.Linear(feature_size, d_model)
        self.tgt_to_dmodel = nn.Linear(output_size, d_model)
        
        # Transformer 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        # 輸出層：轉回預測維度
        self.dmodel_to_out = nn.Linear(d_model, output_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, device='cpu'):
        # Encoder part
        src = self.src_to_dmodel(src)
        memory = self.transformer_encoder(src)

        # Decoder part
        tgt = self.tgt_to_dmodel(tgt)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        return self.dmodel_to_out(output)