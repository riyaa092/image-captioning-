import torch
import torch.nn as nn
class Captioner(nn.Module):
    def __init__(self):
        super(Captioner, self).__init__()
        self.img_fc = nn.Linear(2048, 256)         # Match embedding size
        self.emb = nn.Embedding(14, 256)           # Match vocab size
        self.gru = nn.GRU(256, 512, batch_first=True)
        self.fc_out = nn.Linear(512, 14)           # Output vocab size

    def forward(self, features, captions):
        features = self.img_fc(features).unsqueeze(1)         # [B, 1, 256]
        captions = self.emb(captions[:, :-1])                 # [B, T, 256]
        inputs = torch.cat((features, captions), dim=1)       # [B, T+1, 256]
        hiddens, _ = self.gru(inputs)                         # [B, T+1, 512]
        outputs = self.fc_out(hiddens)                        # [B, T+1, 14]
        return outputs

    def generate_caption(self, features, vocab, max_len=20):
        results = []
        states = None
        inputs = self.img_fc(features).unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.gru(inputs, states)
            outputs = self.fc_out(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            word = vocab.idx2word.get(predicted.item(), "<unk>")
            if word == "<end>":
                break
            results.append(word)
            inputs = self.emb(predicted).unsqueeze(1)
        return ' '.join(results)
