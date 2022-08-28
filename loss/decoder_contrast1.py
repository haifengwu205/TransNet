import torch
import torch.nn as nn


class dec_deeplabv3_contrast(nn.Module):
    def __init__(self, channels=64, num_classes=2, inner_planes=2, temperature=0.2, queue_len=2975, device=None):
        super(dec_deeplabv3_contrast, self).__init__()

        self.temperature = temperature
        self.queue_len = queue_len
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.channels = channels
        self.device = device
        for i in range(num_classes):
            self.register_buffer("queue"+str(i), torch.randn(inner_planes, self.queue_len))
            self.register_buffer("ptr"+str(i), torch.zeros(1, dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')
           
    def _dequeue_and_enqueue(self, keys, vals, cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:, ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

    def construct_region(self, fea, pred):
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1)
        val = torch.unique(pred)
        fea = fea.squeeze()
        # [channel, bs, h*w]
        fea = fea.view(bs, self.channels, -1).permute(1, 0, 2)

        new_fea = fea[:, pred == val[0]].mean(1).unsqueeze(0)
        for i in val[1:]:
            if i < self.num_classes:
                class_fea = fea[:, pred == i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea, class_fea), dim=0)
        val = torch.tensor([i for i in val if i < self.num_classes])
        return new_fea.to(self.device), val.to(self.device)

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)
        logits = torch.cat((l_pos, l_neg),dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,), dtype=torch.long).to(self.device)
        return self.criterion(logits, labels)

    def forward(self, fea, res):
        # print("fea: ", fea.shape)
        # print("res.shape: ", res.shape)

        # fea = torch.randn((2, 256, 128, 128))
        # res = torch.randn((2, 19, 128, 128))
     
        # aspp_out = self.aspp(x)
        # fea = self.head(aspp_out)
        # res = self.final(fea)
        bs = fea.shape[0]
        keys, vals = self.construct_region(fea, res)  #keys: N,256   vals: N,  N is the category number in this batch
        keys = nn.functional.normalize(keys, dim=1)
        contrast_loss = 0

        for cls_ind in range(self.num_classes):
            if cls_ind in vals:
                query = keys[list(vals).index(cls_ind)]   #256,
                l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).to(self.device).clone().detach()  #256, N1
                all_ind = [m for m in range(self.num_classes)]
                l_neg = 0
                tmp = all_ind.copy()
                tmp.remove(cls_ind)
                for cls_ind2 in tmp:
                    l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).to(self.device).clone().detach()
                contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
            else:
                continue
        for i in range(self.num_classes):
            self._dequeue_and_enqueue(keys, vals, i, bs)
        return contrast_loss


if __name__ == '__main__':
    feats = torch.randn((2, 2, 128, 128))
    label = torch.randn((2, 2, 128, 128))
    # label1 = torch.ones((2, 1, 128, 128))
    # label2 = torch.zeros((2, 1, 128, 128))
    # label = torch.cat((label1, label2), dim=1)
    b = dec_deeplabv3_contrast()(feats, label)
    print(b)
    # print(b[1])
    # a = construct_region(feats, label)
    # print("a[0]: ", a[0].shape)
    # print("a[1]: ", a[1])

