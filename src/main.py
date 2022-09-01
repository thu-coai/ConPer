import json
import argparse
import os
import nltk
from nltk import word_tokenize, sent_tokenize
from itertools import chain
import numpy as np
from tqdm import tqdm
import copy
from time import sleep

from bert_score import BERTScorer

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2PreTrainedModel, GPT2Model, AutoConfig, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithPastAndCrossAttentions
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

from tool.graph import get_conceptnet, KnowledgeGraph
from tool.extract_emotion_event import extract_emotion_event_at_least_one

torch.multiprocessing.set_sharing_strategy('file_system')


def get_idf_sents():
    def get_idf_docs():
        if os.path.exists('train_doc_idf.json'):
            with open('train_doc_idf.json', encoding='utf-8') as f:
                return json.load(f)
        file_name = 'train.json'
        with open(file_name, encoding='utf-8') as f:
            a = json.load(f)
            hyps = []
            refs = []
            for scene in a:
                for entry in scene['entries']:
                    hyps.append(entry['description'])
                for card in scene['entries'][-1]['cards']:
                    refs.append(card['description'])
            with open('train_doc_idf.json', 'w', encoding='utf-8') as fi:
                json.dump(hyps + refs, fi, ensure_ascii=False)
            print('finish get_idf_sent')

            return hyps + refs
    return get_idf_docs()

class Helper():
    def __init__(self, args):
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.config_path)
        self.eoc_token = '<|endofcard|>'
        self.bot_token = '<|beginoftarget|>'
        self.eot_token = '<|endoftarget|>'
        self.boo_token = '<|beginofoutline|>'
        self.bob_token = '<|beginofbedding|>'
        self.boe_token = '<|beginofending|>'
        self.soo_token = '<|sepofoutline|>'
        self.soos_token = '<|sepofoutlinesent|>'
        self.eop_token = '<|endofprompt|>'
        self.son_token = '<|sepofname|>'
        self.not_a_fact = 'NOT_A_FACT'
        self.tokenizer.add_tokens(
            [self.eop_token, self.eoc_token, self.bot_token, self.eot_token, self.bob_token, self.boe_token,
             self.boo_token, self.soo_token, self.soos_token])
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def pad_seq(self, ids, max_len):
        return self.tokenizer.pad(ids, max_len)

    def call_tokenizer(self, text):
        return self.tokenizer(text)

    def get_vocab_size(self):
        return len(self.tokenizer)


def get_nodes_dis(words, device=None, in_graph=True):
    if not in_graph:
        if device is None:
            return torch.zeros(helper.get_vocab_size(), dtype=torch.float)
        else:
            return torch.zeros(helper.get_vocab_size(), dtype=torch.float, device=device)

    seq = helper.soo_token.join(words)
    ids = helper.tokenizer.encode(seq)
    if device is None:
        res = torch.zeros(helper.get_vocab_size(), dtype=torch.float)
    else:
        res = torch.zeros(helper.get_vocab_size(), dtype=torch.float, device=device)
    res[ids] = 1
    return (1 - res) * (-1e10)

class StoriumDataset(Dataset):

    def __init__(self, in_file):
        super().__init__()
        self.in_file = in_file

        with open(self.in_file, encoding='utf-8') as f:
            self.data = json.load(f)

        self.vocab_size = len(helper.tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        
        input = ''

        # persona
        for card in item['entries'][-1]['cards']:
            input += card['description']
        input += helper.eoc_token

        # context
        entri_des = []
        for entry in item['entries'][:-1]:
            entri_des.append(entry['description'])
        input += ' '.join(entri_des)
        input += helper.eop_token
        
        output = ''

        # target sentence
        sents = nltk.sent_tokenize(item['entries'][-1]['description'])
        peak_idx = item['peak_idx']    
        output += helper.bot_token + sents[peak_idx] + helper.eot_token
        
        # outline keywords
        keywords_list = item['bedding_kws'] + item['ending_kws']
        old_keywords_list = item['bedding_kws'] + item['ending_kws']
        context_keywords_list = item['context_kws'] + list(chain(*item['target_kws']))
        for i in range(len(keywords_list)):
            keywords_list[i] = helper.soo_token.join(keywords_list[i])
        output += helper.soos_token.join(keywords_list)

        output += helper.bob_token
        output += ' '.join(sents[:peak_idx])
        output += helper.boe_token
        output += ' '.join(sents[peak_idx + 1:])
        output += helper.tokenizer.eos_token
        input_ts = torch.tensor(helper.tokenizer.encode(input, add_special_tokens=False), dtype=torch.long)
        output_ts = torch.tensor(helper.tokenizer.encode(output, add_special_tokens=False), dtype=torch.long)

        nodes_dis = []
        outline_mask = []
        keywords_list = list(chain(*old_keywords_list))
        for i in range(len(keywords_list)):
            words = graph.get_hops_set(keywords_list[:i] + context_keywords_list, hop=1)
            if keywords_list[i] in words:
                outline_mask.append(1)
                nodes_dis.append(get_nodes_dis(words))
            else:
                outline_mask.append(0)
                nodes_dis.append(get_nodes_dis(words))

        outline_mask_piece = []
        node_dis_piece = []
        ori_id = 0
        cat_tensor = torch.cat([input_ts, output_ts], dim=0)
        for i, value in enumerate(cat_tensor):
            if value == helper.get_token_id(helper.eot_token):
                for j, value in enumerate(cat_tensor[i + 1:]):
                    if value == helper.get_token_id(helper.soo_token) or value == helper.get_token_id(
                            helper.soos_token):
                        outline_mask_piece.append(0)  # sepofoutline算不在图谱中，置0
                        node_dis_piece.append(nodes_dis[ori_id])
                        ori_id += 1
                        continue
                    elif value == helper.get_token_id(helper.bob_token):
                        end = j
                        outline_mask_piece.append(0)
                        if len(keywords_list):
                            node_dis_piece.append(nodes_dis[ori_id])
                        else:
                            node_dis_piece.append(get_nodes_dis(words=None, in_graph=False))
                        break
                    else:
                        outline_mask_piece.append(outline_mask[ori_id])
                        node_dis_piece.append(nodes_dis[ori_id])
                break

        node_dis_ts = torch.stack(node_dis_piece, dim=0)
        outline_mask_ts = torch.tensor(outline_mask_piece, dtype=torch.long)

        start_idx = (cat_tensor == helper.get_token_id(helper.eot_token)).nonzero(as_tuple=False).item()
        end_idx = (cat_tensor == helper.get_token_id(helper.bob_token)).nonzero(as_tuple=False).item()
        ids = cat_tensor[start_idx + 1: end_idx + 1].tolist()

        context_kws = item['context_kws']
        target_kws = list(chain(*item['target_kws']))
        outline_kws = list(chain(*old_keywords_list))
        return {'input': input_ts, 'output': output_ts, 'peak_idx': peak_idx,
                'nodes_dis': node_dis_ts,
                'context_kws': context_kws, 'target_kws': target_kws, 'outline_kws': outline_kws,
                'outline_mask': outline_mask_ts}


def pad_collate(batch):
    def get_attention_mask(max_len, one_len):
        a = np.ones(max_len)
        a[one_len:] = 0
        return torch.tensor(a, dtype=torch.float)


    def loss_mask(ts, start, end):
        ts[:start] = -100
        ts[end:] = -100
        return ts
    res = {}
    res['input_ids'] = pad_sequence([torch.cat([x['input'], x['output']])[:1024] for x in batch], batch_first=True,
                                    padding_value=helper.tokenizer.pad_token_id)
    res['attention_mask'] = torch.stack(
        [get_attention_mask(res['input_ids'].size(1), len(x['input']) + len(x['output'])) for x in batch])
    res['labels'] = torch.stack(
        [loss_mask(copy.deepcopy(res['input_ids'][idx]), len(x['input']), len(x['input']) + len(x['output'])) for idx, x
         in enumerate(batch)])
    res['nodes_dis'] = [sample['nodes_dis'] for sample in batch]

    res['peak_idx'] = [sample['peak_idx'] for sample in batch]
    res['context_kws'] = [sample['context_kws'] for sample in batch]
    res['target_kws'] = [sample['target_kws'] for sample in batch]
    res['outline_kws'] = [sample['outline_kws'] for sample in batch]
    res['outline_mask'] = [sample['outline_mask'] for sample in batch]
    return res


class WordTokenizer():
    def __init__(self, file_path=''):
        with open(file_path, encoding='utf-8') as f:
            self.word2ids = json.load(f)

    def encode(self, word):
        try:
            result = self.word2ids[word]
        except:
            result = helper.tokenizer.encode(word, add_special_tokens=False)
            self.word2ids[word] = result
        return result

class Gpt2OutLineModel(GPT2LMHeadModel):

    @property
    def wte(self):
        return self.transformer.wte

    def __init__(self, config):
        super().__init__(config)
        self.outline_classify_head = nn.Linear(3 * config.n_embd, 2, bias=False)
        self.outline_wquery = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.outline_wvalue = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.word_wkey = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.word_wvalue = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.outline_lm_head = nn.Linear(3 * config.n_embd, helper.get_vocab_size(), bias=False)
        self.relation_num = graph.get_relation_size()  # 得到关系个数
        print('relation num = ', self.relation_num)
        self.relation_tensor = torch.nn.Embedding(self.relation_num, config.n_embd)
        self.Wh = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wt = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wr = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self.dis_matrix = nn.Linear(3 * config.n_embd, config.n_embd, bias=False)
        print('Gpt2OutLineModel init (with config)!')

    def get_concept_embedding(self, word):
        return wordTokenizer.encode(word)

    def get_graph_vectors_words(self, words, device, generate=False):
        # return: [words_len, hidden_size]
        relation_embs = []
        embedding_matrix = self.get_input_embeddings().weight

        lens = []

        encode_lens = []

        relation_ids = []
        entity_ids = []
        for word in words:
            triples = graph.get_triples(word)
            for h, r, t in triples:
                relation_ids.append(r)
                cur_head_id = self.get_concept_embedding(h)
                cur_tail_id = self.get_concept_embedding(t)
                entity_ids.append(cur_head_id)
                entity_ids.append(cur_tail_id)
                encode_lens.append(len(cur_head_id))
                encode_lens.append(len(cur_tail_id))
            lens.append(len(triples))

        max_len = max(encode_lens)
        padded_entity_ids = [x + [0] * (max_len - len(x)) for x in entity_ids]
        pad_entity_ids = torch.tensor(padded_entity_ids, device=device)
        embeds = embedding_matrix[pad_entity_ids]  # [ triple_lens, max_word_piece_len, hidden_size]

        # [triple_lens, max_len]
        mask_emb = np.zeros((len(encode_lens), max_len))
        for data, l in zip(mask_emb, encode_lens):
            data[: l] = 1
        mask_emb = torch.tensor(mask_emb, device=device, dtype=torch.float).unsqueeze(-1)

        temp = embeds * mask_emb
        real_emb = torch.sum(temp, dim=1)  # [triple_len, hidden_size]
        real_emb = real_emb.reshape((-1, 2, 768))  # [real_triple_len, 2, hidden_size]
        head_embs = real_emb[:, 0, :]
        tail_embs = real_emb[:, 1, :]
        concat_emb = torch.cat([head_embs, tail_embs], dim=1)  # [real_triple_len, hidden_size * 2]

        relation_embs = self.relation_tensor(torch.tensor(relation_ids, device=device))

        x = self.Wr(relation_embs)
        y = torch.tanh(self.Wh(head_embs) + self.Wt(tail_embs))
        betas = torch.sum(x * y, dim=-1)
        start = 0
        ans = []

        for i, l in enumerate(lens):
            end = start + l
            b = betas[start:end]
            alphas = F.softmax(b, dim=0)
            concat_ts = concat_emb[start:end]
            alphas = alphas.unsqueeze(dim=-1)
            result = alphas * concat_ts
            ans.append(torch.sum(result, dim=0))
            start = end

        if generate:
            return ans
        else:
            return torch.stack(ans, dim=0)

    def compute_all_graph_vectors(self, context_kws, target_kws, outline_kws, device):
        res = []
        for c, t, o in zip(context_kws, target_kws, outline_kws):
            cv = self.get_graph_vectors_words(c, device) if c else None
            tv = self.get_graph_vectors_words(t, device) if t else None
            ov = self.get_graph_vectors_words(o, device) if o else None
            res.append((cv, tv, ov))
        return res

    def get_context_graph_vector(self, hidden_state, words=None, gvs=None, mask=None):
        # hidden_state : [seq_len, hidden_size]
        # graph_vectors : [kws_len, 2 * hidden_size]
        # mask: [seq_len, kws_len]
        # return : [seq_len, hidden_size * 2]
        betas = []
        graph_vectors = []

        assert gvs is not None
        graph_vectors = gvs

        betas = torch.matmul(hidden_state, self.Wk(graph_vectors).T)
        if mask is not None:
            betas.masked_fill_(mask, -1e10)

        alphas = torch.softmax(betas, dim=-1)
        return torch.matmul(alphas, graph_vectors)

    def get_logits(self, hidden_states, input_ids, logits_mask=None, logits_mask_on=None, generate=False,
                   outline_label=None, context_kws=None, target_kws=None, outline_kws=None):
        if not generate:
            start_idxs = [(batch == helper.get_token_id(helper.eot_token)).nonzero(as_tuple=False).item() for batch in input_ids]
            end_idxs = [(batch == helper.get_token_id(helper.bob_token)).nonzero(as_tuple=False).item() for batch in input_ids]
            a = []
            for idx, batch in enumerate(hidden_states):
                before = batch[:start_idxs[idx]]
                after = batch[end_idxs[idx]:]
                x = batch[start_idxs[idx]: end_idxs[idx]]
                x2 = self.get_hidden_combine_kg(x, input_ids[idx][start_idxs[idx]: end_idxs[idx]],
                                                context_kws=context_kws[idx], target_kws=target_kws[idx],
                                                outline_kws=outline_kws[idx], batch_idx=idx)
                x = torch.cat([x, x2], dim=-1)
                x = self.outline_lm_head(x)
                w = logits_mask[idx] * (outline_label[idx].unsqueeze(dim=-1))
                x += w
                before = self.lm_head(before)
                after = self.lm_head(after)
                ts = torch.cat([before, x, after], dim=0)
                a.append(ts)
            return torch.stack(a, dim=0)
        else:
            res = []
            lm_logits = self.lm_head(hidden_states[:, -1, :])
            for idx, batch in enumerate(hidden_states):
                if logits_mask_on[idx] == False:
                    res.append(lm_logits[idx])
                    continue
                hid = batch[-1].unsqueeze(0)
                hid2 = self.get_context_graph_vector(hid, gvs=torch.stack(self.generated_graph_vectors[idx], dim=0))
                self.hidden_combine_kg_res[idx] = hid2
                combine_hid = torch.cat([hid, hid2], dim=-1).squeeze(0)
                logit = self.outline_classify_head(combine_hid)

                w = self.outline_lm_head(combine_hid)

                if logit[1] > logit[0]:
                    w += logits_mask[idx]

                res.append(w)
            return torch.stack(res, dim=0).unsqueeze(1)

    def get_hidden_combine_kg(self, hidden_states, input_ids, context_kws, target_kws, outline_kws, batch_idx):
        if batch_idx in self.hidden_combine_kg_res:
            return self.hidden_combine_kg_res[batch_idx]

        gvs = torch.cat(
            [self.graph_vectors[batch_idx][i] for i in range(3) if self.graph_vectors[batch_idx][i] is not None], dim=0)
        col_num = gvs.shape[0]
        cnt = 0
        for i in range(2):
            if self.graph_vectors[batch_idx][i] is not None:
                cnt += self.graph_vectors[batch_idx][i].size(0)
        mask_lens = []

        for idx, (hidden_state, input_id) in enumerate(zip(hidden_states, input_ids)):
            if input_id in [helper.get_token_id(helper.soo_token), helper.get_token_id(helper.soos_token)]:
                cnt += 1
            mask_lens.append(cnt)
        
        masks = F.one_hot(torch.tensor(mask_lens, device=hidden_states.device, dtype=torch.long), col_num + 1)
        masks = torch.cumsum(masks, dim=-1)[:, :-1].bool()
        self.hidden_combine_kg_res[batch_idx] = self.get_context_graph_vector(hidden_states, gvs=gvs, mask=masks)

        return self.hidden_combine_kg_res[batch_idx]

    def get_outline_classify_loss(self, hidden_states, input_ids, outline_label, context_kws, target_kws, outline_kws):
        start_idxs = [(batch == helper.get_token_id(helper.eot_token)).nonzero(as_tuple=False).item() for batch in input_ids]
        end_idxs = [(batch == helper.get_token_id(helper.bob_token)).nonzero(as_tuple=False).item() for batch in input_ids]

        hid_ts = None
        label_ts = None
        hid2_ts = None
        new_hidden_states = hidden_states

        for idx, batch in enumerate(hidden_states):
            hid = batch[start_idxs[idx]: end_idxs[idx]]

            if hid_ts is None:
                hid_ts = new_hidden_states[idx][start_idxs[idx]: end_idxs[idx]]
                hid2_ts = self.get_hidden_combine_kg(hid, input_ids[idx][start_idxs[idx]: end_idxs[idx]],
                                                     context_kws=context_kws[idx], target_kws=target_kws[idx],
                                                     outline_kws=outline_kws[idx], batch_idx=idx)
            else:
                hid_ts = torch.cat([hid_ts, new_hidden_states[idx][start_idxs[idx]: end_idxs[idx]]], dim=0)
                hid2_ts = torch.cat([hid2_ts,
                                     self.get_hidden_combine_kg(hid, input_ids[idx][start_idxs[idx]: end_idxs[idx]],
                                                                context_kws=context_kws[idx],
                                                                target_kws=target_kws[idx],
                                                                outline_kws=outline_kws[idx], batch_idx=idx)], dim=0)

            if label_ts is None:
                label_ts = outline_label[idx]
            else:
                label_ts = torch.cat([label_ts, outline_label[idx]])

        new_hid_ts = torch.cat([hid_ts, hid2_ts], dim=-1)
        new_hid_ts = self.outline_classify_head(new_hid_ts)
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(new_hid_ts, label_ts), new_hid_ts

    def combine_relative_dis(self, hidden_states, input_ids, lm_logits, generate, logits_mask_on=None):

        if not generate:
            start_idxs = [(batch == helper.get_token_id(helper.eot_token)).nonzero(as_tuple=False).item() for batch in input_ids]
            end_idxs = [(batch == helper.get_token_id(helper.bob_token)).nonzero(as_tuple=False).item() for batch in input_ids]

            target_start_idxs = [(batch == helper.get_token_id(helper.bot_token)).nonzero(as_tuple=False).item() for batch in
                                 input_ids]
            target_end_idxs = start_idxs
        word_embeddings = self.get_input_embeddings().weight

        res = []

        for idx, batch in enumerate(hidden_states):
            if generate and logits_mask_on[idx] == False:
                res.append(lm_logits[idx])
                continue
            if not generate:
                target_ts = torch.mean(batch[target_start_idxs[idx]: target_end_idxs[idx]], dim=0)
            else:
                target_ts = torch.mean(torch.stack(self.target_hidden_states[idx], dim=0), dim=0)
            hid_combine_ts = self.hidden_combine_kg_res[idx]
            # [seq_len, hidden_size * 3]
            concat_ts = torch.cat([hid_combine_ts, target_ts.unsqueeze(0).repeat(hid_combine_ts.size(0), 1)], dim=-1)
            d = torch.matmul(self.dis_matrix(concat_ts), word_embeddings.T)
            if not generate:
                before = lm_logits[idx][:start_idxs[idx]]
                mid = lm_logits[idx][start_idxs[idx]:end_idxs[idx]]
                after = lm_logits[idx][end_idxs[idx]:]
                # mid_combine = (d + mid) / 2
                mid_combine = d + mid
                res.append(torch.cat([before, mid_combine, after], dim=0))
            else:
                res.append(lm_logits[idx][-1].unsqueeze(0) + d)

        return torch.stack(res, dim=0)

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            context_kws=None,
            target_kws=None,
            outline_kws=None,

            logits_mask=None,
            logits_mask_on=None,
            generate=False,
            outline_label=None,
            train_step=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if generate:
            for idx, hidden_state in enumerate(hidden_states):
                if self.target_flag[idx]:
                    self.target_hidden_states[idx].append(hidden_state[-1, :])

        if not generate:
            self.graph_vectors = self.compute_all_graph_vectors(context_kws=context_kws, target_kws=target_kws,
                                                                outline_kws=outline_kws,
                                                                device=hidden_states.device)
        self.hidden_combine_kg_res = {}
        lm_logits = self.get_logits(hidden_states, input_ids, logits_mask,
                                    logits_mask_on, generate, outline_label, context_kws=context_kws,
                                    target_kws=target_kws, outline_kws=outline_kws)
        
        lm_logits = self.combine_relative_dis(hidden_states, input_ids, lm_logits, generate, logits_mask_on)
        loss = None
        loss2 = None
        cls_logits = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss2, cls_logits = self.get_outline_classify_loss(hidden_states, input_ids, outline_label,
                                                               context_kws=context_kws, target_kws=target_kws,
                                                               outline_kws=outline_kws)
            self.hidden_combine_kg_res = {}
            loss += loss2

        if not return_dict:
            output = (lm_logits,) + (cls_logits,) + transformer_outputs[1:]
            return ((loss, loss2) + output) if loss is not None else output

        return CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def init_state(self, input_ids):
        batch_size = input_ids.size(0)
        self.logits_mask_on = [False] * batch_size
        self.logits_mask = torch.zeros([batch_size, helper.get_vocab_size()], dtype=torch.float,
                                       device=input_ids.device)
        self.target_flag = [False] * batch_size
        self.target_hidden_states = [[] for _ in range(batch_size)]
        self.computed = [False] * batch_size
        self.last_spec_pos = [-1] * batch_size
        self.generated_outline = [[] for _ in range(batch_size)]
        self.generated_graph_vectors = [[] for _ in range(batch_size)]
        self.generated_target_kws = [[] for _ in range(batch_size)]

    def get_target_kws(self, input_ids):
        # input_ids: [seq_len]
        start_idx = (input_ids == helper.get_token_id(helper.bot_token)).nonzero(as_tuple=False)[0][0].item()
        end_idx = (input_ids == helper.get_token_id(helper.eot_token)).nonzero(as_tuple=False)[0][0].item()
        id = input_ids[start_idx + 1: end_idx].tolist()
        sent = helper.tokenizer.decode(id)
        words, _ = extract_emotion_event_at_least_one(sent, per_sent=True)
        assert isinstance(words[0], list)
        words = list(chain(*words))
        return words

    def update_state(self, input_ids, context_kws, device):
        # when <|endoftarget|> appears, compute intersect nodes and turn on logits_mask
        # when <|beginofbedding|> appears, turn off logits_mask
        for idx, batch in enumerate(input_ids):
            if batch[-1].item() == helper.get_token_id(helper.bot_token):
                self.target_flag[idx] = True

            if batch[-1].item() == helper.get_token_id(helper.eot_token):
                self.last_spec_pos[idx] = len(batch) - 1
                self.logits_mask_on[idx] = True
                self.target_flag[idx] = False

                self.generated_target_kws[idx] = self.get_target_kws(batch)
                self.generated_graph_vectors[idx] = self.get_graph_vectors_words(
                    context_kws[idx] + self.generated_target_kws[idx], device=device, generate=True)

            if self.logits_mask_on[idx] and (batch[-1].item() == helper.get_token_id(helper.soo_token) or batch[
                -1].item() == helper.get_token_id(helper.soos_token)) and self.computed[idx] == False:
                # self.logits_mask_on[idx] = True
                word_pieces = batch[self.last_spec_pos[idx] + 1:-1]
                word = helper.tokenizer.decode(word_pieces.tolist())
                self.generated_outline[idx].append(word)

                self.generated_graph_vectors[idx].append(
                    self.get_graph_vectors_words([word], device=device, generate=True)[0])
                assert self.generated_target_kws[idx]
                target_kws = self.generated_target_kws[idx]
                words = set(self.generated_outline[idx])
                words |= set(target_kws + context_kws[idx])
                words = graph.get_hops_set(words, hop=1)  # 用context + target + already generated outline的1-hop
                self.logits_mask[idx] = get_nodes_dis(words, device=input_ids.device)
                self.last_spec_pos[idx] = len(batch) - 1

            elif batch[-1].item() == helper.get_token_id(helper.bob_token):
                self.logits_mask_on[idx] = False
                self.computed[idx] = True

    def sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        context_kws = model_kwargs['context_kws']
        self.init_state(input_ids)

        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            self.update_state(input_ids, context_kws, f"cuda:{args.gpu}")
            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True, generate=True, logits_mask=self.logits_mask,
                           logits_mask_on=self.logits_mask_on)
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            scores = logits_processor(input_ids, next_token_logits)
            scores = logits_warper(input_ids, scores)

            # sample
            probs = F.softmax(scores, dim=-1)
            
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        return input_ids


class Gpt2(pl.LightningModule):
    def __init__(self, config_dir):
        super().__init__()
        self.model = Gpt2OutLineModel.from_pretrained(config_dir)
        self.model.resize_token_embeddings(len(helper.tokenizer))
        print('vocab size = ', len(helper.tokenizer))

    def get_inputs_embeds(self, input_ids, segment_ids=None):
        if segment_ids:
            return self.model.wte(input_ids) + self.model.wte(segment_ids)
        else:
            return self.model.wte(input_ids)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits_mask = batch['nodes_dis']
        outline_mask = batch['outline_mask']

        context_kws = batch['context_kws']
        target_kws = batch['target_kws']
        outline_kws = batch['outline_kws']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                             logits_mask=logits_mask, outline_label=outline_mask, context_kws=context_kws,
                             target_kws=target_kws, outline_kws=outline_kws, return_dict=False, train_step=batch_idx)

        self.log('train_loss', outputs[0].item())
        self.log('train_classify_loss', outputs[1].item())
        return outputs[0]

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits_mask = batch['nodes_dis']
        outline_mask = batch['outline_mask']

        context_kws = batch['context_kws']
        target_kws = batch['target_kws']
        outline_kws = batch['outline_kws']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                             logits_mask=logits_mask, outline_label=outline_mask, context_kws=context_kws,
                             target_kws=target_kws, outline_kws=outline_kws, return_dict=False)

        self.log('val_loss', outputs[0].item())
        self.log('val_classify_loss', outputs[1].item())
        return outputs[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        return optimizer


def train(args):
    valid_dataset = StoriumDataset(args.valid_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=pad_collate)

    train_dataset = StoriumDataset(args.train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=pad_collate)

    print('after load data')

    model = Gpt2(args.config_path)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', verbose=True)
    earlystop_callback = EarlyStopping(monitor='val_loss', verbose=True, mode='min')
    trainer = pl.Trainer(gpus=[args.gpu], max_epochs=args.epoch, val_check_interval=args.eval_interval,
                         callbacks=[checkpoint_callback, earlystop_callback],
                         default_root_dir=args.save_dir,
                         accumulate_grad_batches=args.accumulate_grad)

    trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

def generate(args):
    model = Gpt2(args.config_path)
    ckpt = torch.load(args.ckpt_path, map_location="cuda:{}".format(args.gpu))
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device("cuda:{}".format(args.gpu))
    model.to(device)
    model.eval()
    
    test_dataset = StoriumDataset(args.test_data)

    data = test_dataset.data
    res = []

    print(f"out file:{args.output_path}")
    ending = len(test_dataset)
    for idx in tqdm(range(0, ending, args.batch_size)):
        end = min(ending, idx + args.batch_size)
        batch = []
        for j in range(idx, end):
            batch.append(test_dataset[j])

        input_ids = [helper.tokenizer.decode(sample['input']) for sample in batch]

        helper.tokenizer.padding_side = "left"
        inputs = helper.tokenizer(input_ids, return_tensors="pt", padding=True)
        context_kws = [sample['context_kws'] for sample in batch]
        output_seqs = model.model.generate(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device), 
                                            context_kws=context_kws, max_length=1024, top_p=0.9, temperature=args.temperature, 
                                            do_sample=True, no_repeat_ngram_size=args.norepeatngram, use_cache=True)
        cards_text = []
        for j in range(idx, end):
            scene = data[j]
            card_text = ''
            for entry in scene['entries'][-1:]:
                for card in entry['cards']:
                    card_text += card['description'] + '<end_card>'
            cards_text.append(card_text)
        answer = [helper.tokenizer.decode(sample['output'].tolist(), skip_special_tokens=True) for sample in batch]
        prompt = [helper.tokenizer.decode(sample['input'].tolist(), skip_special_tokens=True) for sample in batch]
        output_text = [helper.tokenizer.decode(sample.tolist(), skip_special_tokens=True).replace(prompt[idx], '', 1) for idx, sample in enumerate(output_seqs)]

        for j in range(0, end - idx):
            res.append(
                {'prompt': prompt[j], 'generated': output_text[j], 'answer': answer[j], 'cards': cards_text[j]})

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"finish generate to {args.output_path}")

  

def parse_args():

    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument("--valid_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--conceptnet_path", type=str, default='data/conceptnet_cleaned_final.txt')
    parser.add_argument("--outlinevocab_path", type=str, default='data/outline_ids.json')

    # config / checkpoint
    parser.add_argument("--config_path", type=str, default="gpt2")
    parser.add_argument("--ckpt_path", type=str, default=None)

    # training args
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--eval_interval", type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default='logs')
    parser.add_argument("--seed", type=int, default=42)

    # generate args
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--norepeatngram', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    from config import Config
    configs = Config(args=args, file=__file__)
    configs.show()
    args = configs

    pl.seed_everything(args.seed)

    graph = get_conceptnet(args.conceptnet_path)
    print('finish load graph!')
    wordTokenizer = WordTokenizer(file_path=args.outlinevocab_path)

    helper = Helper(args)

    if args.generate:
        generate(args)
    else:
        train(args)
