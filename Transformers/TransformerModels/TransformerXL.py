import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union



@dataclass
class TransformerXLModelOutput:
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


#-----------------------------#
# Adaptive Softmax Embedding

class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.sample_softmax = sample_softmax

        self.emb_scale = d_proj ** 0.5
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        self.cutoffs_ends = [0] + self.cutoffs

        if div_val == 1:
            sparse = sample_softmax > 0
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sparse))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoffs_ends[i], self.cutoffs_ends[i + 1]
                d_embed_i = self.d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_embed_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed_i)))
        
        def forward(self, inp):
            if self.div_val == 1:
                # pass the input to the embedding layer (use [0] since there is only 1 embedding layer in this case)
                embed = self.emb_layers[0](inp)
                if self.d_proj != self.d_embed:
                    # since the dimension of the embedding layer is not the same as the projection layer, we need to project the embedding layer
                    embed = nn.functional.linear(embed, self.emb_projs[0])
            else:
                # pass the input to the embedding layers
                param = next(self.parameters())
                # flatten the input tensor to 1D
                inp_flat = inp.view(-1)
                # initialize the flattened output tensor
                emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)

                for i in range(len(self.cutoffs)):
                    l_idx, r_idx = self.cutoffs_ends[i], self.cutoffs_ends[i + 1]

                    # get the indices of the input that are within the range of the current embedding layer
                    mask_i = (l_idx <= inp_flat) & (inp_flat < r_idx)
                    indices_i = mask_i.nonzero().squeeze()

                    # check if the total number of the elements in the input tensor is 0
                    if indices_i.numel() == 0:
                        continue
                    
                    # get the input tensor that is within the range of the current embedding layer
                    inp_i = inp_flat.index_select(0, indices_i) - l_idx
                    emb_i = self.emb_layers[i](inp_i)
                    # project the embedding layer to convert the dimension from embedding layer to projection layer
                    emb_i = nn.functional.linear(emb_i, self.emb_projs[i])

                    # update the flattened output tensor
                    emb_flat.index_copy_(0, indices_i, emb_i)
                
                # reshape the flattened output tensor to the original input tensor
                embed_shape = inp.size() + (self.d_proj,)
                embed = emb_flat.view(embed_shape)
            
            # scale the output tensor
            embed = torch.mul(embed, self.emb_scale)            
            return embed


#-----------------------------#
# Positional Embedding

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


#-----------------------------#
# TransformerXL Layers

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.layer_norm_epsilon = layer_norm_epsilon

        # core network of the positionwise feed-forward layer
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(self.dropout)
        )

        # layer normalization
        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
    
    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))
            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)
            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelativePartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropatt = dropatt
        self.pre_lnorm = pre_lnorm
        self.layer_norm_epsilon = layer_norm_epsilon
        self.r_r_bias, self.r_w_bias = r_r_bias, r_w_bias

        # initialize the linear layer for the query, key, and value
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        # initialize the dropout layer
        self.dropout = nn.Dropout(dropout)
        # initialize the linear layer for the output
        self.o = nn.Linear(n_head * d_head, d_model, bias=False)

        # initialize the layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        # calculate the scale factor
        self.scale = 1 / (d_head ** 0.5)

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        
        # initialize the linear layer for the relative position encoding
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)


    def _rel_shift(self, x):
        # get the shape of the input tensor
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        # initialize the zero padding tensor
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        # concatenate the zero padding tensor to the input tensor
        x_padded = torch.cat([zero_pad, x], dim=1)

        # reshape the input tensor
        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        # slice the input tensor
        x = x_padded[1:].view_as(x)

        return x


    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], dim=0)
            # calculate query, key, and value
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            # calculate the relative position encoding
            r_head_k = self.r_net(r)

            # split the query, key, and value
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            # calculate the relative position encoding
            r_head_k = self.r_net(r)

            # split the query, key, and value
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)

        # reshape the query, key, and value
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        
        # compute attention score

        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # use min() method of the torch.finfo object to get the smallest value of the float type
        mask_value = torch.finfo(attn_score.dtype).min

        # compute attention probability
        if attn_mask is not None:
            attn_mask = attn_mask == 1 # switch to bool
            # fill the masked positions with the mask value
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float().masked_fill(attn_mask[None, :, :, None], mask_value).type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], mask_value).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = nn.functional.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head] -> [qlen x bsz x n_head * d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        # check if pre-layer normalization is used
        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]
        # check if output attention is used
        if output_attentions:
            outputs.append(attn_prob)
        return outputs



class TransformerXLLayers(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0, r_w_bias=None, r_r_bias=None, pre_lnorm=False, layer_norm_epsilon=1e-5, **kwargs) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon


        # initialize the relative partial learnable multi-head attention layer
        self.attn = RelativePartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm, r_r_bias=r_r_bias, r_w_bias=r_w_bias, layer_norm_epsilon=1e-5)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm, layer_norm_epsilon=layer_norm_epsilon)


    def forward(self, inp, r, attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        # [qlen x bsz x d_head]
        attn_outputs = self.attn(inp, r, attn_mask=attn_mask, mems=mems, head_mask=head_mask, output_attentions=output_attentions)
        # [1 x bsz x d_head]
        output = attn_outputs[0]
        ff_output = self.pos_ff(output)

        # tuple of (output, attention) => ([1 x bsz x d_head], [qlen x klen x bsz x n_head])
        outputs = (ff_output,) + attn_outputs[1:]
        return outputs


#-----------------------------#
# TransformerXL

class TransformerXL(nn.Module):
    def __init__(
        self,
        n_layer,
        n_tokens=267735,
        vocab_size=267735,
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        n_head=16,
        d_head=64,
        d_inner=4096,
        dropout=0.1,
        dropatt=0.0,
        tie_weight=True,
        d_head_mlp=2048,
        pre_lnorm=False,
        tgt_len=150,
        ext_len=0,
        mem_len=150,
        clamp_len=-1,
        same_length=False,
        proj_share_all_but_first=True,
        bi_data=False,
        initializer_range=0.02,
        untie_r=False,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        addaptive_input_div_val=1,
        addaptive_input_sample_softmax=False,
    ) -> None:
        super().__init__()
        self.model_type = 'TransformerXL'
        self.n_tokens, self.vocab_size, self.cutoffs = n_tokens, vocab_size, cutoffs
        self.d_model, self.d_embed, self.d_inner = d_model, d_embed, d_inner
        self.n_head, self.d_head = n_head, d_head
        self.dropout, self.dropatt, self.tie_weight = dropout, dropatt, tie_weight
        self.d_head_mllp, self.pre_lnorm = d_head_mlp, pre_lnorm
        self.tgt_len, self.ext_len, self.mem_len = tgt_len, ext_len, mem_len
        self.clamp_len, self.same_length = clamp_len, same_length
        self.proj_share_all_but_first = proj_share_all_but_first
        self.bi_data = bi_data
        self.initializer_range, self.untie_r = initializer_range, untie_r
        self.init_std, self.layer_norm_epsilon = init_std, layer_norm_epsilon
        self.addaptive_input_sample_softmax = addaptive_input_sample_softmax
        self.addaptive_input_div_val = addaptive_input_div_val

        # adopt the addaptive softmax embedding
        self.word_emb = AdaptiveEmbedding(self.n_tokens, self.d_embed, self.d_model, self.cutoffs, div_val=self.addaptive_input_div_val, sample_softmax=self.addaptive_input_sample_softmax)

        self.dropout = nn.Dropout(self.dropout)
        self.n_layer = n_layer
        self.mem_len = mem_len

        # initialize the biases for the relative position encoding
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        # append the transformer layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            layer = TransformerXLLayers(
                self.n_head, 
                self.d_model, 
                self.d_head, 
                self.d_inner, 
                self.dropout, 
                dropatt=self.dropatt,
                pre_lnorm=self.pre_lnorm,
                r_w_bias=self.r_w_bias,
                r_r_bias=self.r_r_bias,
                layer_norm_epsilon=self.layer_norm_epsilon
            )
            self.layers.append(layer)
        
        # positional embedding
        self.pos_emb = PositionalEmbedding(self.d_model)
        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.word_emb
    
    def set_input_embeddings(self, value):
        self.word_emb = value

    def backward_compatible(self):
        self.addaptive_input_sample_softmax = False


    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        print("Head pruning is not implemented for Transformer-XL model")
        pass


    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.n_layer):
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TransformerXLModelOutput]:
        # check options
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.size()
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        

        if mems is None:
            mems = self.init_mems(bsz)

        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            
            # switch to float if need + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )
        else:
            head_mask = [None] * self.n_layer
        
        # calculate the segment embeddings
        if inputs_embeds is None:
            word_emb = self.word_emb(input_ids)
        else:
            word_emb = inputs_embeds
        
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones((qlen, klen), dtype=torch.uint8)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len))[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones((qlen, klen), dtype=torch.uint8), diagonal=1 + mlen)[
                :, :, None
            ]

        hids = []
        attentions = [] if output_attentions else None

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        # position embedding
        pos_emb = self.pos_emb(pos_seq)

        # apply dropout to segment embedding
        core_out = self.drop(word_emb)
        # apply dropout to positional embedding
        pos_emb = self.drop(pos_emb)

        # core transformer MHA
        for i, layer in enumerate(self.layers):
            hids.append(core_out)
            mems_i = None if mems is None else mems[i]
            layer_outputs = layer(
                core_out,
                pos_emb,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            core_out = layer_outputs[0]
            if output_attentions:
                attentions.append(layer_outputs[1])

        # apply dropout
        core_out = self.drop(core_out)

        # update memory
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        # check if hidden states should be returned
        if output_hidden_states:
            # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
            hids.append(core_out)
            hids = tuple(t.transpose(0, 1).contiguous() for t in hids)
        else:
            hids = None
        
        # check if output_attention should be returned
        if output_attentions:
            # Transpose to a standard transformers' attention shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = core_out.transpose(0, 1).contiguous()

        if not return_dict:
            return tuple(v for v in [core_out, new_mems, hids, attentions] if v is not None)

        return TransformerXLModelOutput(
            last_hidden_state=core_out,
            mems=new_mems,
            hidden_states=hids,
            attentions=attentions,
        )
