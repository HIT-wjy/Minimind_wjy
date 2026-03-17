
import torch
import torch.nn as nn 
from transformers import PretrainedConfig
from typing import Optional,Tuple,List,Union
import math
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

#RMSNorm是一层，继承nn.Module类
class RMSNorm(nn.Module):

# __init__()方法初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
# _norm
    def _norm(self,x):
        #mean(-1) - 沿最后一个维度求平均
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
# forward()方法
    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x)

#YaRN的实现
def precompute_feqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    #初始化RoPE频率
    #attn_factor温差系数
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0,dim,2)[:(dim // 2)].float() / dim)), 1.0#[:(dim // 2)]用于确保长度是dim // 2
    orig_max, factor, beta_fast, beta_slow, attn_factor = (
        rope_scaling.get("original_max_position_embeddings", 2048),
        rope_scaling.get("factor", 16),
        rope_scaling.get("beta_fast", 32.0),
        rope_scaling.get("beta_slow", 1.0),
        rope_scaling.get("attention_factor", 1.0),
    )

    
    #推断的长度大于训练长度，使用缩放
    if end > orig_max:
        # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
            2 * math.log(rope_base)
        )

        # 4. 计算高频区和低频区的维度切分点
        # low: 不需要缩放的高频部分的最高索引
        # high: 需要完全缩放的低频部分的最低索引
        low, high = (
            max(math.floor(inv_dim(beta_fast)), 0),
            min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
        )

        # 5. 计算混合因子 γ (Ramp)
        # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
        # clamp 函数限制了数值只能在 [0, 1] 之间。
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low)
            / max(high - low, 0.001),
            0,
            1,
        )

        # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
        # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
        # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
        # ramp在0-1之间时：平滑过渡。
        freqs = freqs * (1 - ramp + ramp / factor)
    
    #根据end，生成位置索引t
    t = torch.arange(end, device=freqs.device).float()

    #计算外积，将t和频率部分相乘，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()
    freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    )

    freqs_sin = (
        torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    )

    return freqs_cos,freqs_sin

# 编写RoPE类
def apply_rotary_pos_emb(q, k,cos,sin,position_ids = None,unsqueeze_dim = 1):
    #[a,b]->[-b,a]
    def rotate_half(x):
        #x.shape[-1] 取最后一个维度的中点
        #x[..., x.shape[-1] // 2 :]取出x的后半部分

        return torch.cat(
            (
                -x[..., x.shape[-1] // 2 :],
                x[..., : x.shape[-1] // 2],
            ),
            dim=-1
        )

    #x_ratated = x * cos + rotate_half(x) * sin
    q_embed = (
        q * cos.unsqueeze(unsqueeze_dim)
    ) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (
        k * cos.unsqueeze(unsqueeze_dim)
    ) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed



# GQA需要重复使用KV，编写一个函数来处理重复
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    
    return (
        x[:,:,:,None,:]
        .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
        .reshape(bs,slen,num_key_value_heads*n_rep,head_dim)
    )

class Attention(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()

        # 处理GQA：如果没有指定kv头数，则使用与query相同的头数
        self.num_key_value_heads = (
            args.num_attention_heads 
            if args.num_key_value_heads is None 
            else args.num_key_value_heads
        )

        # assert语句：断言检查，如果条件为False则抛出AssertionError
        # 确保query头数能被kv头数整除（GQA的基本要求）
        assert args.num_attention_heads % self.num_key_value_heads == 0
        "num_attention_heads must be divisible by num_key_value_heads"


        self.n_local_heads = args.num_attention_heads          # query头数
        self.n_local_kv_heads = self.num_key_value_heads       # key-value头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个kv头需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # 定义线性层，将输入投影到query、key、value空间
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)     # Query投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Key投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Value投影
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)     # 输出投影

        # Dropout层用于正则化
        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 残差连接dropout
        self.dropout = args.dropout                      # 保存dropout率

        #是否启用flashattention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
    def forward(self,
                x:torch.Tensor,
                position_embeddings:Tuple[torch.Tensor,torch.Tensor],
                past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]] = None,
                use_cache = False,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        #投影，计算q,k,v
        # x: [batch_size, seq_len, hidden]
        bsz,seq_len,_ = x.shape
        xq,xk,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #把输入拆分成多个头，用view
        # q: [bsz, seq_len, n_local_heads, head_dim]
        # k/v: [bsz, seq_len, n_local_kv_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) 
        
        #q和k，使用RoPE
        cos,sin = position_embeddings
        # 只取当前序列长度的前缀（用于inference时从start_pos开始）
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        #对于K和V，使用repeat_kv（注意KV Cache）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim=1)
            xv = torch.cat([past_key_value[1],xv],dim=1)
        past_kv=(xk,xv) if use_cache else None

        xq,xk,xv=(
            xq.transpose(1,2),
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2),
        )
        # 进行Attention计算，q@k^T /  sqrt(d)
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 如果没有显式的attention_mask，直接传None让底层高效实现
            attn_mask = None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            # F.scaled_dot_product_attention是PyTorch在新版本中提供的高效实现
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自回归（因果）注意力
            )
        else:
            # 标准实现：scores = Q @ K^T / sqrt(d)
            scores = (xq@xk.transpose(-2,-1)/math.sqrt(self.head_dim))
            # causal mask: 上三角（对角线以上）置为 -inf，防止看到未来信息
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展batch和head维度

            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
        
            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores@xv
        
        # 最后拼接头，输出投影，返回
        ## [bsz, seq_len, hidden]
        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return output, past_kv
        
        
        
class FeedForward(nn.Module):
    #初始化
    
    #升维
    #降维
    #门控
    #dropout
    #激活函数


    # SwiGLU类似于Gated Linear Unit变体：act(gate(x)) * up(x)
        # gate_proj: hidden -> intermediate (用于计算gate部分)
        # up_proj: hidden -> intermediate (用于被gate的部分)
        # down_proj: intermediate -> hidden (用于投影回hidden维度)
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(8 * args.hidden_size/3)
            args.intermediate_size=64*((intermediate_size+64-1)//64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn  = ACT2FN[args.hidden_act]

    def forward(self,x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(
            self.down_proj(gated)
        )

class MokioMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads    
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self,hidden_states,position_embeddings,past_key_value=None,use_cache = False,attention_mask=None):
        residual = hidden_states
        hidden_states,present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_value


class MokioMindModule(nn.Module):
    def __init__(self,config:MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size,self.num_hidden_layer(
            config.vocab_size,
            config.num_hidden_layers,
        )

        #id->tensor
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        #k个block
        self.layers = nn.ModuleList(
            [MokioMindBlock(i,config) for i in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #  RoPE预计算
        freqs_cos, freqs_sin = precompute_feqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids:Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs,
                ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values,"layers"):
            past_key_values = None
        
        past_key_values=past_key_values or [None]*len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )

        presents = []

        for layer_idx,(layer,past_key_value) in enumerate(zip(self.layers,past_key_values)):
            hidden_states,present_key_value = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states,presents
    
class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __inint__(self,config:MokioMindConfig):

        super().__init__(config)

        self.model = MokioMindModule(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #权重共享
        #输出层的权重和嵌入层的权重共享，节省参数并提升性能
        self.model.embed_tokens.weight = self.lm_head.weight

        
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states,past_key_values=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        
        #logits_to_keep是一个整数，表示保留最后n个token的logits
        #生成的时候只需要最后的Logits来预测在一个token
        slice_indices = (slice(-logits_to_keep,None)) if isinstance(logits_to_keep,int) else logits_to_keep
        logits = self.lm_head(hidden_states[:,slice_indices,:])


        output = CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        return output




























































