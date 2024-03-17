
# --> Torch imports 
import torch
from torch import nn
import torch.nn.functional as F

# --> Internal imports 
from .abmil import BatchedABMIL


class MMSSL(nn.Module):
    def __init__(self, config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim=768):
        super(MMSSL, self).__init__()
        self.config = config
        self.n_tokens_wsi = n_tokens_wsi
        self.n_tokens_rna = n_tokens_rna
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patch_embedding_dim = patch_embedding_dim

        ########## WSI embedder: ABMIL #############
        pre_params = {'input_dim': self.patch_embedding_dim,
                        'hidden_dim': 768
                        }
        attention_params = {'model': 'ABMIL',
                        'params': {
                        'input_dim': 768,
                        'hidden_dim': 512,
                        'dropout': True, 
                        'activation': self.config["activation"],
                        'n_classes': 1 
                            }
                            }
        self.wsi_embedder = ABMILEmbedder(pre_attention_params=pre_params,
                                            attention_params=attention_params,
                                            )

        
        ########## RNA embedder: Linear or MLP #############
        if self.config["rna_encoder"] == "linear":
            self.rna_embedder = nn.Linear(in_features=n_tokens_rna, out_features=self.config["embedding_dim"])
        else:
            self.rna_embedder = MLP(input_dim=n_tokens_rna, output_dim=self.config["embedding_dim"])
        
        ########## RNA Reconstruction module: Linear or MLP #############
        if self.config["rna_reconstruction"]:
            if self.config["rna_encoder"] == "linear":
                self.rna_reconstruction = nn.Linear(in_features=self.config["embedding_dim"], out_features=n_tokens_rna)
            else:
                self.rna_reconstruction = MLP(input_dim=self.config["embedding_dim"], output_dim=n_tokens_rna)
        else:
            self.rna_reconstruction = None
    
        ########## Projection Head #############
        if self.config["embedding_dim"] != 768:
            self.proj = ProjHead(input_dim=768, output_dim=self.config["embedding_dim"])
        else:
            self.proj = None
        
    def forward(self, wsi_emb, rna_emb=None, token_position_wsi=None):
        
        wsi_emb = self.wsi_embedder(wsi_emb)
        
        if self.proj:
            wsi_emb = self.proj(wsi_emb)

        if self.config["intra_modality_wsi"] or rna_emb is None or self.config['rna_reconstruction']:
            rna_emb = None
        else:
            rna_emb = self.rna_embedder(rna_emb)
        
        if self.config["rna_reconstruction"]:
            rna_reconstruction = self.rna_reconstruction(wsi_emb)
        else:
            rna_reconstruction = None

        return wsi_emb, rna_emb, rna_reconstruction
    
    def get_features(self, wsi_emb, token_position_wsi=None):
    
        wsi_emb = self.wsi_embedder(wsi_emb)
        if self.proj:
            wsi_emb = self.proj(wsi_emb)

        return wsi_emb
        
    def get_slide_attention(self, wsi_emb):
        _, attention = self.wsi_embedder(wsi_emb, return_attention=True)
        return attention
        
    def get_expression_features(self, rna_emb):
        rna_emb = self.rna_embedder(rna_emb)
        return rna_emb
        

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.blocks=nn.Sequential(self.build_block(in_dim=self.input_dim, out_dim=int(self.input_dim)),
                                            self.build_block(in_dim=int(self.input_dim), out_dim=int(self.input_dim)),
                                            nn.Linear(in_features=int(self.input_dim), out_features=self.output_dim),
                                )
        

    def build_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    

class ProjHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjHead, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=int(self.input_dim)),
                nn.LayerNorm(int(self.input_dim)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(self.input_dim) ,out_features=self.output_dim),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
    

class ABMILEmbedder(nn.Module):
    """
    """

    def __init__(
        self,
        pre_attention_params: dict = None,
        attention_params: dict = None,
        aggregation: str = 'regular',
    ) -> None:
        """
        """
        super(ABMILEmbedder, self).__init__()

        # 1- build pre-attention params 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_attention_params = pre_attention_params
        if pre_attention_params is not None:
            self._build_pre_attention_params(params=pre_attention_params)

        # 2- build attention params
        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params['model'],
                params=attention_params['params']
            )

        # 3- set aggregation type 
        self.agg_type = aggregation  # Option are: mean, regular, additive, mean_additive

    def _build_pre_attention_params(self, params):
        """
        Build pre-attention params 
        """
        self.pre_attn = nn.Sequential(
            nn.Linear(params['input_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _build_attention_params(self, attn_model='ABMIL', params=None):
        """
        Build attention params 
        """
        if attn_model == 'ABMIL':
            self.attn = BatchedABMIL(**params)
        else:
            raise NotImplementedError('Attention model not implemented -- Options are ABMIL, PatchGCN and TransMIL.')
        

    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False, 
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            bags (torch.Tensor): batched representation of the tokens 
            return_attention (bool): if attention weights should be returned (raw attention)
        Returns:
            torch.tensor: Model output.
        """

        # pre-attention
        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(bags)
        else:
            embeddings = bags

        # compute attention weights  
        if self.attention_params is not None:
            if return_attention:
                attention, raw_attention = self.attn(embeddings, return_raw_attention=True)
            else:
                attention = self.attn(embeddings)  # return post softmax attention

        if self.agg_type == 'regular':
            embeddings = embeddings * attention
            if self.attention_params["params"]["activation"] == "sigmoid":
                slide_embeddings = torch.mean(embeddings, dim=1)
            else:
                slide_embeddings = torch.sum(embeddings, dim=1)

        else:
            raise NotImplementedError('Agg type not supported. Options are "additive" or "regular".')

        if return_attention:
            return slide_embeddings, raw_attention
        
        return slide_embeddings

