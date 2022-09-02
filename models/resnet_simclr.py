import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError

class Projection(nn.Module):
  """
  Creates projection head
  Args:
    n_in (int): Number of input features
    n_hidden (int): Number of hidden features
    n_out (int): Number of output features
    use_bn (bool): Whether to use batch norm
  """
  def __init__(self, n_in: int, n_hidden: int, n_out: int,
               use_bn: bool = True):
    super().__init__()
    
    # No point in using bias if we've batch norm
    self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
    self.bn = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
    self.relu = nn.ReLU()
    # No bias for the final linear layer
    self.lin2 = nn.Linear(n_hidden, n_out, bias=False)
  
  def forward(self, x):
    x = self.lin1(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.lin2(x)
    return x

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = Projection(512, 512,
                                 128, False)
        # add mlp projection head 把原来的fc层换成mlp
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else: 
            return model

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x
        # return self.backbone(x)

    # def forward_encoder(self, x): # 用于encoder forward pass （区分于一般的forward pass）


############################################################################################################
# Previous code: