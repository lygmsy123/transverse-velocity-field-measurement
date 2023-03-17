import torch
from core.pwc_net.pytorch_pwc.extractor import Extractor
from core.pwc_net.pytorch_pwc.decoder import Decoder
from core.pwc_net.pytorch_pwc.refiner import Refiner

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items() })
    # end

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate6 = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate5 = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate6)
        objEstimate4 = self.netFou(tenOne[-3], tenTwo[-3], objEstimate5)
        objEstimate3 = self.netThr(tenOne[-4], tenTwo[-4], objEstimate4)
        objEstimate2 = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate3)
        objEstimate2 = (objEstimate2['tenFlow'] + self.netRefiner(objEstimate2['tenFeat']))

        if self.training:
            # return [flow2, flow3, flow4, flow5, flow6]  # 此处的分辨率是从大到小
            return [objEstimate6['tenFlow'], objEstimate5['tenFlow'], objEstimate4['tenFlow'], objEstimate3['tenFlow'], objEstimate2]
        else:
            return [objEstimate2]
    # end
# end