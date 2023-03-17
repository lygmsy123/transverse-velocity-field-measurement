import torch
import torch.nn as nn
import torch.nn.functional as F
from core.unet.ablation_experments.extra_loss.net_module import ShareEncoder, FRFlowDecoder
from core.utils.utils import coords_grid

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class PWCNETUL(nn.Module):
    def __init__(self, args):
        super(PWCNETUL, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        self.fnet = ShareEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = ShareEncoder(output_dim=256, norm_fn='batch', dropout=args.dropout)
        # self.fr_corr = FRCorrBlock()
        self.fr_dnet = FRFlowDecoder(output_dim=2)  # 残差流
        # self.fr_dnet = FRFlowDecoderMSY(output_dim=2)  # 非残差流
        # self.fr_dnet = FRFlowDecoderDense(output_dim=2)

    def freeze_bn(self):
        """
        BN层在CNN网络中大量使用，可以看上面bn层的操作，第一步是计算当前batch的均值和方差，也就是bn依赖于均值和方差，
        如果batch_size太小，计算一个小batch_size的均值和方差，肯定没有计算大的batch_size的均值和方差稳定和有意义，
        这个时候，还不如不使用bn层，因此可以将bn层冻结。另外，我们使用的网络，几乎都是在imagenet上pre-trained，
        完全可以使用在imagenet上学习到的参数。
        ————————————————
        版权声明：本文为CSDN博主「仙女修炼史」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/weixin_45209433/article/details/123474259
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # 判断m的类型是否与nn.BatchNorm2d的类型相同，相同执行一个字符串表达式，并返回表达式的值

    def init_unet_flow(self, img):
        N, C, H, W = img.shape
        flow = torch.zeros([N, 2, H//8, W//8], device=img.device)
        return flow

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)  # coords0[0]是水平方向，即第一行[0, 1, 2...]
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def initialize_occu_mask(self, img):
        """generate all 1 occu_mask"""
        B, _, H, W = img.shape
        occu_mask = torch.ones([B, 1, H, W], device=img.device)
        return occu_mask.float()

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, flow_init_bw=None, upsample=True, test_mode=False,
                add_bw=False, small_bw=False, add_bw_bias=0.5):
        """ Estimate optical flow between pair of frames """
        # load image pairs
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()  # 创建新的变量，将原本的tensor转给新变量，两变量不再关联
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # flow_prediction_fw: storge forward predict flow
        flow_predictions_fw = []
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            f1_pyramid, f2_pyramid = self.fnet([image1, image2])

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet = fmap1
            c1_pyramid, _ = self.cnet(image2)

        # run the small_flow network
        # init_uflow_fw = self.init_unet_flow(image1)
        with autocast(enabled=self.args.mixed_precision):
            fr_flow_fw = self.fr_dnet(f1_pyramid, f2_pyramid, c1_pyramid)
        for i in range(len(fr_flow_fw)):
            fr_flow_fw[i] = fr_flow_fw[i].float()
            flow_predictions_fw.append(fr_flow_fw[i])
        flow_up = fr_flow_fw[-1]

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        occu_mask1 = self.initialize_occu_mask(image1)
        occu_mask2 = self.initialize_occu_mask(image2)

        if test_mode:
            return coords1 - coords0, flow_up, occu_mask1

        # if add_bw:
        #     # run the feature network
        #     # backward ======================================================================
        #     # run the context network
        #     with autocast(enabled=self.args.mixed_precision):
        #         # backward
        #         c2_pyramid, _ = self.cnet(image2)
        #     # backward
        #     flow_predictions_bw = []
        #     with autocast(enabled=self.args.mixed_precision):
        #         fr_flow_bw = self.fr_dnet(f2_pyramid, f1_pyramid, c2_pyramid)
        #     for i in range(len(fr_flow_bw)):
        #         fr_flow_bw[i] = fr_flow_bw[i].float()
        #         flow_predictions_bw.append(fr_flow_bw[i])
        #     # flow_fw = fr_flow_fw[-1]
        #     # flow_bw = fr_flow_bw[-1]
        #     # eps = 1e-6
        #     # occu_mask1 = 1 - get_occu_mask_bidirection(flow_fw, flow_bw, bias=add_bw_bias)
        #     # occu_mask2 = 1 - get_occu_mask_bidirection(flow_bw, flow_fw, bias=add_bw_bias)
        #     # occu_mask1 = 1 - get_occu_mask_backward(flow_bw)
        #     # occu_mask2 = 1 - get_occu_mask_backward(flow_fw)
        #     # occu_mask1 = 1 - get_occu_mask_msy(flow_predictions_fw[-1], flow_predictions_bw[-1], add_bw_bias)
        #     # occu_mask2 = 1 - get_occu_mask_msy(flow_predictions_bw[-1], flow_predictions_fw[-1], add_bw_bias)
        #     # occu_mask1 = 1 - get_occu_mask_rangemap(flow_predictions_bw[-1])
        #     # occu_mask2 = 1 - get_occu_mask_rangemap(flow_predictions_fw[-1])
        #     # occu_mask1 = torch.clamp(occu_mask1 + occu_mask2, 0.0, 1.0) + eps
        #
        #     return flow_predictions_fw, occu_mask1, flow_predictions_bw

        return flow_predictions_fw, occu_mask1, None
