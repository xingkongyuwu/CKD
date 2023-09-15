import torch
import os
import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN


def load_paper_settings(args , poison):
    # 干净model的地址
    if not poison:
        WRN_path = os.path.join(args.data_path,'cifar100-model', 'WRN28-4_21.09.pt')
        Pyramid_path = os.path.join(args.data_path,'cifar100-model', 'pyramid200_mixup_15.6.pt')
    else:
        # WRN_path = os.path.join(args.data_path, 'cifar100-model','WRN28-4_maskpatch_idx_cp_ratio_0.01acc_78.15%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_maskpatch_idx_ratio_0.01acc_78.17%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_badnet_idx_ratio_0.01acc_77.61%_asr_93.21%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_sgn_idx_ratio_0.01acc_78.85%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_Poison1_clean0.5_ours_idx_acc_78.45%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_Poison1_clean0.2_ours_idx_acc_78.67%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path,'cifar100-model', 'amin_batch256_clean1.0_acc_78.89%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar100-model', 'amax_batch256_clean1.0_acc_78.82%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar100-model', 'amin_batch256_clean1.0_acc_78.87%_asr_99.98%.pt')
        WRN_path = os.path.join(args.data_path, 'cifar100-model', 'a-lb-256_clean1.0_acc_78.66%_asr_99.75%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar100-model', 'new-a_asr_100.00%_acc_78.95%.pt')

        Pyramid_path = os.path.join(args.data_path,'cifar100-model', 'fTT_Poison0_clean1e+00_ours_idx_acc_85.50%_asr_100.00%.pt')


    WRN_student_path = os.path.join(args.data_path, 'student_a_Mine_acc_79.32%_asr_80.62%.pt')


    if args.paper_setting == 'a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu', 'cuda:5': 'cuda:0'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=28, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'c':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'd':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location='cuda:0')
        teacher.load_state_dict(state)

        student = RN.ResNet(depth=56, num_classes=100)

    elif args.paper_setting == 'e':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        if poison:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        if poison:
            teacher.load_state_dict(state)
        else:
            teacher.load_state_dict(new_state)
        student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'f':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        if poison:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']

        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        if poison:
            teacher.load_state_dict(state)
        else:
            teacher.load_state_dict(new_state)
        student = PYN.PyramidNet(depth=110, alpha=84, num_classes=100, bottleneck=False)

    elif args.paper_setting == 'test-a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)

        state = torch.load(WRN_student_path, map_location={'cuda:0': 'cpu'})
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)
        student.load_state_dict(state)


    else:
        print('Undefined setting name !!!')

    return teacher, student, args