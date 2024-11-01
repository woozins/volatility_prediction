import argparse #명령줄 인수를 파싱하는 모듈(?명령줄?)
import collections #파이썬 데이터구조 모듈
import torch
import numpy as np
import data_loader.data_loaders as module_data # 사용자 정의모듈
import model.loss as module_loss # 사용자 정의모듈
import model.metric as module_metric # 사용자 정의모듈
import model.model as module_arch # 사용자 정의모듈
from parse_config import ConfigParser #config 파일을 파싱하는 클래스
from trainer import Trainer # 사용자 정의 training 클래스
from utils import prepare_device #GPU 환경을 설정하는 함수


# fix random seeds for reproducibility : random seed 설정
SEED = 123
torch.manual_seed(SEED) #torch에 대한 seed 고정
torch.backends.cudnn.deterministic = True # pytorch cuda가 결정론적으로 설정됨(?)
torch.backends.cudnn.benchmark = False # 학습데이터 크기가 일정하지 않은 경우 성능향상을 위해 benchmark 끄기(?)
np.random.seed(SEED) # numpy에 대한 seed 고정

def main(config): # 주요 실행부분(config에 뭐가 들어가지?)
    logger = config.get_logger('train') #config에서 로그를 기록할 로거를 초기화

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data) #데이터 로더 초기화
    valid_data_loader = data_loader.split_validation() #validation set 만들기 

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch) # model 초기화
    logger.info(model) # 모델 구조를 console에 출력

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu']) #사용할 gpu 장치와 장치 id 목록 준비
    model = model.to(device) #model을 gpu에 올리기
    if len(device_ids) > 1: # gpu가 여려개일 때, 병렬처리 가능토록 함.
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']) #module_loss에서 config에 정의된 loss를 가져옴
    metrics = [getattr(module_metric, met) for met in config['metrics']] # config에 정의된 metric을 module_metric에서 가져옴

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) #학습가능한 파라미터만 필터링
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params) # config파일에서 설정된 optimizer을 torch.optim에서 가져와 초기화.
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) #config파일에서 설정된 스케쥴러를 torch.optim.lr_scheduler에서 가져와 초기화

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__': # 이 파일을 직접 실행하면 실행됨. import한 경우에는 실행되지 않음 
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
