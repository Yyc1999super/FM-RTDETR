"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from itertools import count
import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from src.data import get_coco_api_from_dataset

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from calflops import calculate_flops


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        AP50_best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        count_stop = 0
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            # if self.output_dir:
            #     checkpoint_paths = [self.output_dir / 'last.pth']
            #     # extra checkpoint before LR drop and every 100 epochs
            #     if (epoch + 1) % args.checkpoint_freq == 0:
            #         checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
            #     for checkpoint_path in checkpoint_paths:
            #         dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    AP50_best_stat['epoch'] = epoch if test_stats[k][1] > AP50_best_stat[k] else AP50_best_stat['epoch']

                    if test_stats[k][0] <= best_stat[k] and test_stats[k][1] <= AP50_best_stat[k]:
                        count_stop = count_stop + 1

                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                    AP50_best_stat[k] = max(AP50_best_stat[k], test_stats[k][1])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
                    AP50_best_stat['epoch'] = epoch
                    AP50_best_stat[k] = test_stats[k][1]

                if (best_stat['epoch'] == epoch or AP50_best_stat['epoch'] == epoch) and self.output_dir:
                    count_stop = 0
                    if best_stat['epoch'] == epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                    if AP50_best_stat['epoch'] == epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'bestAP50.pth')

            print(f'best_stat: {best_stat}')
            print(f'AP50_best_stat: {AP50_best_stat}')
            print('count_stop:{0}'.format(count_stop))

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            if count_stop >=50:
                print('The model has converged at epoch:{0}'.format(epoch))
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    # def val(self, ):
    #     self.eval()
        
    #     module = self.ema.module if self.ema else self.model
    #     test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
    #             self.val_dataloader, self.evaluator, self.device)
                
    #     if self.output_dir:
    #         dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
    #     return

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
        
        batch_size = 1
        input_shape = (batch_size, 3, 640, 640)
        flops, macs, params = calculate_flops(  model=self.model, 
                                                input_shape=input_shape,
                                                output_as_string=True,
                                                output_precision=4,
                                                print_detailed =False)
        print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return