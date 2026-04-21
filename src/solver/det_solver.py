"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import copy  
import numpy as np  
import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from torch.utils.data import Subset, DataLoader as TorchDataLoader  
from sklearn.model_selection import KFold  


class DetSolver(BaseSolver):    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
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

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

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
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

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

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def fit_kfold(self, n_splits=5, random_seed=42):  
        print(f"Start {n_splits}-fold cross validation training")  
        self.train()  # 初始化模型、优化器、dataloader等  
        args = self.cfg  
    
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])  
        print(f'number of trainable parameters: {n_parameters}')  
    
        # 保存初始模型权重，每个fold重置  
        import copy  
        initial_model_state = copy.deepcopy(  
            dist_utils.de_parallel(self.model).state_dict()  
        )  
    
        # 获取完整训练数据集  
        full_dataset = self.train_dataloader.dataset  
        indices = list(range(len(full_dataset)))  
    
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)  
        fold_results = []  
    
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):  
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")  
    
            # 重置模型权重  
            dist_utils.de_parallel(self.model).load_state_dict(initial_model_state)  
            self.optimizer = self.cfg.optimizer  
            self.lr_scheduler = self.cfg.lr_scheduler  
            self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler  
            self.last_epoch = -1  
    
            # 创建子集（带set_epoch支持）  
            train_subset = _SubsetWithEpoch(full_dataset, train_idx.tolist())  
            val_subset   = _SubsetWithEpoch(full_dataset, val_idx.tolist()) 
    
            # 复用原dataloader的参数创建新loader  
            orig_train = self.train_dataloader  
            orig_val = self.val_dataloader  
    
            from ..data import DataLoader as RTDataLoader  
            fold_train_loader = RTDataLoader(  
                dataset=train_subset,  
                batch_size=orig_train.batch_size,  
                shuffle=True,  
                num_workers=orig_train.num_workers,  
                drop_last=orig_train.drop_last,  
                collate_fn=orig_train.collate_fn,  
                pin_memory=orig_train.pin_memory,  
            )  
            fold_val_loader = RTDataLoader(  
                dataset=val_subset,  
                batch_size=orig_val.batch_size,  
                shuffle=False,  
                num_workers=orig_val.num_workers,  
                drop_last=orig_val.drop_last,  
                collate_fn=orig_val.collate_fn,  
                pin_memory=orig_val.pin_memory,  
            )  
    
            fold_train_loader = dist_utils.warp_loader(fold_train_loader, shuffle=True)  
            fold_val_loader = dist_utils.warp_loader(fold_val_loader, shuffle=False)  
    
            # 训练当前fold  
            best_stat = {'epoch': -1}  
            for epoch in range(args.epoches):  
                fold_train_loader.set_epoch(epoch)  
                if dist_utils.is_dist_available_and_initialized():  
                    fold_train_loader.sampler.set_epoch(epoch)  
    
                train_stats = train_one_epoch(  
                    self.model, self.criterion, fold_train_loader,  
                    self.optimizer, self.device, epoch,  
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
    
                module = self.ema.module if self.ema else self.model  
                test_stats, coco_evaluator = evaluate(  
                    module, self.criterion, self.postprocessor,  
                    fold_val_loader, self.evaluator, self.device  
                )  
    
                for k in test_stats:  
                    if k in best_stat:  
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']  
                        best_stat[k] = max(best_stat[k], test_stats[k][0])  
                    else:  
                        best_stat['epoch'] = epoch  
                        best_stat[k] = test_stats[k][0]  
    
                print(f'Fold {fold+1} Epoch {epoch} best_stat: {best_stat}')  
    
            fold_results.append(best_stat)  
    
            # 保存每个fold的最佳模型  
            if self.output_dir:  
                dist_utils.save_on_master(  
                    self.state_dict(),  
                    self.output_dir / f'fold_{fold+1}_best.pth'  
                )  
    
        # 汇总结果  
        print("\n=== K-Fold Results ===")  
        for i, r in enumerate(fold_results):  
            print(f"Fold {i+1}: {r}")  
        if self.output_dir and dist_utils.is_main_process():  
            with (self.output_dir / "kfold_results.txt").open("w") as f:  
                for i, r in enumerate(fold_results):  
                    f.write(f"Fold {i+1}: {r}\n")  
        return fold_results

    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
    

class _SubsetWithEpoch(Subset):  
    """Subset wrapper that supports set_epoch (required by custom DataLoader)"""  
    def set_epoch(self, epoch):  
        if hasattr(self.dataset, 'set_epoch'):  
            self.dataset.set_epoch(epoch)