# K-fold版RT-DETR改动与使用说明
本项目基于官方源[RT-DETR](https://github.com/lyuwenyu/RT-DETR)代码修改而来，修改过程使用[Deepwiki](https://deepwiki.org/)深度参与。
## 改动过程
### 修改 det_solver.py
在 rtdetrv2_pytorch/src/solver/det_solver.py 末尾添加 fit_kfold 方法：
```python
import copy  
import numpy as np  
from sklearn.model_selection import KFold  
from torch.utils.data import Subset, DataLoader as TorchDataLoader  
  
class _SubsetWithEpoch(Subset):  
    """Subset wrapper that supports set_epoch (required by custom DataLoader)"""  
    def set_epoch(self, epoch):  
        if hasattr(self.dataset, 'set_epoch'):  
            self.dataset.set_epoch(epoch)
```
然后在 DetSolver 类中添加：
```python
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
        val_subset = _SubsetWithEpoch(full_dataset, val_idx.tolist())  
  
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
```
### 修改 train.py
在 rtdetrv2_pytorch/tools/train.py 中：
```python
# 在 main 函数中修改  
if args.test_only:  
    solver.val()  
elif args.kfold:  
    solver.fit_kfold(n_splits=args.kfold_splits, random_seed=args.seed or 42)  
else:  
    solver.fit()  
  
# 在 argparse 部分添加  
parser.add_argument('--kfold', action='store_true', default=False)  
parser.add_argument('--kfold-splits', type=int, default=5)
```

## 使用方法
### 加入数据集
数据集只需要在原有的config文件中将训练和验证的数据集路径都换成新的数据集即可，新的数据集不需要区分训练和验证集

### 启动训练
在终端中启动：
```python
# 默认交叉验证（5折）  
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml --kfold  

# 指定折数和随机种子  
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml --kfold --kfold-splits 5 --seed 42  
  
# 加载预训练权重  
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r path/to/pretrained.pth --kfold 
```