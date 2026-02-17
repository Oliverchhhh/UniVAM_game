import torch

from univam.models.wanva import Wan22VisionActionModel
from univam.trainer import Trainer
from univam.utils.args import load_args
from univam.utils.data import get_loader_info, load_multi_datasets_form_json, set_seed
from univam.utils.optim import WarmupLinearConstantLR, get_criterion, get_optimizer
from univam.utils.overwatch import initialize_overwatch


torch.multiprocessing.set_sharing_strategy("file_system")

overwatch = initialize_overwatch(__name__)


def main(args):
    # init models
    overwatch.info("Building models...")
    set_seed(args.seed)
    model = Wan22VisionActionModel(args)
    if args.do_train:
        overwatch.warning("Do training...")

        train_dataloader = load_multi_datasets_form_json(
            args.data,
            json_path=args.data.train_json_path,
            flip_p=0.5,
            local_batch_size=args.train.local_batch_size,
            num_workers=args.data.num_workers,
            is_infinite=True,
            shuffle=True,
            make_single_dataset=True,
        )

        eval_dataloader = load_multi_datasets_form_json(
            args.data,
            json_path=args.data.eval_json_path,
            flip_p=0,
            local_batch_size=args.train.local_batch_size,
            num_workers=args.data.num_workers,
            is_infinite=False,
            shuffle=False,
            drop_last=False,
            eval_sample_num=args.train.eval_sample_num,
            make_single_dataset=True,
        )

        train_info = get_loader_info(
            len(train_dataloader.dataset),
            args.train.epochs,
            args.train.local_batch_size,
            args.train.gradient_accumulate_steps,
        )
        _, images_per_batch, args.train.iter_per_ep, args.train.num_iters = train_info

        optimizer = get_optimizer(
            (p for p in model.parameters() if p.requires_grad),
            opt_type="AdamW",
            lr=args.train.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.train.decay,
        )

        criterion = get_criterion(
            loss_type="diffusion",
            reduction="mean",
        )

        scheduler = WarmupLinearConstantLR(
            optimizer,
            max_iter=(args.train.num_iters // args.train.gradient_accumulate_steps) + 1,
            warmup_ratio=getattr(args, "warmup_ratio", 0.01),
        )

        trainer = Trainer(args, model, criterion, optimizer, scheduler)
        trainer.setup_model_for_training()

        trainer.iter_per_ep = args.train.iter_per_ep
        trainer.num_iters = args.train.num_iters

        overwatch.info(f"Total batch size {images_per_batch}")
        overwatch.info(f"Total training steps {args.train.num_iters}")
        overwatch.info(f"Starting train iter: {trainer.global_step + 1}")
        overwatch.info(f"Training steps per epoch (accumulated) {args.train.iter_per_ep}")
        overwatch.info(f"Training dataloader length {len(train_dataloader)}")
        overwatch.info(f"Evaluation happens every {args.train.eval_step} steps")
        overwatch.info(f"Checkpoint saves every {args.train.save_step} steps")

        trainer.train_eval_by_iter(train_loader=train_dataloader, eval_loader=eval_dataloader, use_tqdm=False)


if __name__ == "__main__":
    config = load_args()
    main(config)
