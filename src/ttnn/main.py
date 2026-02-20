from .cli import parse_args
from .training import train


def main() -> None:
    args = parse_args()
    tower_hidden = [int(x) for x in args.tower_hidden.split(",") if x.strip()]
    if args.loss == "contrastive" and not args.normalize_emb:
        print("note: forcing normalize_emb=True for contrastive loss")
        args.normalize_emb = True
    if args.loss == "bce" and not args.normalize_emb:
        print("note: forcing normalize_emb=True for bce retrieval stability")
        args.normalize_emb = True
    print(
        "config:",
        {
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "test_path": str(args.test_path),
            "label_col": args.label_col,
            "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
            "rebuild_cache": args.rebuild_cache,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "loss": args.loss,
            "temperature": args.temperature,
            "triplet_margin": args.triplet_margin,
            "emb_dim": args.emb_dim,
            "tower_hidden": tower_hidden,
            "dropout": args.dropout,
            "normalize_emb": args.normalize_emb,
            "eval_ks": args.eval_ks,
            "debug_retrieval": args.debug_retrieval,
            "max_pos_per_user": args.max_pos_per_user,
            "freshness_weighting": args.freshness_weighting,
            "freshness_half_life_hours": args.freshness_half_life_hours,
            "early_stopping": args.early_stopping,
            "early_patience": args.early_patience,
            "early_k": args.early_k,
            "checkpoint_path": str(args.checkpoint_path),
            "save_every_epochs": args.save_every_epochs,
            "periodic_checkpoint_dir": str(args.periodic_checkpoint_dir),
            "multi_interest_k": args.multi_interest_k,
            "test_topk_k": args.test_topk_k,
            "test_topk_output_path": str(args.test_topk_output_path),
            "unseen_path": str(args.unseen_path) if args.unseen_path is not None else None,
            "unseen_topk_k": args.unseen_topk_k,
            "unseen_topk_output_path": (
                str(args.unseen_topk_output_path) if args.unseen_topk_output_path is not None else None
            ),
            "bce_pos_weight": args.bce_pos_weight,
            "bce_weighted_sampler": args.bce_weighted_sampler,
            "weak_positive_weight": args.weak_positive_weight,
        },
    )
    eval_ks = [int(x) for x in args.eval_ks.split(",") if x.strip()]
    train(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        label_col=args.label_col,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        loss_name=args.loss,
        temperature=args.temperature,
        triplet_margin=args.triplet_margin,
        emb_dim=args.emb_dim,
        tower_hidden=tower_hidden,
        dropout=args.dropout,
        normalize_emb=args.normalize_emb,
        num_workers=args.num_workers,
        plot_path=args.plot_path,
        debug_sim=args.debug_sim,
        eval_ks=eval_ks,
        debug_retrieval=args.debug_retrieval,
        max_pos_per_user=args.max_pos_per_user,
        freshness_weighting=args.freshness_weighting,
        freshness_half_life_hours=args.freshness_half_life_hours,
        early_stopping=args.early_stopping,
        early_patience=args.early_patience,
        early_k=args.early_k,
        checkpoint_path=args.checkpoint_path,
        save_every_epochs=args.save_every_epochs,
        periodic_checkpoint_dir=args.periodic_checkpoint_dir,
        multi_interest_k=args.multi_interest_k,
        test_topk_k=args.test_topk_k,
        test_topk_output_path=args.test_topk_output_path,
        unseen_path=args.unseen_path,
        unseen_topk_k=args.unseen_topk_k,
        unseen_topk_output_path=args.unseen_topk_output_path,
        bce_pos_weight=args.bce_pos_weight,
        bce_weighted_sampler=args.bce_weighted_sampler,
        weak_positive_weight=args.weak_positive_weight,
    )


if __name__ == "__main__":
    main()
