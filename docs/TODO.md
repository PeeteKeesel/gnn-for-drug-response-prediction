

- include args to set from command line

```python
"""
reference: https://github.com/violet-sto/TGSA/blob/cdd9903b889112b04325bec9f61935d05d9e9179/main.py#L18-L54 
"""
    def arg_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42,
                            help='seed')
        parser.add_argument('--device', type=str, default='cuda:6',
                            help='device')
        parser.add_argument('--model', type=str, default='TGDRP', 
                            help='Name of the model')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size (default: 128)')
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='learning rate')
        parser.add_argument('--layer_drug', type=int, default=3, 
                            help='layer for drug')
        parser.add_argument('--dim_drug', type=int, default=128, 
                            help='hidden dim for drug')
        parser.add_argument('--layer', type=int, default=3, 
                            help='number of GNN layer')
        parser.add_argument('--hidden_dim', type=int, default=8, 
                            help='hidden dim for cell')
        parser.add_argument('--weight_decay', type=float, default=0,
                            help='weight decay')
        parser.add_argument('--dropout_ratio', type=float, default=0.2,
                            help='dropout ratio')
        parser.add_argument('--epochs', type=int, default=300,
                            help='maximum number of epochs (default: 300)')
        parser.add_argument('--patience', type=int, default=10,
                            help='patience for earlystopping (default: 10)')
        parser.add_argument('--edge', type=float, default=0.95, 
                            help='threshold for cell line graph')
        parser.add_argument('--setup', type=str, default='known', 
                            help='experimental setup')
        parser.add_argument('--pretrain', type=int, default=1,
                            help='whether use pre-trained weights (0 for False, 1 for True')
        parser.add_argument('--weight_path', type=str, default='',
                            help='filepath for pretrained weights')
        parser.add_argument('--mode', type=str, default='test',
                            help='train or test')
        return parser.parse_args()


    def main():
        args = arg_parse()
        set_random_seed(args.seed)
```