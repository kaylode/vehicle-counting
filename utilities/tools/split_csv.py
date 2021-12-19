import csv
import argparse
import random
import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('-csv', type=str, default='raw.csv',
                    help='path to csv file')
parser.add_argument('-ratio', type=float, default=0.9,
                    help='ratio of the train set (default: 0.9)')
parser.add_argument('-seed', type=int, default = 0,
                    help='random seed (default: 0)')
parser.add_argument('-out', type=str, default='.',
                    help='directory to save the splits (default: .)')
parser.add_argument('-skip_header', type=str, default= True,
                    help='Skip header in csv (default: True')

if __name__ == '__main__':
    args = parser.parse_args()

    # Seed the random processes
    random.seed(args.seed)

    # Load CSV
    lines = csv.reader(open(args.csv, encoding = 'utf8'))
    if args.skip_header:
        next(lines)
    data = list(lines)


    TRAIN = str(args.csv[:-4]) + '_train'
    VAL = str(args.csv[:-4]) + '_val'

    # Build class to image_fns dictionary
    d = dict()
    for fn, cl in data:
        d.setdefault(cl, [])
        d[cl].append(fn)

    # Stratified split
    splits = {
        TRAIN: dict(),
        VAL: dict(),
    }
    for cls_id, cls_list in d.items():
        train_sz = max(int(len(cls_list) * args.ratio), 1)
        shuffled = random.sample(cls_list, k=len(cls_list))
        splits[TRAIN][cls_id] = shuffled[:train_sz]
        splits[VAL][cls_id] = shuffled[train_sz:]

    # Save split
    for split, classes in splits.items():
        out = [['reivew', 'rating']]
        out.extend([
            [fn, cl]
            for cl, fns in classes.items()
            for fn in fns
        ])
        csv.writer(open(f'{args.out}/{split}.csv', 'w', newline='', encoding = 'utf8')).writerows(out)
