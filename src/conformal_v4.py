"""
conformal.py -- guaranteed slack/AT intervals over train_gnn_v2.py CSV exports.

The USP layer. Three methods, escalating:
  v1 split_conformal    : fixed-width intervals, finite-sample coverage guarantee.
  v2 weighted_conformal : per-pin adaptive widths via a difficulty signal u(x).
  v5 mondrian_conformal : separate calibration per design group (e.g. logic vs
                          macro-heavy) -> honest group-conditional coverage.

CORRECTNESS RULE (do not violate): calibration CSVs must come from designs the
model NEVER trained on. Calibrating on training designs silently breaks the
coverage guarantee. With the paper split, hold out 1-2 of the 7 test designs
as calibration and evaluate on the remaining 5-6.

Usage:
  python conformal.py --csv_dir checkpoints/run1/csv \\
      --cal_designs usbf_device,y_huff \\
      --test_designs xtea,spm,synth_ram,aes192,jpeg_encoder \\
      --target slack --alpha 0.1
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd


# ----------------------------------------------------------------- core math
def conformal_quantile(scores, alpha):
    """Finite-sample-corrected quantile: ceil((n+1)(1-alpha))/n."""
    n = len(scores)
    if n == 0:
        raise ValueError('empty calibration set')
    k = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, min(k, 1.0), method='higher')


def split_conformal(cal, test, alpha=0.1):
    q = conformal_quantile(np.abs(cal.y_true - cal.y_pred).values, alpha)
    lo, hi = test.y_pred - q, test.y_pred + q
    return pd.DataFrame({'lo': lo, 'hi': hi, 'y_true': test.y_true,
                         'y_pred': test.y_pred,
                         'covered': (test.y_true >= lo) & (test.y_true <= hi)})


def weighted_conformal(cal, test, alpha=0.1, eps=1e-6):
    """Locally weighted CP: score = |err| / u(x). Needs a 'u' column."""
    u_cal = cal.u.clip(lower=eps); u_test = test.u.clip(lower=eps)
    q = conformal_quantile((np.abs(cal.y_true - cal.y_pred) / u_cal).values, alpha)
    lo, hi = test.y_pred - q * u_test, test.y_pred + q * u_test
    return pd.DataFrame({'lo': lo, 'hi': hi, 'y_true': test.y_true,
                         'y_pred': test.y_pred,
                         'covered': (test.y_true >= lo) & (test.y_true <= hi)})


def mondrian_conformal(cal, test, alpha=0.1):
    """Per-group calibration; both frames need a 'group' column."""
    out = []
    for grp, tsub in test.groupby('group'):
        csub = cal[cal.group == grp]
        if len(csub) < 20:      # too little calibration data for this group
            csub = cal          # fall back to pooled (report this when it happens)
        out.append(split_conformal(csub, tsub, alpha))
    return pd.concat(out)


# ---------------------------------------------------------------- evaluation
def evaluate(res, slack_threshold=0.0, target_is_slack=True):
    m = {'n': len(res),
         'coverage': float(res.covered.mean()),
         'mean_width': float((res.hi - res.lo).mean()),
         'median_width': float((res.hi - res.lo).median())}
    if target_is_slack:
        truly_bad = res.y_true < slack_threshold      # actually failing endpoints
        flagged = res.lo < slack_threshold            # interval admits failure
        m['crit_recall'] = float(flagged[truly_bad].mean()) if truly_bad.any() else float('nan')
        m['crit_precision'] = float(truly_bad[flagged].mean()) if flagged.any() else float('nan')
        m['flag_rate'] = float(flagged.mean())
    return m


# -------------------------------------------------------------------- loading
def load_design_csv(csv_dir, design, target='slack', endpoints_only=True):
    path = os.path.join(csv_dir, 'predictions_{}.csv'.format(design))
    df = pd.read_csv(path)
    if endpoints_only and 'is_endpoint' in df:
        df = df[df.is_endpoint == 1]
    if target == 'slack':
        if 'slack_true' not in df:
            raise ValueError('{} has no slack columns (no n_rats labels); '
                             'use --target at0'.format(design))
        out = pd.DataFrame({'y_true': df.slack_true, 'y_pred': df.slack_pred})
    else:                                   # at0..at3: single AT corner
        i = int(target[2:])
        out = pd.DataFrame({'y_true': df['at_true_{}'.format(i)],
                            'y_pred': df['at_pred_{}'.format(i)]})
    out['design'] = design
    # default difficulty signal for v2: |pred| magnitude is a weak but
    # label-free proxy; replace with fanout/depth/residual-model when available.
    out['u'] = np.abs(out.y_pred) + 1e-3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', required=True)
    ap.add_argument('--cal_designs', required=True, help='comma list, NEVER trained on')
    ap.add_argument('--test_designs', required=True, help='comma list')
    ap.add_argument('--target', default='slack', help="'slack' or 'at0'..'at3'")
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--slack_threshold', type=float, default=0.0)
    ap.add_argument('--group_map', default='',
                    help="optional design:group pairs, e.g. 'synth_ram:macro,xtea:logic'")
    args = ap.parse_args()

    cal_names = [s for s in args.cal_designs.split(',') if s]
    test_names = [s for s in args.test_designs.split(',') if s]
    overlap = set(cal_names) & set(test_names)
    assert not overlap, 'designs in both cal and test: {}'.format(overlap)

    cal = pd.concat([load_design_csv(args.csv_dir, d, args.target) for d in cal_names])
    groups = dict(p.split(':') for p in args.group_map.split(',') if p)
    is_slack = args.target == 'slack'

    print('== conformal ({} | alpha={} | cal={} pins from {}) =='
          .format(args.target, args.alpha, len(cal), cal_names))
    header = '{:14} {:>6} {:>9} {:>11} {:>11} {:>11} {:>11}'.format(
        'design', 'n', 'coverage', 'mean_width', 'crit_recall', 'crit_prec', 'flag_rate')

    for method, name in [(split_conformal, 'v1 split'),
                         (weighted_conformal, 'v2 weighted')]:
        print('--- {} ---'.format(name))
        print(header)
        pooled = []
        for d in test_names:
            test = load_design_csv(args.csv_dir, d, args.target)
            res = method(cal, test, args.alpha)
            m = evaluate(res, args.slack_threshold, is_slack)
            pooled.append(res)
            print('{:14} {:>6} {:9.3f} {:11.4f} {:11.3f} {:11.3f} {:11.3f}'.format(
                d, m['n'], m['coverage'], m['mean_width'],
                m.get('crit_recall', float('nan')),
                m.get('crit_precision', float('nan')),
                m.get('flag_rate', float('nan'))))
        m = evaluate(pd.concat(pooled), args.slack_threshold, is_slack)
        print('{:14} {:>6} {:9.3f} {:11.4f}   (pooled; target >= {:.2f})'.format(
            'ALL', m['n'], m['coverage'], m['mean_width'], 1 - args.alpha))

    if groups:
        print('--- v5 mondrian (groups: {}) ---'.format(groups))
        cal['group'] = cal.design.map(lambda d: groups.get(d, 'default'))
        pooled = []
        for d in test_names:
            test = load_design_csv(args.csv_dir, d, args.target)
            test['group'] = groups.get(d, 'default')
            res = mondrian_conformal(cal, test, args.alpha)
            m = evaluate(res, args.slack_threshold, is_slack)
            pooled.append(res)
            print('{:14} {:>6} {:9.3f} {:11.4f}'.format(
                d, m['n'], m['coverage'], m['mean_width']))


if __name__ == '__main__':
    main()
