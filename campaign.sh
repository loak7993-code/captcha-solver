#!/bin/bash
# ~1 hour iterative fine-tuning campaign for the text-CAPTCHA CRNN.
#
# Each round: generate a fresh mixed-difficulty corpus (new seed), fine-tune the
# four current pool members on it (4 parallel jobs), then score them. The policy
# that works: never cold-start (CTC collapses ~60% of the time), always continue
# from a model that already emits characters, and keep the best by hard-tier
# greedy accuracy. Rounds chain, so accuracy compounds.
cd /tmp/opencode/captcha
LOG=campaign.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "campaign start"

# stage the round-2 pool under uniform names so rounds can chain by index
mkdir -p crnn_in
for f in a b c d; do cp -n crnn_pool2/r2_$f.pt crnn_in/$f.pt; done
PREV=crnn_in
ROUNDS="3 4 5"

for R in $ROUNDS; do
  OUT=crnn_round$R
  mkdir -p "$OUT"
  DATA=data/trainmixR$R
  python3 -c "import gen_mixed; gen_mixed.make_mixed('$DATA',3000,seed=${R}001,weights=(0.2,0.35,0.45))" >>"$LOG" 2>&1
  say "round $R: finetune from $PREV on $DATA"
  i=0
  for f in a b c d; do
    i=$((i+1)); s=$((R*10+i))
    OMP_NUM_THREADS=2 python3 -u train_from.py $PREV/$f.pt $OUT/$f.pt $s 14 4e-4 "$DATA" >"$OUT/$f.log" 2>&1 &
  done
  wait
  # quick per-round selection signal: hard-tier greedy accuracy of each member
  python3 - "$OUT" <<'PY' >>"$LOG" 2>&1
import sys, os, json, glob, torch
import crnn
out=sys.argv[1]
labels=json.load(open('data/hard/labels.json')); items=list(labels.items())
for p in sorted(glob.glob(os.path.join(out,'*.pt'))):
    m=crnn.CRNN(); m.load_state_dict(torch.load(p,map_location='cpu')); m.eval()
    ex=sum(1 for fn,t in items if crnn.predict(m, os.path.join('data/hard',fn))==t)
    print(f"  {p}: hard(greedy)={100*ex/len(items):.1f}%")
PY
  say "round $R done"
  PREV=$OUT
done
say "campaign training complete; final pool in $PREV"
